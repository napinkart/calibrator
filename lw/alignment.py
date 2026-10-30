"""
Módulo de alineación de mapas de proyectores usando Thin Plate Spline (TPS).

Proporciona funciones para:
1. Extraer puntos de control de los bordes marcados.
2. Calcular transformaciones TPS entre proyectores adyacentes.
3. Transformar mapas locales a coordenadas globales.
4. Fusionar mapas en un mapa unificado.

Autor: Proyecto LibreWarp
"""

import json
import numpy as np
from scipy.interpolate import Rbf


def combine_screen_json(screen_files):
    """
    Combina múltiples screen.json en uno solo.

    Args:
        screen_files: Lista de rutas de archivos de screen.json.

    Returns:
        Un diccionario con las "edges" combinadas.
    """
    combined = {"top": [], "right": [], "bottom": [], "left": []}
    for file in screen_files:
        try:
            with open(file, "r") as f:
                data = json.load(f)
                for edge in ["top", "right", "bottom", "left"]:
                    combined[edge].extend(data.get("edges", {}).get(edge, []))
        except Exception as e:
            raise RuntimeError(f"Error al leer {file}: {e}") from e
    return {"edge_param": "arclength", "edges": combined}


def extract_control_points(screen_json, projectors):
    """
    Extrae puntos de control para alineación entre proyectores adyacentes.

    Args:
        screen_json: Diccionario con los bordes marcados.
        projectors: Lista de diccionarios de proyectores.

    Returns:
        Lista de diccionarios con puntos correspondientes entre proyectores adyacentes.
    """
    correspondences = []
    edges = screen_json.get("edges", {})

    for i in range(len(projectors) - 1):
        right_i = np.array(edges.get("right", []))
        left_j = np.array(edges.get("left", []))

        matched = []
        if len(right_i) > 0 and len(left_j) > 0:
            for p_i in right_i:
                for p_j in left_j:
                    if np.linalg.norm(p_i - p_j) < 5:
                        matched.append((p_i, p_j))

        if matched:
            correspondences.append({
                "proj0_points": np.array([p[0] for p in matched]),
                "proj1_points": np.array([p[1] for p in matched])
            })

    return correspondences


def compute_tps(src_points, dst_points):
    """
    Calcula una transformación TPS (Thin Plate Spline) entre dos conjuntos de puntos.
    """
    if len(src_points) < 3:
        from cv2 import findHomography
        H, _ = findHomography(src_points.astype(np.float32), dst_points.astype(np.float32), cv2.RANSAC, 5.0)
        return lambda u, v: np.dot(H, [u, v, 1])[:2]

    u_src, v_src = src_points[:, 0], src_points[:, 1]
    U_dst, V_dst = dst_points[:, 0], dst_points[:, 1]

    rbf_u = Rbf(u_src, v_src, U_dst, function='thin_plate')
    rbf_v = Rbf(u_src, v_src, V_dst, function='thin_plate')

    def transform(u, v):
        U = float(rbf_u(u, v))
        V = float(rbf_v(u, v))
        return np.array([U, V])

    return transform


def compute_tps_alignment(correspondences):
    """
    Calcula transformaciones TPS para todos los pares de proyectores.
    """
    transforms = {}
    for i, corr in enumerate(correspondences):
        transforms[i] = compute_tps(corr["proj0_points"], corr["proj1_points"])
    return transforms


def transform_map_with_tps(map_local, transform, output_width, output_height):
    """
    Transforma un mapa local a coordenadas globales usando TPS.
    """
    global_map = np.zeros((output_height, output_width, 3), dtype=map_local.dtype)
    for v in range(map_local.shape[0]):
        for u in range(map_local.shape[1]):
            U, V = transform(u, v)
            ui, vi = int(round(U)), int(round(V))
            if 0 <= ui < output_width and 0 <= vi < output_height:
                global_map[vi, ui] = map_local[v, u]
    return global_map


def fuse_maps_with_tps(projector_maps, tps_transforms, config):
    """
    Fusiona múltiples mapas usando TPS para alineación.
    """
    total_width = sum(int(p.get("width", 1920)) for p in config.get("projectors", []))
    max_height = max(int(p.get("height", 1080)) for p in config.get("projectors", []))
    unified_map = np.zeros((max_height, total_width, 3), dtype=np.float32)

    for map_local, transform in zip(projector_maps, tps_transforms.values()):
        map_global = transform_map_with_tps(map_local, transform, total_width, max_height)
        unified_map += map_global

    num_proj = len(projector_maps)
    if num_proj > 0:
        unified_map /= num_proj

    return unified_map


def compute_unified_with_tps(config, work_dir, output_dir):
    """
    Calcula un mapa unificado usando TPS para alinear mapas de proyectores individuales.
    """
    scans = sorted(work_dir.glob("scan_proj*.npz"))
    if not scans:
        raise RuntimeError(f"No hay escaneos en {work_dir}. Ejecuta primero 'scan'.")
    projector_maps = [dict(np.load(scan)) for scan in scans]

    screen_json = work_dir / "screen.json"
    if not screen_json.exists():
        raise RuntimeError("Falta work/screen.json. Marca los bordes primero.")
    with open(screen_json) as f:
        screen_data = json.load(f)

    correspondences = extract_control_points(screen_data, config.get("projectors", []))
    tps_transforms = compute_tps_alignment(correspondences)

    unified_map = fuse_maps_with_tps(projector_maps, tps_transforms, config)

    import cv2
    output_path = output_dir / "unified_map.exr"
    ok = cv2.imwrite(str(output_path), unified_map)
    if not ok:
        raise RuntimeError("Fallo al escribir unified_map.exr")

    print(f"✅ Mapa unificado generado en {output_path}")