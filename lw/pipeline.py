import numpy as np
import cv2
import json
import os
import time
from pathlib import Path
from . import patterns, correspond, screen, blend

# ... (existing imports and functions above line 165) ...

# AUTO-CELL-SIZE CALCULATION
def calculate_optimal_cell_size(cam_width, cam_height, proj_width, proj_height, min_pixels_per_cell=4, max_cell_size=32):
    """Calcula el cell_size óptimo basado en la relación de resolución.

    Regla: cada celda del proyector debe ocupar entre 4 y 6 píxeles en la cámara.
    Si la cámara ve menos de 3 píxeles por celda, se aumenta el cell_size.
    Si ve más de 8 píxeles por celda, se puede bajar (pero mínimo 8).
    """
    # Píxeles del proyector por píxel de cámara (aprox, asumiendo frontal y sin distorsión)
    proj_per_cam = (proj_width / cam_width + proj_height / cam_height) / 2

    # Píxeles por celda en la cámara
    pixels_per_cell = proj_per_cam * cell_size

    # Ajuste adaptativo
    if pixels_per_cell < min_pixels_per_cell:
        # Cámara muy lejos o proyector muy grande para el encuadre -> aumentar cell_size
        cell_size = max(8, int(cell_size * (min_pixels_per_cell / max(1, pixels_per_cell))))
    elif pixels_per_cell > 8:
        # Cámara cerca o proyector pequeño -> puede bajar cell_size para más detalle
        cell_size = max(8, int(cell_size * (8 / pixels_per_cell)))

    # Ajuste final dentro de límites seguros
    cell_size = max(8, min(max_cell_size, cell_size))
    return cell_size

# ENHANCED PREFlight WITH AUTO-CELL-SIZE ADJUSTMENT
def _preflight_auto_cell(i, black, white, sc, work, log, cam_width, cam_height, proj_width, proj_height):
    """Preflight que también ajusta cell_size si la decodificación es muy baja.
    """
    wt = sc.get("white_threshold", 25)
    lit = (white - black) > wt
    st, verdicts = diagnose.analyze(white, black, lit, lit, wt)

    # Nueva lógica: si valid.mean() < 0.3 (menos del 30% de píxeles válidos), intentamos con cell_size mayor
    if st["valid"].mean() < 0.3:
        log(f"  Proyector {i}: baja decodificación ({st['valid'].mean()*100:.1f}%), intentando cell_size mayor")
        # Guardar el cell_size original para restaurar después
        original_cell = sc.get("cell_size", 16)
        new_cell = min(32, original_cell * 2)
        sc["cell_size"] = new_cell
        # Re-ejecutar la decodificación de blanco/negro con nuevo cell_size (simplificado)
        # En la práctica, se volvería a capturar, pero aquí solo avisamos y guardamos config
        log(f"  Ajustado cell_size de {original_cell} a {new_cell} para reintentar")
        # Guardar el ajuste temporal en config.json
        cfg = json.load(open('config.json'))
        cfg['scan']['cell_size'] = new_cell
        with open('config.json', 'w') as f:
            json.dump(cfg, f, indent=2)
        # Este cambio será permanente; el usuario puede reescanear con el nuevo valor

    # ... (rest of existing preflight logic from line 369) ...

# MODIFIED scan_projector TO AUTO-DETECT CELL-SIZE

def scan_projector(config, projector_index, work=None, log=print, rig=None):
    """Escanea UN proyector (la cámara puede estar sólo sobre él).
    Ahora auto-detecta el cell_size óptimo basado en la resolución de la cámara.
    """
    work = Path(work or config.get("_work", "work"))
    work.mkdir(parents=True, exist_ok=True)
    i = int(projector_index)
    own_rig = rig is None
    if own_rig:
        from lw.devices import RealRig
        rig = RealRig(config)
    try:
        _clean_projector(work, i)
        _log_camera(rig, log)

        # AUTO-CELL-SIZE CALCULATION
        cam_w, cam_h = rig.camera_shape
        proj_w, proj_h = rig.proj_size(i)
        current_cell = config["scan"].get("cell_size", 16)

        # Detectar si estamos usando una webcam vs una cámara de alta resolución
        is_webcam = config.get("camera", {}).get("type") == "webcam"

        # Calcular cell_size óptimo
        if is_webcam:
            # Para webcams, ser más conservadores (mayor cell_size)
            opt_cell = calculate_optimal_cell_size(cam_w, cam_h, proj_w, proj_h, min_pixels_per_cell=6, max_cell_size=32)
        else:
            opt_cell = calculate_optimal_cell_size(cam_w, cam_h, proj_w, proj_h, min_pixels_per_cell=4, max_cell_size=24)

        log(f"  Proyector {i}: cell_size calculado={opt_cell} (anterior={current_cell})")
        config["scan"]["cell_size"] = opt_cell
        # Guardar el cambio inmediatamente
        with open('config.json', 'w') as f:
            json.dump(config, f, indent=2)

        # ... (rest of scan_projector logic from line 262 onward, using updated config["scan"]) ...

        # Save npz with new cell_size
        np.savez_compressed(work / f"scan_proj{i}.npz", width=w, height=h, cell=opt_cell,
                            overlap_left=ov["left"], overlap_right=ov["right"], **dec,
                            **meta)

        log(f"  Proyector {i}: Escaneado con cell_size={opt_cell}")
        return ov
    finally:
        if own_rig:
            rig.close()
