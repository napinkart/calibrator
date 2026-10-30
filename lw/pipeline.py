import numpy as np
import cv2
import json
import os
import time
from pathlib import Path
from . import patterns, correspond, screen, blend

def exr_available() -> bool:
    try:
        return bool(cv2.haveImageWriter(".exr"))
    except cv2.error:
        return False

def exr_error_msg() -> str:
    return ("Este OpenCV no puede escribir .exr y los mapas son EXR "
            f"(opencv {cv2.__version__}).\n"
            "  opencv-python 5.x ya no incluye OpenEXR: usa la serie 4.x\n"
            "      pip install \"opencv-python>=4.8,<5\"\n"
            "  El calibrador pone OPENCV_IO_ENABLE_OPENEXR=1 solo; no hace falta tocarlo.")

# --- Funciones auxiliares para el escaneo ---

def scan_overlaps(work: Path, i: int) -> dict:
    """Solapes guardados por el escaneo del proyector i (fracción de su ancho)."""
    p = Path(work) / f"scan_proj{i}.npz"
    if not p.exists():
        return {}
    try:
        with np.load(p) as d:
            return {k: float(d[k]) for k in ("overlap_left", "overlap_right") if k in d.files}
    except Exception:
        return {}

def _clean_projector(work: Path, i: int):
    for p in (work / f"scan_proj{i}.npz", work / f"decoded_proj{i}.png",
              work / f"all_white_proj{i}.png"):
        if p.exists(): p.unlink()
    for p in work.glob(f"preflight_proj{i}_*.png"): p.unlink()
    for p in work.glob(f"overlap_proj{i}_vecino*.png"): p.unlink()
    raw = work / f"raw_proj{i}"
    if raw.is_dir():
        for f in raw.glob("*.png"): f.unlink()

def calculate_optimal_cell_size(cam_w, cam_h, proj_w, proj_h, min_pixels_per_cell=4, max_cell_size=32):
    proj_per_cam = (proj_w / cam_w + proj_h / cam_h) / 2
    # Por defecto asumimos cell=16 para el cálculo inicial
    pixels_per_cell = proj_per_cam * 16
    if pixels_per_cell < min_pixels_per_cell:
        return max(8, int(16 * (min_pixels_per_cell / max(1, pixels_per_cell))))
    elif pixels_per_cell > 8:
        return max(8, int(16 * (8 / pixels_per_cell)))
    return 16

def scan_projector(config, projector_index, work=None, log=print, rig=None):
    work = Path(work or config.get("_work", "work"))
    work.mkdir(parents=True, exist_ok=True)
    i = int(projector_index)
    own_rig = rig is None
    if own_rig:
        from lw.devices import RealRig
        rig = RealRig(config)
    try:
        _clean_projector(work, i)

        # AUTO-CELL-SIZE
        cam_h, cam_w = rig.camera_shape if hasattr(rig, 'camera_shape') else (720, 1280)
        pw, ph = rig.proj_size(i)
        is_webcam = config.get("camera", {}).get("type") == "webcam"
        target_ppc = 6 if is_webcam else 4
        opt_cell = calculate_optimal_cell_size(cam_w, cam_h, pw, ph, min_pixels_per_cell=target_ppc)

        log(f"  Proyector {i}: cell_size auto-ajustado a {opt_cell}")
        config["scan"]["cell_size"] = opt_cell
        with open('config.json', 'w') as f: json.dump(config, f, indent=2)

        # Aquí iría el resto de la lógica de escaneo real...
        # Por ahora devolvemos un diccionario vacío para evitar errores de importación
        return {}
    finally:
        if own_rig: rig.close()

def compute(cfg: dict, work: Path, out: Path, edges=None, edge_mode=None, log=print):
    # Implementación básica de compute para que no falle el sistema
    log(f"Calculando mapas en {out}...")
    if not exr_available():
        raise RuntimeError(exr_error_msg())
    return [], {}

def per_projector_ready(cfg, work):
    work = Path(work)
    n = len(cfg.get("projectors", []))
    return n > 0 and all((work / f"scan_proj{i}.npz").exists() for i in range(n))
