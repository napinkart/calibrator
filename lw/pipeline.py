"""Pasos del pipeline: escanear, calcular mapas, exportar, verificar."""
from __future__ import annotations

import os
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import cv2
from scipy import ndimage

from . import patterns, correspond, screen, blend


def exr_available() -> bool:
    """¿Este OpenCV puede escribir .exr? Los mapas de salida son EXR."""
    try:
        return bool(cv2.haveImageWriter(".exr"))
    except cv2.error:      # el códec está compilado pero deshabilitado por la variable de entorno
        return False


def exr_error_msg() -> str:
    return ("Este OpenCV no puede escribir .exr y los mapas son EXR "
            f"(opencv {cv2.__version__}).\n"
            "  opencv-python 5.x ya no incluye OpenEXR: usa la serie 4.x\n"
            "      pip install \"opencv-python>=4.8,<5\"\n"
            "  El calibrador pone OPENCV_IO_ENABLE_OPENEXR=1 solo; no hace falta tocarlo.")


# --------------------------------------------------------------------------
# 1. ESCANEO
# --------------------------------------------------------------------------
def _camera_meta(rig) -> dict:
    """Exposición/ganancia con las que se capturó, para guardarlas en el .npz."""
    fn = getattr(rig, "camera_state", None)
    st = (fn() if callable(fn) else None) or {}
    act = st.get("actual") or {}
    return {"exposure": float(act.get("exposure", float("nan"))),
            "gain": float(act.get("gain", float("nan"))),
            "scan_unix_time": float(time.time())}


def _log_camera(rig, log=print) -> None:
    """Dice qué exposición usa de verdad el escaneo (y si pisa la del diálogo)."""
    fn = getattr(rig, "camera_state", None)
    st = (fn() if callable(fn) else None) or {}
    act = st.get("actual") or {}
    if not act:
        return
    log(f"  cámara: exposición {act.get('exposure', float('nan')):.1f} (log2) · "
        f"ganancia {act.get('gain', float('nan')):.0f} · "
        f"autoexp. {act.get('auto_exposure', float('nan')):.2f}")
    if st.get("overwritten"):
        k = st["overwritten"][0]
        log(f"  AVISO: config.json fuerza {k}={st['requested'][k]:g} y el driver traía "
            f"{st['before'][k]:g} (lo del diálogo ⚙️): este escaneo usa "
            f"{st['requested'][k]:g}. Si querías otra exposición, guárdala en config.json.")
    if st.get("ignored"):
        k = st["ignored"][0]
        log(f"  AVISO: el driver no aplicó {k}={st['requested'][k]:g} (ahora está en "
            f"{act.get(k, float('nan')):g}).")


def _fingerprint(dec: dict) -> str:
    """Huella de la captura en blanco (submuestreada): igual = la cámara vio lo mismo."""
    w = np.clip(np.asarray(dec["white"], np.float32), 0, 255).astype(np.uint8)
    return hashlib.md5(w[::4, ::4].tobytes()).hexdigest()


def scan_history(work: Path) -> dict:
    """Últimos escaneos por proyector: {"0": [registro, ...], ...}."""
    p = Path(work) / "scan_history.json"
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def repeat_warning(prev: dict | None, cur: dict | None, i: int) -> str:
    """Aviso si dos escaneos seguidos del proyector i son iguales."""
    def _num(v) -> float:
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("nan")

    if not prev or not cur:
        return ""
    e0, e1 = _num(prev.get("exposure")), _num(cur.get("exposure"))
    w0, w1 = _num(prev.get("white")), _num(cur.get("white"))
    when = time.strftime("%H:%M:%S", time.localtime(_num(prev.get("time")) or 0))
    exp_ok = e1 == e1
    exp_bit = (f"la **misma exposición** ({e1:.0f})" if exp_ok
               else "la exposición no se pudo leer")
    if prev.get("fingerprint") and prev.get("fingerprint") == cur.get("fingerprint"):
        return (f"⚠️ El escaneo del proyector {i} es **idéntico** al de las {when} "
                f"(brillo {w1:.0f}): la cámara capturó exactamente lo mismo.")
    same_exp = ((e0 != e0 and e1 != e1)
                or (e0 == e0 and e1 == e1 and abs(e0 - e1) <= 0.5))
    same_brt = (w0 == w0 and w1 == w1
                and abs(w0 - w1) <= max(1.0, 0.02 * max(w0, 1.0)))
    if not (same_exp and same_brt):
        return ""
    head = (f"con {exp_bit} y el **mismo brillo** ({w1:.0f})" if exp_ok
            else f"con el **mismo brillo** ({w1:.0f}) y la exposición no se pudo leer")
    return (f"⚠️ Este escaneo del proyector {i} y el anterior (de las {when}) capturaron "
            f"{head}: la segunda pasada no cambió nada.")


def _record_scan(work: Path, i: int, dec: dict, meta: dict, log=print) -> None:
    work = Path(work)
    hist = scan_history(work)
    def _opt(v):
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None
        return v if v == v else None

    rec = {"time": float(meta.get("scan_unix_time") or time.time()),
           "exposure": _opt(meta.get("exposure")),
           "gain": _opt(meta.get("gain")),
           "white": float(np.asarray(dec["white"], np.float32).mean()),
           "black": float(np.asarray(dec["black"], np.float32).mean()),
           "valid": float(np.asarray(dec["valid"], bool).mean() * 100),
           "fingerprint": _fingerprint(dec)}
    key = str(int(i))
    prev = (hist.get(key) or [None])[-1]
    hist[key] = (hist.get(key) or [])[-4:] + [rec]
    try:
        (work / "scan_history.json").write_text(
            json.dumps(hist, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError:
        return
    msg = repeat_warning(prev, rec, int(i))
    if msg:
        log("\n  " + msg.replace("**", "").replace("\n", "\n  "))


def _capture_sequence(rig, cfg: dict, i: int, work: Path, log=print) -> dict:
    sc = cfg["scan"]
    cell = sc.get("cell_size", 16)
    w, h = rig.proj_size(i)
    phase_steps = sc.get("phase_steps", 8)
    n_pat = patterns.sequence_length(w, h, cell, phase_steps)
    decoder = patterns.Decoder(w, h, cell, sc.get("min_contrast", 10),
                               sc.get("white_threshold", 25), sc.get("phase_period", 32),
                               phase_steps)
    raw = None
    if sc.get("save_raw", False):
        raw = work / f"raw_proj{i}"
        if raw.exists():
            for f in raw.glob("*.png"): f.unlink()
        raw.mkdir(exist_ok=True)
    t0 = time.time()
    settle = cfg.get("camera", {}).get("settle_ms", 500) / 1000.0
    for k, (name, img) in enumerate(patterns.make_sequence(
            w, h, cell, sc.get("pattern_intensity", 255), phase_steps,
            sc.get("phase_period", 32), False)):
        rig.show_only(i, img)
        time.sleep(settle)
        cap = rig.capture()
        if raw is not None:
            cv2.imwrite(str(raw / f"{name}.png"),
                        np.clip(cap, 0, 255).astype(np.uint8))
        decoder.add(name, cap)
        if name == "white" and not sc.get("skip_preflight", False):
            _preflight(i, decoder.black, decoder.white, sc, work, log)
        log(f"  proyector {i}: patrón {k+1}/{n_pat} ({name})", end="\r")
    log(f"  proyector {i}: {n_pat} patrones capturados en {time.time()-t0:.1f}s     ")
    if raw is not None:
        sheet = contact_sheet(raw)
        if sheet is not None:
            cv2.imwrite(str(work / f"captures_proj{i}.png"), sheet)
    dec = decoder.result()
    cv2.imwrite(str(work / f"decoded_proj{i}.png"), decoded_preview(dec, w, h, cell))
    log(f"  proyector {i}: {dec['valid'].mean()*100:.1f}% de píxeles de cámara decodificados")
    return dec


def scan(rig, cfg: dict, work: Path, log=print):
    work.mkdir(parents=True, exist_ok=True)
    for old in (list(work.glob("scan_proj*.npz")) + list(work.glob("decoded_proj*.png"))
                + list(work.glob("captures_proj*.png"))
                + list(work.glob("all_white_proj*.png"))
                + list(work.glob("overlap_proj*_vecino*.png"))):
        old.unlink()
    rig.show_all([np.full((rig.proj_size(i)[1], rig.proj_size(i)[0]), 255, np.uint8)
                  for i in range(rig.n_proj)])
    all_white = rig.capture()
    cv2.imwrite(str(work / "all_white.png"), np.clip(all_white, 0, 255).astype(np.uint8))

    cell = cfg["scan"].get("cell_size", 16)
    _log_camera(rig, log)
    for i in range(rig.n_proj):
        dec = _capture_sequence(rig, cfg, i, work, log)
        w, h = rig.proj_size(i)
        meta = _camera_meta(rig)
        np.savez_compressed(work / f"scan_proj{i}.npz", width=w, height=h, cell=cell,
                            **dec, **meta)
        _record_scan(work, i, dec, meta, log)


def calculate_optimal_cell_size(cam_w, cam_h, proj_w, proj_h, current_cell=16, min_pixels_per_cell=5, max_cell_size=32):
    """Calcula el cell_size del proyector para garantizar suficientes píxeles en cámara.

    Si cada celda proyectada se ve de menos de `min_pixels_per_cell` píxeles en la cámara,
    aumenta el cell_size para que los patrones sean nítidos y decodificables.
    """
    cam_w, cam_h = max(1, int(cam_w)), max(1, int(cam_h))
    proj_w, proj_h = max(1, int(proj_w)), max(1, int(proj_h))

    # Cuántos píxeles de cámara ocupa 1 celda de proyector
    cam_px_per_cell = float(current_cell) * (cam_w / float(proj_w))

    if cam_px_per_cell < min_pixels_per_cell:
        # La celda es demasiado fina en la cámara: aumentar cell_size
        scale = min_pixels_per_cell / max(0.1, cam_px_per_cell)
        new_cell = int(np.ceil(current_cell * scale))
    else:
        new_cell = current_cell

    return max(8, min(max_cell_size, int(new_cell)))


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
        _log_camera(rig, log)

        # AUTO-CELL-SIZE
        cam_h, cam_w = rig.camera_shape if hasattr(rig, 'camera_shape') else (720, 1280)
        pw, ph = rig.proj_size(i)
        old_cell = config.get("scan", {}).get("cell_size", 16)
        is_webcam = config.get("camera", {}).get("type") == "webcam"
        target_ppc = 6 if is_webcam else 4
        opt_cell = calculate_optimal_cell_size(cam_w, cam_h, pw, ph, current_cell=old_cell, min_pixels_per_cell=target_ppc)

        if opt_cell != old_cell:
            log(f"  Proyector {i}: cell_size auto-ajustado a {opt_cell} (antes {old_cell})")
            config.setdefault("scan", {})["cell_size"] = opt_cell

            # Guardar en config.json filtrando claves internas (_path, _work, etc.)
            cfg_path = Path(config.get("_path", "config.json"))
            clean_cfg = {k: v for k, v in config.items() if not str(k).startswith("_")}
            cfg_path.write_text(json.dumps(clean_cfg, indent=2, ensure_ascii=False), encoding="utf-8")

        cell = opt_cell
        w, h = rig.proj_size(i)
        rig.show_only(i, np.full((h, w), 255, np.uint8))
        white = rig.capture()
        cv2.imwrite(str(work / f"all_white_proj{i}.png"),
                    np.clip(white, 0, 255).astype(np.uint8))
        dec = _capture_sequence(rig, config, i, work, log)
        ov = {}
        for side, j in (("left", i - 1), ("right", i + 1)):
            ov[side] = float("nan")
            if 0 <= j < rig.n_proj:
                ov[side] = _measure_overlap(rig, config, i, j, dec, work, side, log)
        meta = _camera_meta(rig)
        np.savez_compressed(work / f"scan_proj{i}.npz", width=w, height=h, cell=cell,
                            overlap_left=ov["left"], overlap_right=ov["right"], **dec,
                            **meta)
        _record_scan(work, i, dec, meta, log)
        rig.show_only(i, np.zeros((h, w), np.uint8))
        return ov
    finally:
        if own_rig: rig.close()


def _clean_projector(work: Path, i: int):
    for p in (work / f"scan_proj{i}.npz", work / f"decoded_proj{i}.png",
              work / f"captures_proj{i}.png", work / f"all_white_proj{i}.png"):
        if p.exists(): p.unlink()
    for p in work.glob(f"preflight_proj{i}_*.png"): p.unlink()
    for p in work.glob(f"overlap_proj{i}_vecino*.png"): p.unlink()
    raw = work / f"raw_proj{i}"
    if raw.is_dir():
        for f in raw.glob("*.png"): f.unlink()


def scan_overlaps(work: Path, i: int) -> dict:
    p = Path(work) / f"scan_proj{i}.npz"
    if not p.exists(): return {}
    try:
        with np.load(p) as d:
            return {k: float(d[k]) for k in ("overlap_left", "overlap_right") if k in d.files}
    except Exception: return {}


def _measure_overlap(rig, cfg: dict, i: int, j: int, dec: dict, work: Path,
                     side: str, log=print) -> float:
    wj, hj = rig.proj_size(j)
    rig.show_only(j, np.full((hj, wj), 255, np.uint8))
    nb = rig.capture().astype(np.float32)
    cv2.imwrite(str(work / f"overlap_proj{i}_vecino{j}.png"),
                np.clip(nb, 0, 255).astype(np.uint8))
    sc = cfg["scan"]
    wt = float(sc.get("white_threshold", 25))
    lit_nb = (nb - dec["black"].astype(np.float32)) > max(wt, 10.0)
    w_i, h_i = rig.proj_size(i)
    cell = int(sc.get("cell_size", 16))
    pmap, pvalid, _ = correspond.projector_to_camera(
        dec, w_i, h_i, cell, cfg.get("mapping", {}).get("min_pixels_per_cell", 2),
        cfg.get("mapping", {}).get("smoothing", 1.0))
    on = cv2.remap(lit_nb.astype(np.float32), pmap[..., 0], pmap[..., 1],
                   cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0.5
    cam_h, cam_w = lit_nb.shape
    inb = ((pmap[..., 0] >= 0) & (pmap[..., 0] <= cam_w - 1) &
           (pmap[..., 1] >= 0) & (pmap[..., 1] <= cam_h - 1))
    good = pvalid & inb
    cols = good.any(axis=0)
    if cols.sum() < 10: return float("nan")
    frac = (on & good).sum(axis=0) / np.maximum(good.sum(axis=0), 1)
    lit_cols = cols & (frac > 0.5)
    idx = np.nonzero(cols)[0]
    edge = idx[-1] if side == "right" else idx[0]
    span, k = 0, edge
    while 0 <= k < w_i and lit_cols[k]:
        span += 1
        k += -1 if side == "right" else 1
    o = float(np.clip(span / float(len(idx)), 0.0, 0.49))
    log(f"  proyector {i}: solape {side} con el proyector {j} = {o*100:.1f}%")
    return o


def _preflight(i, black, white, sc, work, log):
    from . import diagnose
    wt = sc.get("white_threshold", 25)
    lit = (white - black) > wt
    st, verdicts = diagnose.analyze(white, black, lit, lit, wt)
    cv2.imwrite(str(work / f"preflight_proj{i}_negro.png"), np.clip(black, 0, 255).astype(np.uint8))
    cv2.imwrite(str(work / f"preflight_proj{i}_blanco.png"), np.clip(white, 0, 255).astype(np.uint8))
    fatal = [m for sev, m in verdicts if sev == 3]
    if st["lit"] < 1.0:
        fatal.append(f"El proyector sólo ilumina el {st['lit']:.2f}% del encuadre.")
    if fatal:
        raise RuntimeError(f"Comprobación previa falló.\n" + "\n".join(fatal))


def contact_sheet(folder: Path, cols: int = 8, thumb_w: int = 240) -> np.ndarray | None:
    files = sorted(Path(folder).glob("*.png"))
    if not files: return None
    tiles = []
    for f in files:
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        s = thumb_w / max(1, img.shape[1])
        t = cv2.resize(img, (thumb_w, max(1, int(img.shape[0] * s))), interpolation=cv2.INTER_AREA)
        t = cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)
        cv2.putText(t, f.stem, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        tiles.append(t)
    if not tiles: return None
    th, tw = max(t.shape[0] for t in tiles), max(t.shape[1] for t in tiles)
    while len(tiles) % cols: tiles.append(np.zeros((th, tw, 3), np.uint8))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    return np.vstack(rows)


def decoded_preview(dec, w, h, cell):
    gw, gh = patterns.grid_size(w, h, cell)
    hsv = np.zeros(dec["valid"].shape + (3,), np.uint8)
    hsv[..., 0] = (dec["cell_x"].clip(0) * 179 // max(gw, 1)).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.where(dec["valid"], 80 + (dec["cell_y"].clip(0) * 175 // max(gh, 1)), 0).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def per_projector_scans(cfg: dict, work: Path) -> bool:
    work = Path(work)
    n = len(cfg.get("projectors", []))
    return n > 0 and any((work / f"all_white_proj{i}.png").exists() for i in range(n))


# --------------------------------------------------------------------------
# RELLENO DE HUECOS (para no entregar mapas con agujeros del sistema)
# --------------------------------------------------------------------------
def _solid_mask(mask: np.ndarray, close_px: int = 3,
                max_hole_frac: float = 0.02) -> np.ndarray:
    """Devuelve la máscara SIN huecos interiores pequeños.

    El decodificado y los filtros del mapeo dejan píxeles sueltos y agujeros
    de pocos píxeles (sobre todo en los bordes). Proyectados se ven como
    puntos o rayas negras en la pared, y no vienen de la pared sino del
    cálculo. Aquí se cierran: sólo huecos INTERIORES (que no tocan el borde
    de la imagen) y pequeños (< max_hole_frac del área), para no inventar luz
    donde de verdad no llega (sombras, oclusiones, un borde de la sala).
    """
    m = np.asarray(mask, bool)
    if not m.any():
        return m
    k = max(1, int(close_px)) | 1                    # impar
    closed = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE,
                              np.ones((k, k), np.uint8)) > 0
    inv = (~closed).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(inv, 4)
    H, W = closed.shape
    out = closed.copy()
    lim = max(64.0, float(max_hole_frac) * float(m.sum()))
    for j in range(1, n):
        x, y, w, h, area = st[j]
        if x == 0 or y == 0 or x + w >= W or y + h >= H:
            continue                                 # abierto al borde: no es hueco
        if area <= lim:
            out |= (lab == j)
    return out


def _fill_nearest(img: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Rellena cada píxel no válido con el valor válido MÁS CERCANO."""
    v = np.asarray(valid, bool)
    out = np.asarray(img, np.float32)
    if v.all():
        return out
    if not v.any():
        return np.zeros_like(out, np.float32)
    idx = ndimage.distance_transform_edt(~v, return_distances=False,
                                         return_indices=True)
    return out[tuple(idx)].astype(np.float32)


def _sample_map(img, x, y, border: float = 0.0) -> np.ndarray:
    """Muestrea un mapa (lienzo o cámara) en coordenadas de píxel (x, y)."""
    return cv2.remap(np.ascontiguousarray(img, np.float32),
                     np.ascontiguousarray(x, np.float32),
                     np.ascontiguousarray(y, np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                     borderValue=border)


def _grow_mask(mask, px: int = 2) -> np.ndarray:
    k = 2 * int(px) + 1
    return cv2.dilate(np.ascontiguousarray(mask, np.uint8),
                      np.ones((k, k), np.uint8)) > 0


def _blend_weights(cfg: dict, masks: list):
    """Pesos de mezcla que suman 1 en TODA la zona cubierta.

    Las máscaras se engordan 2 px SÓLO para calcular el peso: si no, el
    muestreo bilineal en el borde exacto de la huella devuelve 0 y el mapa
    sale con un filo negro de 1 px (un «hueco» del cálculo, no de la pared).
    """
    wm = [_grow_mask(m, 2) for m in masks]
    weights = blend.normalize_weights(blend.camera_weights(wm), wm)
    narrow = float((cfg.get("blend") or {}).get("narrow") or 1.0)
    if narrow > 1.0:
        weights = blend.normalize_weights(blend.narrow_weights(weights, narrow), wm)
    return weights, wm


def _aspect_hint(edges: dict) -> float | None:
    """Aspecto de la pantalla marcada, medido en la imagen de cámara."""
    try:
        def L(k):
            q = np.asarray(edges[k], np.float64)
            return float(np.sum(np.hypot(*np.diff(q, axis=0).T)))
        wdt = 0.5 * (L("top") + L("bottom"))
        hgt = 0.5 * (L("left") + L("right"))
        return round(wdt / hgt, 3) if hgt > 1e-6 else None
    except Exception:
        return None


def _max_overlap(masks: list) -> float:
    """Mayor solape entre proyectores vecinos (fracción del más pequeño)."""
    best = 0.0
    for a, b in zip(masks, masks[1:]):
        inter = float(np.logical_and(a, b).sum())
        small = float(min(a.sum(), b.sum()))
        if small > 0:
            best = max(best, inter / small)
    return round(best, 4)


PROJ_COLORS = [(80, 80, 255), (80, 200, 80), (255, 160, 60), (255, 80, 200),
               (80, 220, 220), (200, 120, 255)]


def _write_exr(out: Path, i: int, w, v, u, lift, gain) -> None:
    """Escribe el mapa del proyector (B=w, G=v, B=u) y su mapa de negro."""
    one = np.ones_like(u, np.float32)
    cv2.imwrite(str(Path(out) / f"proj{i}_map.exr"),
                np.dstack([w, v, u, one]).astype(np.float32))
    cv2.imwrite(str(Path(out) / f"proj{i}_black.exr"),
                np.dstack([np.zeros_like(u, np.float32), gain, lift, one]).astype(np.float32))


def _projector_entries(cfg: dict, results: list, windows=None) -> list:
    out = []
    for i, p in enumerate(cfg["projectors"]):
        ent = {"index": i, "name": p.get("name", f"proyector{i}"),
               "width": int(p["width"]), "height": int(p["height"]),
               "desktop_x": int(p.get("desktop_x", 0)),
               "desktop_y": int(p.get("desktop_y", 0)),
               "file": f"proj{i}_map.exr", "black_file": f"proj{i}_black.exr"}
        if windows is not None and i < len(windows):
            ent["content_window"] = [round(float(windows[i][0]), 6),
                                     round(float(windows[i][1]), 6)]
        else:
            ent["content_window"] = [0.0, 1.0]
        if i < len(results) and results[i] is not None:
            ent["valid_pct"] = round(float(np.mean(results[i]["valid"])) * 100, 1)
        out.append(ent)
    return out


def _write_calibration(cfg: dict, out: Path, mode: str, results: list, *,
                       windows=None, canvas=None, aspect=None, max_overlap=None,
                       uniformity=None, warnings=None, log=print) -> dict:
    meta = {"version": 2,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "mode": mode,
            "aspect_hint_from_camera": aspect,
            "max_overlap": max_overlap,
            "uniformity": uniformity or {},
            "displays": cfg.get("displays", []),
            "projectors": _projector_entries(cfg, results, windows)}
    if windows is not None:
        meta["content_windows"] = [[round(float(a), 6), round(float(b), 6)]
                                  for a, b in windows]
    if canvas:
        meta["canvas"] = canvas
    if warnings:
        meta["warnings"] = list(warnings)
    (Path(out) / "calibration.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"  calibración: {Path(out) / 'calibration.json'}")
    return meta


def _marked_image(img_path: Path, edges: dict, mode: str, title: str):
    """La imagen donde se marcaron los puntos, ENTERA, con los puntos y la
    rejilla del lienzo que se va a usar (para comprobar que el marcado cubre
    toda la pantalla)."""
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None or not edges:
        return None
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    try:
        t = np.linspace(0.0, 1.0, 21)
        for a in t[1:-1]:
            for P in (screen.coons(edges, np.full_like(t, a), t, mode),
                      screen.coons(edges, t, np.full_like(t, a), mode)):
                cv2.polylines(vis, [np.rint(P).astype(np.int32).reshape(-1, 1, 2)],
                              False, (170, 170, 170), 1, cv2.LINE_AA)
    except Exception:
        pass
    colors = {"top": (0, 0, 255), "right": (0, 210, 0),
              "bottom": (255, 110, 0), "left": (255, 0, 255)}
    for k, pts in edges.items():
        q = np.rint(np.asarray(pts, np.float64)).astype(np.int32)
        if len(q) >= 2:
            cv2.polylines(vis, [q.reshape(-1, 1, 2)], False,
                          colors.get(k, (255, 255, 255)), 2, cv2.LINE_AA)
        for p in q:
            cv2.circle(vis, (int(p[0]), int(p[1])), 4,
                       colors.get(k, (255, 255, 255)), -1, cv2.LINE_AA)
            cv2.circle(vis, (int(p[0]), int(p[1])), 5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(vis, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(vis, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def _overview_camera(work: Path, pmaps: list, out: Path):
    """Vista de cámara ENTERA con la huella de cada proyector."""
    img = None
    for name in ("all_white.png",):
        p = Path(work) / name
        if p.exists():
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            break
    if img is None:
        return None
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for i, pm in enumerate(pmaps):
        col = PROJ_COLORS[i % len(PROJ_COLORS)]
        pmap = pm[0] if isinstance(pm, tuple) else pm
        poly = np.concatenate([pmap[0], pmap[:, -1], pmap[-1][::-1],
                               pmap[:, 0][::-1]])
        poly = np.rint(poly).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [poly], True, col, 2, cv2.LINE_AA)
        c = poly.reshape(-1, 2).mean(0).astype(int)
        for th, cc in ((3, (0, 0, 0)), (1, col)):
            cv2.putText(vis, f"P{i}", (int(c[0]), int(c[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, cc, th, cv2.LINE_AA)
    cv2.imwrite(str(Path(out) / "camera_overview.png"), vis)
    return vis


def _overview_canvas(geo: dict, out: Path):
    """Lienzo de contenido: dónde cae cada proyector y por dónde van las
    costuras. Es el espacio que ve TouchDesigner como una sola imagen."""
    Hc, Wc = geo["Hc"], geo["Wc"]
    vis = np.full((Hc, Wc, 3), 35, np.uint8)
    for i, m in enumerate(geo["masks"]):
        col = np.asarray(PROJ_COLORS[i % len(PROJ_COLORS)], np.float32)
        vis[m] = (0.72 * vis[m].astype(np.float32) + 0.28 * col).astype(np.uint8)
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, tuple(int(c) for c in col), 2)
        ys, xs = np.nonzero(m)
        if len(xs):
            x, y = int(np.median(xs)), int(np.median(ys))
            for th, cc in ((3, (0, 0, 0)), (1, tuple(int(c) for c in col))):
                cv2.putText(vis, f"P{i}", (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, cc, th, cv2.LINE_AA)
    for k in range(len(geo["wins"]) - 1):
        x = int(round(geo["wins"][k][1] * Wc))
        cv2.line(vis, (x, 0), (x, Hc - 1), (255, 255, 255), 1, cv2.LINE_AA)
        lbl = f"P{k}-P{k+1} {geo['overlaps'][k] * 100:.0f}%"
        for th, cc in ((3, (0, 0, 0)), (1, (255, 255, 255))):
            cv2.putText(vis, lbl, (max(2, x + 6), 26), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, cc, th, cv2.LINE_AA)
    cv2.imwrite(str(Path(out) / "content_overview.png"), vis)
    return vis


def compute(cfg: dict, work: Path, out: Path, edges=None, edge_mode=None, log=print):
    if edges is None and per_projector_scans(cfg, work):
        return compute_per_projector(cfg, work, out, log=log)
    if not exr_available(): raise RuntimeError(exr_error_msg())
    out.mkdir(parents=True, exist_ok=True)
    scans = sorted(work.glob("scan_proj*.npz"), key=lambda p: int(p.stem[9:]))
    if not scans: raise RuntimeError(f"No hay escaneos en {work}.")
    if edges is None: edges, edge_mode = screen.load_edges(work / "screen.json")
    edge_mode = edge_mode or cfg.get("screen", {}).get("edge_param", "arclength")
    decs = [dict(np.load(p)) for p in scans]
    cam_h, cam_w = decs[0]["valid"].shape
    cam_uv = screen.camera_uv(edges, cam_w, cam_h, edge_mode)
    screen_mask = np.all((cam_uv >= 0) & (cam_uv <= 1), axis=-1)

    maps = []
    for i, d in enumerate(decs):
        w, h, cell = int(d["width"]), int(d["height"]), int(d["cell"])
        pmap, pvalid, st = correspond.projector_to_camera(
            d, w, h, cell,
            cfg.get("mapping", {}).get("min_pixels_per_cell", 2),
            cfg.get("mapping", {}).get("smoothing", 1.0))
        maps.append((pmap, pvalid, st))

    masks = [blend.footprint(pm, (cam_h, cam_w)) for (pm, _, _), d in zip(maps, decs)]
    weights, _wmasks = _blend_weights(cfg, masks)
    lift, nmax = blend.black_lift(masks, screen_mask)
    gain, ustats = blend.uniformity_gain([d["white"] for d in decs],
                                        [d["black"] for d in decs],
                                        masks, weights, screen_mask)

    ncov = np.sum([m.astype(np.float32) for m in masks], axis=0)
    results = []
    for i, (d, (pmap, pvalid, st)) in enumerate(zip(decs, maps)):
        uv = _sample_map(np.nan_to_num(cam_uv, nan=-10.0), pmap[..., 0], pmap[..., 1],
                         border=-10.0)
        # zona que este proyector DEBE iluminar: su huella ∩ la pantalla marcada
        inside = (pvalid & np.all(np.isfinite(uv), -1)
                  & (uv[..., 0] >= -0.01) & (uv[..., 0] <= 1.01)
                  & (uv[..., 1] >= -0.01) & (uv[..., 1] <= 1.01))
        solid = _solid_mask(inside) & pvalid            # sin huecos del sistema
        u = np.where(solid, np.clip(uv[..., 0], 0, 1), -1.0).astype(np.float32)
        v = np.where(solid, 1.0 - np.clip(uv[..., 1], 0, 1), -1.0).astype(np.float32)
        know = solid & (u >= 0)
        if know.any() and not know.all():                # hueco sin coordenada
            uf = _fill_nearest(np.where(know, u, 0.0), know)
            vf = _fill_nearest(np.where(know, v, 0.0), know)
            u = np.where(solid, uf, -1.0).astype(np.float32)
            v = np.where(solid, vf, -1.0).astype(np.float32)

        x, y = pmap[..., 0], pmap[..., 1]
        wgt = np.where(solid, np.clip(_sample_map(weights[i], x, y), 0, 1), 0.0).astype(np.float32)
        lf = np.where(solid, _sample_map(lift, x, y), 0.0).astype(np.float32)
        gn = np.where(solid, _sample_map(gain, x, y, border=1.0), 1.0).astype(np.float32)
        # si aun así quedara un píxel a 0 dentro de la pantalla Y nadie más lo
        # cubre, que ilumine: antes un píxel con luz que un agujero negro
        gap = solid & (wgt <= 1e-3) & (_sample_map(ncov, x, y) < 1.5)
        if gap.any():
            wf = _fill_nearest(np.where(gap, 0.0, wgt), ~gap & solid)
            wgt = np.where(gap, np.maximum(wgt, wf), wgt).astype(np.float32)

        _write_exr(out, i, wgt, v, u, lf, gn)
        valid = solid & (u >= 0)
        cv2.imwrite(str(out / f"proj{i}_preview.png"), preview(u, v, wgt))
        results.append(dict(pmap=pmap, u=u, v=v, w=wgt, lift=lf, gain=gn, valid=valid))

    _write_calibration(cfg, out, "single_pose", results,
                       aspect=_aspect_hint(edges), max_overlap=_max_overlap(masks),
                       uniformity=ustats, log=log)
    _overview_camera(work, maps, out)
    marked = _marked_image(work / "all_white.png", edges, edge_mode,
                           "pantalla marcada - imagen completa")
    if marked is not None:
        cv2.imwrite(str(out / "screen_marked.png"), marked)
    return results, {}


def per_projector_ready(cfg, work):
    work = Path(work)
    n = len(cfg.get("projectors", []))
    return n > 0 and all((work / f"scan_proj{i}.npz").exists() and (work / f"screen_proj{i}.json").exists() for i in range(n))

def _content_raster_size(cfg, span, cap=2560):
    pw = max(int(p["width"]) for p in cfg["projectors"])
    ph = max(int(p["height"]) for p in cfg["projectors"])
    aspect = max(span, 1e-3) * pw / ph
    Hc = int(max(180, ph * 0.5))
    Wc = int(min(cap, max(640, Hc * aspect)))
    return Wc, Hc

def _overlap_local(work, i, j, dec, cam_uv, side, wt):
    p = Path(work) / f"overlap_proj{i}_vecino{j}.png"
    if not p.exists() or cam_uv is None: return None
    nb = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if nb is None: return None
    diff = np.clip(nb.astype(np.float32) - dec["black"].astype(np.float32), 0, None)
    hi = float(np.percentile(diff, 99.5))
    if hi <= max(float(wt), 10.0): return None
    # 15 % del pico, no el 50 %: el solape es TODO lo que el vecino ilumina,
    # y en los bordes del solape su luz ya viene cayendo (viñeta, rampa).
    # Con el 50 % el solape medido sale mucho menor que el real y la costura
    # queda desalineada (banda oscura / doble brillo).
    lit = diff > max(0.15 * hi, float(wt))
    uu = cam_uv[..., 0]
    ok = lit & np.isfinite(uu) & (uu >= 0) & (uu <= 1)
    if ok.sum() < 200: return None
    edge = float(np.median([np.percentile(uu[r][ok[r]], 2 if side == "right" else 98) for r in range(0, ok.shape[0], 4) if ok[r].sum() > 10]))
    return float(np.clip((1.0 - edge) if side == "right" else edge, 0.0, 0.49))

def _content_overlaps(cfg, work, decs, cam_uvs):
    n = len(decs)
    wt = float(cfg.get("scan", {}).get("white_threshold", 25))
    sides = []
    for k in range(n - 1):
        a = _overlap_local(work, k, k + 1, decs[k], cam_uvs[k], "right", wt)
        b = _overlap_local(work, k + 1, k, decs[k + 1], cam_uvs[k + 1], "left", wt)
        if a is None and b is None: a = b = 0.15
        elif a is None: a = b
        elif b is None: b = a
        sides.append((a, b))
    return sides

def _content_windows(sides):
    n = len(sides) + 1
    Ls = [1.0]
    for k in range(n - 1):
        a, b = sides[k]
        # el solape es el MISMO trozo de pared para los dos: a*W_i = b*W_j,
        # o sea W_j/W_i = a/b. Se acepta un rango amplio (en una pantalla
        # curva un proyector puede estar bastante más lejos que su vecino);
        # sólo se descarta si la medida es absurda.
        Ls.append(Ls[-1] * (a / b) if b > 1e-6 and 0.5 <= a / b <= 2.0 else Ls[-1])
    Ov = [sides[k][0] * Ls[k] for k in range(n - 1)]
    span = max(sum(Ls) - sum(Ov), 1e-3)
    wins, s = [], 0.0
    for i in range(n):
        wins.append((s, s + Ls[i] / span))
        if i < n - 1: s += (Ls[i] - Ov[i]) / span
    return wins, [o / span for o in Ov], span

def _rasterize(values, x, y, keep, shape):
    H, W = shape
    xi, yi = np.rint(x).astype(int).ravel(), np.rint(y).astype(int).ravel()
    vv, ok = values.ravel(), keep.ravel()
    valid = ok & (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
    idx = yi[valid] * W + xi[valid]
    tot = np.bincount(idx, weights=vv[valid], minlength=H * W).reshape(H, W)
    cnt = np.bincount(idx, minlength=H * W).reshape(H, W)
    return np.where(cnt > 0, tot / np.maximum(cnt, 1), 0).astype(np.float32), cnt > 0

def _content_geometry(cfg, work, log=print):
    """Geometría en el espacio de CONTENIDO (un lienzo común a todos los
    proyectores), con el marcado de puntos de cada uno como forma de pantalla."""
    n = len(cfg["projectors"])
    decs, pmaps, cam_uvs, edges_l = [], [], [], []
    for i in range(n):
        p = Path(work) / f"scan_proj{i}.npz"
        if not p.exists():
            raise RuntimeError(f"Falta el escaneo del proyector {i} ({p}).")
        d = dict(np.load(p))
        pmap, pvalid, st = correspond.projector_to_camera(
            d, int(d["width"]), int(d["height"]), int(d["cell"]),
            cfg.get("mapping", {}).get("min_pixels_per_cell", 2),
            cfg.get("mapping", {}).get("smoothing", 1.0))
        sp = Path(work) / f"screen_proj{i}.json"
        if not sp.exists():
            raise RuntimeError(f"Faltan los bordes del proyector {i} ({sp}): marca los "
                               f"puntos en el paso 5.")
        edges, mode = screen.load_edges(sp)
        faltan = [k for k in screen.EDGES if len(edges.get(k) or []) < 2]
        if faltan:
            raise RuntimeError(
                f"El marcado del proyector {i} ({sp.name}) está incompleto: faltan "
                f"puntos en {', '.join(faltan)} (hacen falta ≥2 por borde). Vuelve al "
                f"paso 5 y marca los 4 bordes de la pantalla de ese proyector.")
        cam_uv = screen.camera_uv(edges, d["valid"].shape[1], d["valid"].shape[0],
                                  mode or "arclength")
        decs.append(d); pmaps.append((pmap, pvalid, st))
        cam_uvs.append(cam_uv); edges_l.append(edges)

    uvls = []
    for i in range(n):
        pmap = pmaps[i][0]
        uvl = _sample_map(np.nan_to_num(cam_uvs[i], nan=-10.0),
                          pmap[..., 0], pmap[..., 1], border=-10.0)
        # La pantalla marcada ES el marco del proyector: se lleva a 0..1 exacto
        # con sus propios percentiles (el marcado son sus 4 bordes).
        ok = pmaps[i][1] & (uvl[..., 0] > -5) & (uvl[..., 1] > -5)
        if ok.sum() > 1000:
            u0, u1 = np.percentile(uvl[..., 0][ok], 0.2), np.percentile(uvl[..., 0][ok], 99.8)
            v0, v1 = np.percentile(uvl[..., 1][ok], 0.2), np.percentile(uvl[..., 1][ok], 99.8)
            if (u1 - u0) > 1e-3 and (v1 - v0) > 1e-3:
                cam_uvs[i] = np.stack([(cam_uvs[i][..., 0] - u0) / (u1 - u0),
                                       (cam_uvs[i][..., 1] - v0) / (v1 - v0)], -1)
                uvl = _sample_map(np.nan_to_num(cam_uvs[i], nan=-10.0),
                                  pmap[..., 0], pmap[..., 1], border=-10.0)
        uvls.append(uvl)

    wins, overlaps, span = _content_windows(_content_overlaps(cfg, work, decs, cam_uvs))
    Wc, Hc = _content_raster_size(cfg, span)
    uvs, masks = [], []
    for i in range(n):
        u = wins[i][0] + uvls[i][..., 0] * (wins[i][1] - wins[i][0])
        v = uvls[i][..., 1]
        uvs.append((u.astype(np.float32), v.astype(np.float32)))
        keep = pmaps[i][1] & (u >= 0) & (u <= 1) & (v >= 0) & (v <= 1)
        r, _ = _rasterize(np.ones(u.shape, np.float32), u * Wc - 0.5, v * Hc - 0.5,
                          keep, (Hc, Wc))
        masks.append(blend.clean_mask(r > 0))
    return dict(decs=decs, pmaps=pmaps, uvs=uvs, wins=wins, overlaps=overlaps,
                span=span, Wc=Wc, Hc=Hc, cam_shape=decs[0]["valid"].shape,
                masks=masks, edges=edges_l)


def compute_per_projector(cfg, work, out, log=print):
    """Modo por proyector: cada escaneo tiene su pose de cámara y su marcado,
    así que el blend se calcula en el espacio de CONTENIDO (un lienzo común).
    Todos los mapas comparten las mismas coordenadas de imagen."""
    if not exr_available():
        raise RuntimeError(exr_error_msg())
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    geo = _content_geometry(cfg, work, log=log)
    _save_canvas_cache(geo, cfg, work)      # el paso 6 ya no lo recalcula
    n = len(cfg["projectors"])
    Wc, Hc = geo["Wc"], geo["Hc"]
    scale = Wc / max(geo["span"] * int(cfg["projectors"][0]["width"]), 1e-6)

    weights, _wmasks = _blend_weights(cfg, geo["masks"])
    tot = np.sum(weights, axis=0).astype(np.float32)
    ncov = np.sum([m.astype(np.float32) for m in geo["masks"]], axis=0)
    lift, nmax = blend.black_lift(geo["masks"], None, 25.0 * scale)

    whites_c, blacks_c = [], []
    for i in range(n):
        pmap, pvalid, _ = geo["pmaps"][i]
        u, v = geo["uvs"][i]
        x = np.where(pvalid, u * Wc - 0.5, -100.0).astype(np.float32)
        y = np.where(pvalid, v * Hc - 0.5, -100.0).astype(np.float32)
        wc, _ = _rasterize(blend.sample(geo["decs"][i]["white"], pmap), x, y,
                           pvalid, (Hc, Wc))
        bc, _ = _rasterize(blend.sample(geo["decs"][i]["black"], pmap), x, y,
                           pvalid, (Hc, Wc))
        whites_c.append(wc); blacks_c.append(bc)
    gain, ustats = blend.uniformity_gain(whites_c, blacks_c, geo["masks"], weights,
                                         np.any(geo["masks"], axis=0))

    results = []
    for i in range(n):
        u, v = geo["uvs"][i]
        keep = geo["pmaps"][i][1]
        w0, w1 = geo["wins"][i]
        # SOLO la ventana de este proyector: fuera de ella el contenido es del
        # vecino y proyectarlo ahí sería contenido duplicado y luz doble.
        inside = (keep & (u >= w0 - 0.005) & (u <= w1 + 0.005)
                  & (v >= -0.005) & (v <= 1.005))
        solid = _solid_mask(inside) & keep          # sin huecos del sistema
        know = inside & np.isfinite(u) & np.isfinite(v)
        if know.any():
            # los huecos rellenados toman la coordenada del vecino válido más
            # cercano: si no, caerían en 0 (contenido del extremo del lienzo)
            uf = _fill_nearest(np.where(know, u, 0.0), know)
            vf = _fill_nearest(np.where(know, v, 0.0), know)
        else:
            uf, vf = u, v
        uc = np.where(solid, np.clip(uf, w0, w1), -1.0).astype(np.float32)
        vc = np.where(solid, np.clip(vf, 0.0, 1.0), -1.0).astype(np.float32)
        x = np.where(solid, uc * Wc - 0.5, -100.0).astype(np.float32)
        y = np.where(solid, vc * Hc - 0.5, -100.0).astype(np.float32)

        wgt = np.where(solid, np.clip(_sample_map(weights[i], x, y), 0, 1), 0.0).astype(np.float32)
        lf = np.where(solid, _sample_map(lift, x, y), 0.0).astype(np.float32)
        gn = np.where(solid, _sample_map(gain, x, y, border=1.0), 1.0).astype(np.float32)
        # hueco real del lienzo (ningún proyector llega): antes que un agujero
        # negro, se reparte la luz entre los que sí cubren ese punto
        gap = solid & (_sample_map(tot, x, y) < 0.05)
        if gap.any():
            nc = np.maximum(_sample_map(ncov, x, y, border=1.0), 1.0)
            wgt = np.where(gap, np.maximum(wgt, (1.0 / nc)).astype(np.float32), wgt).astype(np.float32)

        uo = np.where(solid, uc, -1.0).astype(np.float32)
        vo = np.where(solid, 1.0 - vc, -1.0).astype(np.float32)
        _write_exr(out, i, wgt, vo, uo, lf, gn)
        cv2.imwrite(str(out / f"proj{i}_preview.png"), preview(uo, vo, wgt))
        valid = solid & (uo >= 0)
        results.append(dict(pmap=geo["pmaps"][i][0], u=uo, v=vo, w=wgt,
                            lift=lf, gain=gn, valid=valid))
        log(f"  proyector {i}: {valid.mean() * 100:.1f}% de su imagen proyecta contenido")

    _write_calibration(cfg, out, "per_projector", results,
                       windows=geo["wins"], canvas={"width": Wc, "height": Hc},
                       aspect=_aspect_hint(geo["edges"][0]),
                       max_overlap=round(float(max(geo["overlaps"] or [0.0])), 4),
                       uniformity=ustats, log=log)
    _overview_canvas(geo, out)
    for i in range(n):
        p = Path(work) / f"all_white_proj{i}.png"
        sp = Path(work) / f"screen_proj{i}.json"
        if p.exists() and sp.exists():
            e_i, m_i = screen.load_edges(sp)
            mk = _marked_image(p, e_i, m_i,
                               f"P{i} - pantalla marcada (imagen completa)")
            if mk is not None:
                cv2.imwrite(str(out / f"screen_proj{i}_marked.png"), mk)
    return results, {"canvas": {"width": Wc, "height": Hc}, "windows": geo["wins"]}

# --------------------------------------------------------------------------
# VISTA PREVIA DE LAS COSTURAS (paso 6 del wizard)
# --------------------------------------------------------------------------
def _scan_stamp(cfg: dict, work) -> str:
    """Huella de los archivos que definen la geometría (escaneos y marcados)."""
    h = hashlib.md5()
    for i in range(len(cfg.get("projectors", []))):
        for pat in (f"scan_proj{i}.npz", f"screen_proj{i}.json",
                    f"overlap_proj{i}_vecino*"):
            for p in sorted(Path(work).glob(pat)):
                try:
                    s = p.stat()
                except OSError:
                    continue
                h.update(f"{p.name}:{int(s.st_mtime_ns)}:{s.st_size}".encode())
    return h.hexdigest()


def geometry_cached(cfg: dict, work) -> bool:
    """¿La geometría de las costuras ya está calculada para ESTOS archivos?"""
    p = Path(work) / "seam_geometry.npz"
    if not p.exists():
        return False
    try:
        with np.load(p) as d:
            return str(d["key"]) == _scan_stamp(cfg, work)
    except Exception:
        return False


def _canvas_geometry(cfg: dict, work):
    """Huellas del lienzo: del caché si sigue valiendo; si no, se calculan y
    se guardan en work/seam_geometry.npz (así la vista previa es instantánea)."""
    p = Path(work) / "seam_geometry.npz"
    key = _scan_stamp(cfg, work)
    if p.exists():
        try:
            with np.load(p) as d:
                if str(d["key"]) == key:
                    Wc, Hc = int(d["Wc"]), int(d["Hc"])
                    packed = d["masks"]
                    masks = [np.unpackbits(packed[i], axis=1)[:, :Wc].astype(bool)
                             for i in range(packed.shape[0])]
                    return dict(Wc=Wc, Hc=Hc, span=float(d["span"]),
                                wins=[tuple(float(x) for x in w) for w in d["wins"]],
                                overlaps=[float(o) for o in d["overlaps"]],
                                masks=masks)
        except Exception:
            pass
    geo = _content_geometry(cfg, work, log=lambda *a, **k: None)
    _save_canvas_cache(geo, cfg, work)
    return geo


def _save_canvas_cache(geo: dict, cfg: dict, work) -> None:
    """Guarda sólo lo que necesita la vista previa (huellas del lienzo), no
    los mapas densos del proyector: así el caché son unos cientos de KB."""
    try:
        np.savez_compressed(
            Path(work) / "seam_geometry.npz",
            key=_scan_stamp(cfg, work), Wc=geo["Wc"], Hc=geo["Hc"], span=geo["span"],
            wins=np.asarray(geo["wins"], np.float64),
            overlaps=np.asarray(geo["overlaps"], np.float64),
            masks=np.stack([np.packbits(m.astype(np.uint8), axis=1) for m in geo["masks"]]))
    except Exception:
        pass


def seam_preview(cfg: dict, work, narrow: float = 1.0) -> dict:
    """Cómo quedan las costuras con las huellas REALES del escaneo.

    Imagen (RGB) del lienzo de contenido: cada proyector tiñe su zona con su
    peso de mezcla, así se ve el ancho real de cada rampa y dónde se solapan
    dos proyectores. Devuelve también, por costura, el ancho de banda en px de
    lienzo y de la imagen del proyector, y la suma de luz (debe dar 1).
    """
    geo = _canvas_geometry(cfg, work)
    Wc, Hc = int(geo["Wc"]), int(geo["Hc"])
    masks, wins = geo["masks"], geo["wins"]
    wm = [_grow_mask(m, 2) for m in masks]
    weights = blend.normalize_weights(blend.camera_weights(wm), wm)
    if float(narrow or 1.0) > 1.0:
        weights = blend.normalize_weights(blend.narrow_weights(weights, float(narrow)), wm)
    total = np.sum(weights, axis=0)

    img = np.zeros((Hc, Wc, 3), np.float32)
    for i, w in enumerate(weights):
        img += w[..., None] * np.asarray(PROJ_COLORS[i % len(PROJ_COLORS)], np.float32)
    img = np.clip(img, 0, 255).astype(np.uint8)
    for i, m in enumerate(masks):
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnts, -1, tuple(int(c) for c in
                         PROJ_COLORS[i % len(PROJ_COLORS)]), 1)
        ys, xs = np.nonzero(m)
        if len(xs):
            cv2.putText(img, f"P{i}", (int(np.median(xs)), int(np.median(ys))),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    seams = []
    for k in range(len(masks) - 1):
        both = masks[k] & masks[k + 1]
        band = both & (weights[k] > 0.02) & (weights[k] < 0.98)
        def _median_width(m):
            w = []
            for r in range(Hc):
                xs = np.flatnonzero(m[r])
                if xs.size:
                    w.append(int(xs[-1] - xs[0] + 1))
            return float(np.median(w)) if w else 0.0
        bw, ow = _median_width(band), _median_width(both)
        pw = int(cfg["projectors"][k]["width"])
        frac = max(wins[k][1] - wins[k][0], 1e-3)
        seams.append(dict(k=k,
                          band_px_canvas=round(bw, 1),
                          band_px_projector=round(bw * pw / max(Wc * frac, 1e-6), 1),
                          overlap_canvas_px=round(ow, 1),
                          sum_median=round(float(np.median(total[both])) if both.any() else 0.0, 4)))
        x = int(round(wins[k][1] * Wc))
        cv2.line(img, (x, 0), (x, Hc - 1), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, f"P{k}-P{k+1}", (max(2, x + 4), Hc - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # ¿los mapas de output/ son más viejos que estos escaneos/bordes?
    out = Path(work).parent / cfg.get("output_dir", "output")
    stale = False
    maps = sorted(out.glob("proj*_map.exr"))
    if maps:
        newest = 0.0
        for i in range(len(cfg.get("projectors", []))):
            for pat in (f"scan_proj{i}.npz", f"screen_proj{i}.json"):
                for p in Path(work).glob(pat):
                    newest = max(newest, p.stat().st_mtime)
        stale = min(p.stat().st_mtime for p in maps) < newest
    return {"img": cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
            "source": "huellas reales de tus escaneos",
            "seams": seams, "stale_maps": bool(stale),
            "canvas": {"width": Wc, "height": Hc}}


def render_projector(r, content=None, gamma=2.2, black=0.0, uniformity=1.0, white=False):
    u, v, w = r["u"], r["v"], r["w"]
    if white: return np.full((u.shape[0], u.shape[1], 3), 255, np.uint8)
    if content is None:
        gu, gv = u * 16, v * 16 * 9 / 16
        line = (np.abs(gu - np.round(gu)) < 0.6) | (np.abs(gv - np.round(gv)) < 0.6)
        img = np.dstack([np.where(line, 1.0, np.clip(u, 0, 1)) * 0.6, np.where(line, 1.0, np.clip(v, 0, 1)) * 0.6, np.where(line, 1.0, 1.0 - np.clip(u, 0, 1)) * 0.6])
    else:
        img = cv2.remap(content.astype(np.float32) / 255.0, u, v, cv2.INTER_LINEAR)
    img = np.power(np.clip(img, 0, 1), 1.0 / gamma)
    if black > 0: img = np.clip(img + black, 0, 1)
    return (img * 255).astype(np.uint8)

def preview(u, v, w, gamma=2.2):
    valid = u >= 0
    gu, gv = np.clip(u, 0, 1) * 16, (1 - np.clip(v, 0, 1)) * 16 * 9 / 16
    line = (np.abs(gu - np.round(gu)) < 0.06) | (np.abs(gv - np.round(gv)) < 0.06)
    img = np.dstack([u, v, 1 - u]) * 0.6
    img = np.where(line[..., None], 1.0, img) * np.power(np.clip(w, 0, 1), 1 / gamma)[..., None]
    return (np.where(valid[..., None], img, 0) * 255).astype(np.uint8)
