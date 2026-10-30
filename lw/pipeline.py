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
    for k, (name, img) in enumerate(patterns.make_sequence(
            w, h, cell, sc.get("pattern_intensity", 255), phase_steps,
            sc.get("phase_period", 32), False)):
        rig.show_only(i, img)
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


def calculate_optimal_cell_size(cam_w, cam_h, proj_w, proj_h, current_cell=16, min_pixels_per_cell=4, max_cell_size=32):
    proj_per_cam = (proj_w / cam_w + proj_h / cam_h) / 2
    pixels_per_cell = proj_per_cam * current_cell
    if pixels_per_cell < min_pixels_per_cell:
        return max(8, int(current_cell * (min_pixels_per_cell / max(1, pixels_per_cell))))
    elif pixels_per_cell > 8:
        return max(8, int(current_cell * (8 / pixels_per_cell)))
    return current_cell


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
        cam_h, cam_w = rig.camera_shape if hasattr(rig, 'camera_shape') else (1080, 1920)
        pw, ph = rig.proj_size(i)
        old_cell = config["scan"].get("cell_size", 16)
        is_webcam = config.get("camera", {}).get("type") == "webcam"
        target_ppc = 6 if is_webcam else 4
        opt_cell = calculate_optimal_cell_size(cam_w, cam_h, pw, ph, current_cell=old_cell, min_pixels_per_cell=target_ppc)

        log(f"  Proyector {i}: cell_size auto-ajustado a {opt_cell} (antes {old_cell})")
        config["scan"]["cell_size"] = opt_cell
        with open('config.json', 'w') as f: json.dump(config, f, indent=2)

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
        pmap, pvalid, st = correspond.projector_to_camera(d, w, h, cell)
        maps.append((pmap, pvalid, st))

    masks = [blend.footprint(pm, (cam_h, cam_w)) for (pm, _, _), d in zip(maps, decs)]
    weights = blend.camera_weights(masks)
    lift, nmax = blend.black_lift(masks, screen_mask)
    gain, _ = blend.uniformity_gain([d["white"] for d in decs], [d["black"] for d in decs], masks, weights, screen_mask)

    results = []
    for i, (d, (pmap, pvalid, st)) in enumerate(zip(decs, maps)):
        uv = cv2.remap(np.nan_to_num(cam_uv, nan=-10.0), pmap[..., 0], pmap[..., 1], cv2.INTER_LINEAR)
        wgt = blend.sample(weights[i], pmap)
        lf = blend.sample(lift, pmap)
        gn = blend.sample(gain, pmap, border=1.0)
        valid = pvalid & np.all((uv >= 0) & (uv <= 1), axis=-1)
        u, v = np.where(valid, uv[..., 0], -1.0), np.where(valid, 1.0 - uv[..., 1], -1.0)

        fname, bname = f"proj{i}_map.exr", f"proj{i}_black.exr"
        cv2.imwrite(str(out / fname), np.dstack([wgt, v, u, np.ones_like(u)]))
        cv2.imwrite(str(out / bname), np.dstack([np.zeros_like(u), gn, lf, np.ones_like(u)]))
        results.append(dict(pmap=pmap, u=u, v=v, w=wgt, lift=lf, gain=gn, valid=valid))
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
    lit = diff > max(0.5 * hi, float(wt))
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
        Ls.append(Ls[-1] * (a / b) if b > 1e-6 and 0.85 <= a / b <= 1.18 else Ls[-1])
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
    n = len(cfg["projectors"])
    decs, pmaps, cam_uvs = [], [], []
    for i in range(n):
        d = dict(np.load(work / f"scan_proj{i}.npz"))
        pmap, pvalid, st = correspond.projector_to_camera(d, int(d["width"]), int(d["height"]), int(d["cell"]))
        edges, mode = screen.load_edges(work / f"screen_proj{i}.json")
        cam_uv = screen.camera_uv(edges, d["valid"].shape[1], d["valid"].shape[0], mode or "arclength")
        decs.append(d); pmaps.append((pmap, pvalid, st)); cam_uvs.append(cam_uv)

    uvls = []
    for i in range(n):
        uvl = cv2.remap(np.nan_to_num(cam_uvs[i], nan=-10.0), pmaps[i][0], pmaps[i][1], cv2.INTER_LINEAR)
        ok = pmaps[i][1] & (uvl[..., 0] > -5.0)
        if ok.sum() > 1000:
            u0, u1 = np.percentile(uvl[..., 0][ok], 0.2), np.percentile(uvl[..., 0][ok], 99.8)
            v0, v1 = np.percentile(uvl[..., 1][ok], 0.2), np.percentile(uvl[..., 1][ok], 99.8)
            cam_uvs[i] = np.stack([(cam_uvs[i][..., 0] - u0) / (u1 - u0), (cam_uvs[i][..., 1] - v0) / (v1 - v0)], -1)
            uvl = cv2.remap(np.nan_to_num(cam_uvs[i], nan=-10.0), pmaps[i][0], pmaps[i][1], cv2.INTER_LINEAR)
        uvls.append(uvl)

    wins, overlaps, span = _content_windows(_content_overlaps(cfg, work, decs, cam_uvs))
    Wc, Hc = _content_raster_size(cfg, span)
    uvs = [(wins[i][0] + uvls[i][..., 0] * (wins[i][1] - wins[i][0]), uvls[i][..., 1]) for i in range(n)]
    masks = [blend.clean_mask(_rasterize(np.ones(u.shape), u * Wc - 0.5, v * Hc - 0.5, pmaps[i][1] & (u>=0) & (u<=1), (Hc, Wc))[0] > 0) for i, (u, v) in enumerate(uvs)]
    return dict(decs=decs, pmaps=pmaps, uvs=uvs, wins=wins, overlaps=overlaps, span=span, Wc=Wc, Hc=Hc, cam_shape=decs[0]["valid"].shape, masks=masks)

def compute_per_projector(cfg, work, out, log=print):
    geo = _content_geometry(cfg, work, log=log)
    Wc, Hc, n = geo["Wc"], geo["Hc"], len(cfg["projectors"])
    scale = Wc / (geo["span"] * int(cfg["projectors"][0]["width"]))
    weights = blend.camera_weights(geo["masks"])
    lift, nmax = blend.black_lift(geo["masks"], None, 25 * scale)

    whites_c, blacks_c = [], []
    for i in range(n):
        x, y = geo["uvs"][i][0] * Wc - 0.5, geo["uvs"][i][1] * Hc - 0.5
        wc, _ = _rasterize(blend.sample(geo["decs"][i]["white"], geo["pmaps"][i][0]), x, y, geo["pmaps"][i][1], (Hc, Wc))
        bc, _ = _rasterize(blend.sample(geo["decs"][i]["black"], geo["pmaps"][i][0]), x, y, geo["pmaps"][i][1], (Hc, Wc))
        whites_c.append(wc); blacks_c.append(bc)
    gain, _ = blend.uniformity_gain(whites_c, blacks_c, geo["masks"], weights, np.any(geo["masks"], axis=0))

    for i in range(n):
        u, v = geo["uvs"][i]
        valid = geo["pmaps"][i][1] & (u >= 0) & (u <= 1)
        x, y = np.where(valid, u * Wc - 0.5, -100.0), np.where(valid, v * Hc - 0.5, -100.0)
        wgt = cv2.remap(weights[i], x.astype(np.float32), y.astype(np.float32), cv2.INTER_LINEAR)
        lf = cv2.remap(lift, x.astype(np.float32), y.astype(np.float32), cv2.INTER_LINEAR)
        gn = cv2.remap(gain, x.astype(np.float32), y.astype(np.float32), cv2.INTER_LINEAR)
        uo, vo = np.where(valid, u, -1.0), np.where(valid, 1.0 - v, -1.0)
        cv2.imwrite(str(out / f"proj{i}_map.exr"), np.dstack([wgt, vo, uo, np.ones_like(uo)]))
        cv2.imwrite(str(out / f"proj{i}_black.exr"), np.dstack([np.zeros_like(uo), gn, lf, np.ones_like(uo)]))
    return [], {}

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
