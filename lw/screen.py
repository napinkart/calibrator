import json
import numpy as np
import cv2
from pathlib import Path
from scipy import ndimage
from scipy.interpolate import CubicSpline, griddata
from scipy.signal import savgol_filter

EDGES = ["top", "right", "bottom", "left"]
EDGE_HELP = {
    "top": "Borde SUPERIOR: de IZQUIERDA a DERECHA",
    "right": "Borde DERECHO: de ARRIBA a ABAJO",
    "bottom": "Borde INFERIOR: de IZQUIERDA a DERECHA",
    "left": "Borde IZQUIERDO: de ARRIBA a ABAJO",
}

GHOST_COLORS = {"left": (70, 70, 255), "right": (255, 200, 40)}


class EdgeCurve:
    def __init__(self, pts, mode="arclength"):
        pts = np.asarray(pts, np.float64)
        if len(pts) < 2:
            raise ValueError("Cada borde necesita al menos 2 puntos")
        if mode == "uniform":
            t = np.linspace(0, 1, len(pts))
        else:
            d = np.r_[0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]
            t = d / d[-1]
        self.p0, self.p1 = pts[0], pts[-1]
        self.cs = CubicSpline(t, pts, bc_type="natural")
        self.d0 = self.cs(0.0, 1)
        self.d1 = self.cs(1.0, 1)

    def __call__(self, t):
        t = np.asarray(t, np.float64)
        out = self.cs(np.clip(t, 0, 1))
        lo, hi = t < 0, t > 1
        out[lo] = self.p0 + t[lo, None] * self.d0
        out[hi] = self.p1 + (t[hi, None] - 1) * self.d1
        return out


def snap_corners(edges: dict) -> dict:
    e = {k: np.asarray(v, np.float64).copy() for k, v in edges.items()}
    tl = (e["top"][0] + e["left"][0]) / 2
    tr = (e["top"][-1] + e["right"][0]) / 2
    br = (e["bottom"][-1] + e["right"][-1]) / 2
    bl = (e["bottom"][0] + e["left"][-1]) / 2
    e["top"][0], e["left"][0] = tl, tl
    e["top"][-1], e["right"][0] = tr, tr
    e["bottom"][-1], e["right"][-1] = br, br
    e["bottom"][0], e["left"][-1] = bl, bl
    return e


def coons(edges: dict, u: np.ndarray, v: np.ndarray, mode="arclength"):
    e = snap_corners(edges)
    T, B = EdgeCurve(e["top"], mode), EdgeCurve(e["bottom"], mode)
    L, R = EdgeCurve(e["left"], mode), EdgeCurve(e["right"], mode)
    tl, tr, br, bl = e["top"][0], e["top"][-1], e["bottom"][-1], e["bottom"][0]
    uf, vf = u.ravel(), v.ravel()
    U, V = uf[:, None], vf[:, None]
    P = ((1 - V) * T(uf) + V * B(uf) + (1 - U) * L(vf) + U * R(vf)
         - ((1 - U) * (1 - V) * tl + U * (1 - V) * tr
            + (1 - U) * V * bl + U * V * br))
    return P.reshape(u.shape + (2,))


def camera_uv(edges: dict, cam_w: int, cam_h: int, mode="arclength",
              samples: int = 160, margin: float = 0.08):
    t = np.linspace(-margin, 1 + margin, samples)
    uu, vv = np.meshgrid(t, t)
    P = coons(edges, uu, vv, mode).reshape(-1, 2)
    x0, y0 = np.floor(P.min(0)).astype(int)
    x1, y1 = np.ceil(P.max(0)).astype(int)
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, cam_w - 1), min(y1, cam_h - 1)
    out = np.full((cam_h, cam_w, 2), np.nan, np.float32)
    if x1 <= x0 or y1 <= y0:
        raise RuntimeError("La pantalla definida queda fuera de la imagen de cámara")
    gy, gx = np.mgrid[y0:y1 + 1, x0:x1 + 1]
    for c, src in enumerate([uu.ravel(), vv.ravel()]):
        out[y0:y1 + 1, x0:x1 + 1, c] = griddata(P, src, (gx, gy), method="linear")
    return out


def edge_normals(pts) -> np.ndarray:
    p = np.asarray(pts, np.float64)
    if len(p) < 2:
        return np.tile([0.0, 1.0], (len(p), 1))
    t = np.empty_like(p)
    if len(p) == 2:
        t[:] = p[1] - p[0]
    else:
        t[0] = p[1] - p[0]
        t[-1] = p[-1] - p[-2]
        t[1:-1] = p[2:] - p[:-2]
    n = np.c_[-t[:, 1], t[:, 0]]
    ln = np.hypot(n[:, 0], n[:, 1])[:, None]
    return n / np.where(ln == 0, 1, ln)


def refine_point(gray, pt, normal, search: float = 6.0, step: float = 0.2,
                 min_step: float = 8.0, k_noise: float = 3.0):
    g = np.asarray(gray, np.float32)
    ts = np.arange(-search, search + 1e-9, step)
    xs = pt[0] + ts * normal[0]
    ys = pt[1] + ts * normal[1]
    vals = ndimage.map_coordinates(g, [ys, xs], order=1, mode="nearest")
    d = np.abs(np.diff(vals)) / step
    k = int(np.argmax(d))
    p0 = [float(pt[0]), float(pt[1])]
    if k == 0 or k == len(d) - 1:
        return p0, 0.0
    a, b, c = d[k - 1], d[k], d[k + 1]
    den = a - 2 * b + c
    off = 0.5 * (a - c) / den if den != 0 else 0.0
    t = ts[k] + float(np.clip(off, -1.0, 1.0)) * step

    pre = vals[(ts >= t - 4) & (ts <= t - 2)]
    post = vals[(ts >= t + 2) & (ts <= t + 4)]
    if not len(pre) or not len(post):
        return p0, 0.0
    jump = float(post.mean() - pre.mean())
    ti = np.arange(np.ceil(-search), np.floor(search) + 1.0)
    vi = ndimage.map_coordinates(g, [pt[1] + ti * normal[1], pt[0] + ti * normal[0]],
                                 order=1, mode="nearest")
    sigma = 1.4826 * float(np.median(np.abs(np.diff(vi, 2)))) / np.sqrt(6.0) if len(vi) > 2 else 0.0
    if abs(jump) < max(min_step, k_noise * sigma):
        return p0, jump

    q = [float(pt[0] + t * normal[0]), float(pt[1] + t * normal[1])]
    H, W = g.shape[:2]
    if not (1.5 <= q[0] <= W - 1.5 and 1.5 <= q[1] <= H - 1.5):
        return p0, jump
    return q, jump


def refine_edges(gray, edges: dict, search: float = 6.0, min_step: float = 8.0):
    out, skipped = {}, 0
    for k in EDGES:
        pts = np.asarray(edges.get(k, []), np.float64)
        if len(pts) < 3:
            out[k] = [[float(x), float(y)] for x, y in pts]
            continue
        nrm = edge_normals(pts)
        new = pts.copy()
        for j in range(1, len(pts) - 1):
            p, s = refine_point(gray, pts[j], nrm[j], search, min_step=min_step)
            if p[0] == pts[j][0] and p[1] == pts[j][1]:
                skipped += 1
            else:
                new[j] = p
        out[k] = [[float(x), float(y)] for x, y in new]
    return out, skipped


def _keep_main_blobs(mask: np.ndarray, min_ratio: float = 0.2) -> np.ndarray:
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = np.zeros_like(mask, dtype=bool)
    thr = max(400.0, float(areas.max()) * min_ratio)
    for k, a in enumerate(areas, start=1):
        if a >= thr:
            keep |= (lab == k)
    return keep


def _inner_curve(mask: np.ndarray, side: str, min_rows: int = 20,
                 tol: float = 25.0) -> np.ndarray | None:
    H, W = mask.shape
    xs = np.full(H, np.nan)
    for y in range(H):
        idx = np.flatnonzero(mask[y])
        if idx.size >= 4:
            xs[y] = idx[0] if side == "right" else idx[-1]
    ok = np.isfinite(xs)
    if ok.sum() < min_rows:
        return None
    yy = np.arange(H)
    v = xs.copy()
    v[~ok] = np.interp(yy[~ok], yy[ok], xs[ok])
    v = ndimage.median_filter(v, size=41, mode="nearest")
    good = ok & (np.abs(xs - v) <= tol)
    best, start = (0, 0), None
    for y in range(H + 1):
        on = y < H and bool(good[y])
        if on and start is None:
            start = y
        elif not on and start is not None:
            if y - start > best[1] - best[0]:
                best = (start, y)
            start = None
    y0, y1 = best
    if y1 - y0 < min_rows:
        return None
    med = float(np.median(v[y0:y1]))
    if med <= 2 or med >= W - 3:
        return None
    curve = v[y0:y1]
    if curve.size >= 11:
        win = min(101, curve.size if curve.size % 2 else curve.size - 1)
        if win >= 11:
            curve = savgol_filter(curve, win, 2, mode="nearest")
    out = np.full(H, np.nan)
    out[y0:y1] = curve
    return out


def neighbor_ghosts(work, projector: int, n_proj: int,
                    white_threshold: float = 25.0) -> list[dict]:
    work = Path(work)
    black = None
    sp = work / f"scan_proj{int(projector)}.npz"
    if sp.exists():
        try:
            with np.load(sp) as d:
                if "black" in d.files:
                    black = np.asarray(d["black"], np.float32)
        except Exception:
            black = None
    out = []
    for side, j in (("left", int(projector) - 1), ("right", int(projector) + 1)):
        if not (0 <= j < int(n_proj)):
            continue
        f = work / f"overlap_proj{int(projector)}_vecino{j}.png"
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        gh = {"side": side, "neighbor": j, "mask": None, "x": None,
              "stale": False, "why": "sin captura (re-escanea este proyector)"}
        if img is None:
            out.append(gh)
            continue
        g = img.astype(np.float32)
        if black is not None and black.shape == g.shape:
            g = g - black
        g = cv2.GaussianBlur(g, (0, 0), 3.0)
        lo = float(np.percentile(g, 10))
        hi = float(np.percentile(g, 99))
        if hi - lo < max(20.0, float(white_threshold)):
            gh["stale"] = True
            gh["why"] = "el vecino casi no ilumina en esta pose"
            out.append(gh)
            continue
        thr = max(lo + 0.5 * (hi - lo), 10.0)
        mask = (g > thr).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        mask = _keep_main_blobs(mask > 0)
        x = _inner_curve(mask, side)
        gh["mask"] = mask
        if x is None:
            gh["stale"] = True
            gh["why"] = "sin borde visible en esta captura"
        else:
            gh["x"], gh["why"] = x, ""
        out.append(gh)
    usable = [gh for gh in out if gh["x"] is not None]
    if len(usable) == 2:
        a, b = usable[0]["mask"], usable[1]["mask"]
        uni = int(np.logical_or(a, b).sum())
        iou = int(np.logical_and(a, b).sum()) / uni if uni else 0.0
        if iou > 0.95:
            for gh in usable:
                gh["stale"] = True
                gh["why"] = "captura repetida (mismo frame que el otro vecino)"
    return out


def draw_ghosts(img_bgr: np.ndarray, ghosts: list[dict], alpha: float = 0.15,
                labels: bool = True) -> np.ndarray:
    if not ghosts:
        return img_bgr
    vis = img_bgr.copy()
    for gh in ghosts:
        if gh.get("x") is None:
            continue
        col = GHOST_COLORS[gh["side"]]
        m = gh["mask"]
        if m is not None and m.any():
            vis[m] = (vis[m].astype(np.float32) * (1.0 - alpha)
                      + np.asarray(col, np.float32) * alpha).astype(np.uint8)
        x = gh["x"]
        ys = np.flatnonzero(np.isfinite(x))
        if ys.size < 2:
            continue
        pts = np.stack([x[ys], ys], axis=1).astype(np.int32)
        cv2.polylines(vis, [pts.reshape(-1, 1, 2)], False, col, 2, cv2.LINE_AA)
        if labels:
            name = f"P{gh['neighbor']} borde {'izq' if gh['side'] == 'right' else 'der'}"
            if gh["stale"]:
                name += " ?"
            (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ox, oy = int(pts[0, 0]), int(pts[0, 1])
            tx = ox + 8 if gh["side"] == "left" else ox - tw - 8
            tx = int(min(max(2, tx), vis.shape[1] - tw - 4))
            ty = int(max(th + 10, min(oy, vis.shape[0] - 6)))
            cv2.rectangle(vis, (tx - 4, ty - th - 5), (tx + tw + 4, ty + 5), (25, 25, 25), -1)
            cv2.putText(vis, name, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    return vis


def ghost_legend(ghosts: list[dict]) -> str:
    parts = []
    for gh in ghosts:
        name = f"P{gh['neighbor']} borde {'izq' if gh['side'] == 'right' else 'der'}"
        if gh.get("x") is None:
            name += f" (sin fantasma: {gh.get('why', 'captura no utilizable')})"
        elif gh["stale"]:
            name += f" (dudoso: {gh.get('why', 'captura repetida')})"
        parts.append(name)
    return " · ".join(parts)


def click_edges(image: np.ndarray, existing: dict | None = None,
                max_display: int = 1600, ghosts: list[dict] | None = None) -> dict | None:
    img = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    scale = min(1.0, max_display / max(img.shape[:2]))
    base = cv2.resize(img, None, fx=scale, fy=scale) if scale < 1 else img.copy()
    ghosts = [gh for gh in (ghosts or []) if gh.get("x") is not None]
    ghost_vis = bool(ghosts)
    base_ghost = base
    if ghosts:
        painted = draw_ghosts(img, ghosts, labels=False)
        base_ghost = (cv2.resize(painted, None, fx=scale, fy=scale) if scale < 1
                      else painted)
        for gh in ghosts:
            x = gh["x"]
            ys = np.flatnonzero(np.isfinite(x))
            if ys.size < 2:
                continue
            col = GHOST_COLORS[gh["side"]]
            name = f"P{gh['neighbor']} borde {'izq' if gh['side'] == 'right' else 'der'}"
            (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ox, oy = int(x[ys[0]] * scale), int(ys[0] * scale)
            tx = ox + 8 if gh["side"] == "left" else ox - tw - 8
            tx = min(max(2, tx), base_ghost.shape[1] - tw - 4)
            ty = max(th + 4, min(oy, base_ghost.shape[0] - 4))
            cv2.rectangle(base_ghost, (tx - 4, ty - th - 5), (tx + tw + 4, ty + 5),
                          (25, 25, 25), -1)
            cv2.putText(base_ghost, name, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1,
                        cv2.LINE_AA)
    edges = {k: list(map(tuple, v)) for k, v in (existing or {}).items()}
    colors = {"top": (0, 255, 255), "right": (255, 0, 255),
              "bottom": (0, 255, 0), "left": (255, 128, 0)}
    state = {"i": 0}
    win = "LibreWarp - definir pantalla"

    def on_mouse(ev, x, y, flags, _):
        k = EDGES[state["i"]]
        if ev == cv2.EVENT_LBUTTONDOWN:
            edges.setdefault(k, []).append((x / scale, y / scale))
        elif ev == cv2.EVENT_RBUTTONDOWN and edges.get(k):
            edges[k].pop()

    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        vis = (base_ghost if ghost_vis else base).copy()
        for k, pts in edges.items():
            for j, p in enumerate(pts):
                q = (int(p[0] * scale), int(p[1] * scale))
                cv2.circle(vis, q, 5, colors[k], -1)
                if j:
                    pp = pts[j - 1]
                    cv2.line(vis, (int(pp[0] * scale), int(pp[1] * scale)), q, colors[k], 1)
        if all(len(edges.get(k, [])) >= 2 for k in EDGES):
            try:
                t = np.linspace(0, 1, 11)
                uu, vv = np.meshgrid(t, t)
                P = coons({k: edges[k] for k in EDGES}, uu, vv) * scale
                for r in range(11):
                    cv2.polylines(vis, [P[r].astype(np.int32)], False, (200, 200, 200), 1)
                    cv2.polylines(vis, [P[:, r].astype(np.int32)], False, (200, 200, 200), 1)
            except Exception:
                pass
        k = EDGES[state["i"]]
        msg = f"[{state['i']+1}/4] {EDGE_HELP[k]}  ({len(edges.get(k, []))} pts)  Enter=siguiente  clic der=borrar  Esc=cancelar"
        if ghosts:
            msg += f"  g=vecinos ({'visibles' if ghost_vis else 'ocultos'})"
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 28), (0, 0, 0), -1)
        cv2.putText(vis, msg, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colors[k], 1, cv2.LINE_AA)
        cv2.imshow(win, vis)
        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            cv2.destroyWindow(win)
            return None
        if key == ord("g") and ghosts:
            ghost_vis = not ghost_vis
        if key in (8, 127) and edges.get(k):
            edges[k].pop()
        if key in (13, 10, 32):
            if len(edges.get(k, [])) < 2:
                continue
            if state["i"] == 3:
                break
            state["i"] += 1
    cv2.destroyWindow(win)
    return {k: [list(map(float, p)) for p in edges[k]] for k in EDGES}


def save_edges(path, edges, mode):
    with open(path, "w") as f:
        json.dump({"edge_param": mode, "edges": edges}, f, indent=2)


def load_edges(path):
    get = lambda p: json.load(open(p))
    # compatible con screen.json y screen_projN.json
    d = get(path)
    return d["edges"], d.get("edge_param", "arclength")
