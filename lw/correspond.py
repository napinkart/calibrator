"""
De la decodificación por celdas a un mapa denso proyector -> cámara.

1. Centroide (en cámara) de cada celda del proyector, con rechazo de outliers.
2. Limpieza del grid (filtro de mediana para celdas erróneas).
3. Relleno de huecos + suavizado normalizado.
4. Interpolación a resolución completa del proyector.
"""
from __future__ import annotations

import warnings
import numpy as np
import cv2
from scipy.interpolate import RBFInterpolator, RegularGridInterpolator
from scipy import ndimage

from .patterns import grid_size


def cell_centroids(dec: dict, width: int, height: int, cell: int,
                   min_pixels: int = 2, outlier_k: float = 3.0):
    gw, gh = grid_size(width, height, cell)
    valid = dec["valid"]
    ys, xs = np.nonzero(valid)
    idx = dec["cell_y"][ys, xs].astype(np.int64) * gw + dec["cell_x"][ys, xs]
    xs = xs.astype(np.float64)
    ys = ys.astype(np.float64)
    n = gw * gh

    def _acc(idx, xs, ys):
        cnt = np.bincount(idx, minlength=n).astype(np.float64)
        sx = np.bincount(idx, weights=xs, minlength=n)
        sy = np.bincount(idx, weights=ys, minlength=n)
        with np.errstate(invalid="ignore", divide="ignore"):
            mx, my = sx / cnt, sy / cnt
        return cnt, mx, my

    cnt, mx, my = _acc(idx, xs, ys)

    # Rechazo de píxeles mal decodificados lejos del centroide de su celda
    if len(idx):
        d = np.hypot(xs - mx[idx], ys - my[idx])
        spread = np.median(d) if len(d) else 1.0
        keep = d <= max(outlier_k * spread * 2.0, 2.0)
        idx, xs, ys = idx[keep], xs[keep], ys[keep]
        cnt, mx, my = _acc(idx, xs, ys)

    ok = cnt >= min_pixels
    cam = np.full((gh, gw, 2), np.nan, np.float64)
    cam[..., 0] = np.where(ok, mx, np.nan).reshape(gh, gw)
    cam[..., 1] = np.where(ok, my, np.nan).reshape(gh, gw)
    return cam, ok.reshape(gh, gw), cnt.reshape(gh, gw)


def cell_affine(dec: dict, width: int, height: int, cell: int, min_pixels: int = 8,
                neighbors: int = 1):
    """
    Con phase shifting cada píxel de cámara tiene una coordenada de proyector
    continua (proj_x, proj_y). Para cada celda se ajusta por mínimos cuadrados
    un modelo afín local  cam = a0 + A * (proj - centro)  con los píxeles de la
    celda y sus vecinas, y se evalúa en el centro: precisión sub-píxel y sin
    sesgo aunque la celda esté cortada por el borde de la pantalla.
    """
    gw, gh = grid_size(width, height, cell)
    n = gw * gh
    m = np.isfinite(dec["proj_x"])
    ys, xs = np.nonzero(m)
    xp = dec["proj_x"][m].astype(np.float64)
    yp = dec["proj_y"][m].astype(np.float64)
    ccx = dec["cell_x"][m].astype(np.int64)
    ccy = dec["cell_y"][m].astype(np.int64)
    camx, camy = xs.astype(np.float64), ys.astype(np.float64)

    def accumulate(w_px):
        acc = {k: np.zeros(n) for k in
               ("n", "dx", "dy", "dxx", "dxy", "dyy", "x", "xdx", "xdy", "y", "ydx", "ydy")}
        for oy in range(-neighbors, neighbors + 1):
            for ox in range(-neighbors, neighbors + 1):
                tx, ty = ccx + ox, ccy + oy
                ok = (tx >= 0) & (tx < gw) & (ty >= 0) & (ty < gh)
                idx = (ty * gw + tx)[ok]
                dx = xp[ok] - ((tx[ok] + 0.5) * cell - 0.5)
                dy = yp[ok] - ((ty[ok] + 0.5) * cell - 0.5)
                ww = w_px[ok]
                cx, cy = camx[ok], camy[ok]
                for k, v in (("n", ww), ("dx", ww * dx), ("dy", ww * dy), ("dxx", ww * dx * dx),
                             ("dxy", ww * dx * dy), ("dyy", ww * dy * dy), ("x", ww * cx),
                             ("xdx", ww * cx * dx), ("xdy", ww * cx * dy), ("y", ww * cy),
                             ("ydx", ww * cy * dx), ("ydy", ww * cy * dy)):
                    acc[k] += np.bincount(idx, weights=v, minlength=n)
        return acc

    def solve(acc):
        M = np.stack([np.stack([acc["n"], acc["dx"], acc["dy"]], -1),
                      np.stack([acc["dx"], acc["dxx"], acc["dxy"]], -1),
                      np.stack([acc["dy"], acc["dxy"], acc["dyy"]], -1)], -2)
        rx = np.stack([acc["x"], acc["xdx"], acc["xdy"]], -1)
        ry = np.stack([acc["y"], acc["ydx"], acc["ydy"]], -1)
        good = acc["n"] >= min_pixels * (2 * neighbors + 1) ** 2 * 0.25
        det = np.linalg.det(M)
        good &= np.abs(det) > 1e-6 * np.maximum(acc["n"], 1) ** 3
        M[~good] = np.eye(3)
        sx = np.linalg.solve(M, rx[..., None])[..., 0]
        sy = np.linalg.solve(M, ry[..., None])[..., 0]
        return sx, sy, good

    w1 = np.ones_like(xp)
    sx, sy, good = solve(accumulate(w1))
    # segunda pasada: descartar píxeles con residuo grande (reflejos, errores de fase)
    idx = ccy * gw + ccx
    dx = xp - ((ccx + 0.5) * cell - 0.5)
    dy = yp - ((ccy + 0.5) * cell - 0.5)
    rx = camx - (sx[idx, 0] + sx[idx, 1] * dx + sx[idx, 2] * dy)
    ry = camy - (sy[idx, 0] + sy[idx, 1] * dx + sy[idx, 2] * dy)
    r = np.hypot(rx, ry)
    thr = max(3.0 * np.median(r[good[idx]]) if good[idx].any() else 1.0, 0.75)
    sx, sy, good = solve(accumulate((r <= thr).astype(np.float64)))

    cnt = np.bincount(idx, minlength=n).astype(np.float64)
    cam = np.full((gh, gw, 2), np.nan)
    cam[..., 0] = np.where(good, sx[:, 0], np.nan).reshape(gh, gw)
    cam[..., 1] = np.where(good, sy[:, 0], np.nan).reshape(gh, gw)
    # sólo celdas que tienen píxeles propios (evita extrapolar desde vecinas)
    own = (cnt >= 1).reshape(gh, gw)
    return cam, good.reshape(gh, gw) & own, cnt.reshape(gh, gw)


def drop_partial_cells(ok: np.ndarray, cnt: np.ndarray, frac: float = 0.8,
                       edge_rings: int = 1):
    """
    Las celdas en el borde de la zona visible (borde de la pantalla, sombras)
    sólo se ven parcialmente y su centroide queda sesgado. Se descartan:
    - celdas con menos píxeles que `frac` * mediana local
    - los `edge_rings` anillos exteriores de la zona válida
    Luego se extrapolan desde el interior.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        local = ndimage.generic_filter(np.where(ok, cnt, np.nan), np.nanmedian,
                                       size=5, mode="nearest")
    keep = ok & (cnt >= frac * np.nan_to_num(local))
    if edge_rings > 0:
        keep &= ndimage.binary_erosion(ok, iterations=edge_rings, border_value=1)
    return keep


def clean_grid(cam: np.ndarray, ok: np.ndarray, max_dev_cells: float = 0.75):
    """Invalida celdas que se desvían de la mediana de sus vecinas."""
    gh, gw = ok.shape
    out_ok = ok.copy()
    # paso esperado de una celda en cámara (mediana de distancias vecinas)
    dx = np.hypot(*(cam[:, 1:] - cam[:, :-1]).transpose(2, 0, 1))
    dy = np.hypot(*(cam[1:] - cam[:-1]).transpose(2, 0, 1))
    step = np.nanmedian(np.concatenate([dx.ravel(), dy.ravel()]))
    if not np.isfinite(step) or step <= 0:
        return out_ok, 1.0
    for c in range(2):
        a = cam[..., c].copy()
        a_f = np.where(ok, a, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = ndimage.generic_filter(a_f, np.nanmedian, size=3, mode="nearest")
        bad = ok & (np.abs(a - med) > max_dev_cells * step * 1.5)
        out_ok &= ~bad
    return out_ok, float(step)


def fill_and_smooth(cam: np.ndarray, ok: np.ndarray, region_seed: np.ndarray,
                    smoothing: float = 1.0):
    """
    Ajusta un thin-plate spline (local, 64 vecinos) a las celdas buenas y lo
    evalúa en todo el grid: suaviza el ruido, rellena huecos y extrapola los
    bordes sin el sesgo de un filtro gaussiano.
    """
    gh, gw = ok.shape
    if ok.sum() < 10:
        raise RuntimeError("Muy pocas celdas válidas: revisa exposición / umbrales.")
    yy, xx = np.mgrid[0:gh, 0:gw]
    pts = np.stack([yy[ok], xx[ok]], 1).astype(np.float64)
    rbf = RBFInterpolator(pts, cam[ok], kernel="thin_plate_spline",
                          neighbors=min(64, len(pts)), smoothing=float(smoothing))
    q = np.stack([yy.ravel(), xx.ravel()], 1).astype(np.float64)
    grid = rbf(q).reshape(gh, gw, 2)

    # región útil: todo lo que se decodificó originalmente + huecos internos
    pad = 3  # padding: binary_closing erosiona los bordes del array
    r = np.pad(region_seed, pad, mode="edge")
    r = ndimage.binary_closing(r, iterations=2)[pad:-pad, pad:-pad]
    region = ndimage.binary_fill_holes(r)
    return grid, region


def dense_map(grid: np.ndarray, region: np.ndarray, width: int, height: int,
              cell: int):
    """Interpola el grid de centros de celda a cada píxel del proyector."""
    gh, gw = region.shape
    gy = (np.arange(gh) + 0.5) * cell - 0.5
    gx = (np.arange(gw) + 0.5) * cell - 0.5
    py, px = np.mgrid[0:height, 0:width].astype(np.float64)
    q = np.stack([py.ravel(), px.ravel()], 1)
    out = np.empty((height, width, 2), np.float32)
    for c in range(2):
        f = RegularGridInterpolator((gy, gx), grid[..., c], method="linear",
                                    bounds_error=False, fill_value=None)
        out[..., c] = f(q).reshape(height, width)
    # validez: celda más cercana dentro de la región
    ci = np.clip((py // cell).astype(int), 0, gh - 1)
    cj = np.clip((px // cell).astype(int), 0, gw - 1)
    valid = region[ci, cj]
    return out, valid


def projector_to_camera(dec: dict, width: int, height: int, cell: int,
                        min_pixels: int = 2, smoothing: float = 1.0):
    if "proj_x" in dec:      # phase shifting: ajuste afín sub-píxel (sin sesgo en bordes)
        cam, ok, cnt = cell_affine(dec, width, height, cell)
        ok1 = ok
    else:                    # sólo Gray code: centroides + descarte de celdas parciales
        cam, ok, cnt = cell_centroids(dec, width, height, cell, min_pixels)
        ok1 = drop_partial_cells(ok, cnt)
    ok2, step = clean_grid(cam, ok1)
    grid, region = fill_and_smooth(cam, ok2, ok, smoothing)
    pmap, valid = dense_map(grid, region, width, height, cell)
    stats = dict(cells_total=int(ok.size), cells_decoded=int(ok.sum()),
                 cells_after_clean=int(ok2.sum()), cam_px_per_cell=step)
    return pmap, valid, stats
