"""
Cálculo de blending (edge blend) y compensación de nivel de negro.

Todo se calcula en el espacio de cámara (común a todos los proyectores) y
luego se muestrea en cada proyector con su mapa proyector->cámara.

Pesos: w_i = d_i / sum_j d_j, con d_i = distancia al borde de la huella del
proyector i (transformada de distancia). La suma de pesos es 1 en todo punto
(en luz lineal). La curva gamma y la forma de la rampa se aplican en vivo en
el shader de TouchDesigner.

Negro: en zonas con menos proyectores superpuestos el negro es más oscuro.
L = (Nmax - n) / n indica cuánto "levantar" el negro en cada zona; el shader
lo multiplica por el parámetro Blacklevel.
"""
from __future__ import annotations

import numpy as np
import cv2


def clean_mask(mask: np.ndarray, k: int = 5) -> np.ndarray:
    m = mask.astype(np.uint8) * 255
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ker)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, ker, iterations=2)
    # quedarnos con la componente más grande y rellenar agujeros
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n > 1:
        big = 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])
        m = np.where(lab == big, 255, 0).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(m)
    cv2.drawContours(out, cnts, -1, 255, -1)
    return out > 0


def footprint(pmap: np.ndarray, cam_shape, lit: np.ndarray | None = None,
              lit_dilate: int = 15, step: int = 4) -> np.ndarray:
    """
    Huella exacta del proyector en cámara: el contorno de su marco (bordes del
    mapa proyector->cámara) rasterizado con precisión sub-píxel. Se intersecta
    con la zona iluminada dilatada para respetar sombras/oclusiones reales.
    """
    h, w = pmap.shape[:2]
    top = pmap[0, ::step]
    right = pmap[::step, -1]
    bottom = pmap[-1, ::-step]
    left = pmap[::-step, 0]
    poly = np.concatenate([top, pmap[0:1, -1], right, pmap[-1:, -1], bottom,
                           pmap[-1:, 0], left, pmap[0:1, 0]])
    poly = (poly + 0.5) * 16  # coordenadas de píxel -> sub-píxel (shift=4)
    poly = np.clip(poly, -1e6, 1e6).astype(np.int32)
    m = np.zeros(cam_shape, np.uint8)
    cv2.fillPoly(m, [poly], 255, lineType=cv2.LINE_8, shift=4)
    m = m > 0
    if lit is not None:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * lit_dilate + 1,) * 2)
        m &= cv2.dilate(lit.astype(np.uint8), ker) > 0
    return m


def edge_distance(mask: np.ndarray, pad: int = 64) -> np.ndarray:
    """Distancia al borde de la máscara. Los bordes de la IMAGEN no cuentan
    como borde del proyector (la cámara podría no ver todo el proyector)."""
    m = cv2.copyMakeBorder(mask.astype(np.uint8), pad, pad, pad, pad,
                           cv2.BORDER_REPLICATE)
    d = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    return d[pad:-pad, pad:-pad].astype(np.float32)


def camera_weights(masks: list[np.ndarray], max_width: float = 0.0,
                   power: float = 1.0):
    ds = []
    for m in masks:
        d = edge_distance(m)
        if max_width and max_width > 0:
            d = np.minimum(d, max_width)
        ds.append(np.power(d, power))
    s = np.sum(ds, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        ws = [np.where(s > 0, d / s, 0).astype(np.float32) for d in ds]
    return ws


def narrow_weights(weights: list[np.ndarray], k: float = 1.0) -> list[np.ndarray]:
    """Adelgaza k veces la rampa de mezcla alrededor de w=0.5 (centro de la costura).

    La rampa tal cual ocupa TODO el solape (w=0 en un borde, 1 en el otro), así
    que con un solape grande se ve como una banda muy ancha. Comprimirla en
    "ancho de w" la hace más fina y, al renormalizar (w/sum w), la suma de luz
    sigue siendo 1 en cada punto: se adelgaza SIN banda brillante, también si
    algún día se solapan tres proyectores.

    k=1 deja el mapa como estaba (0 y 1 se quedan en 0 y 1).
    """
    k = float(k or 1.0)
    if k <= 1.0:
        return weights
    out = [np.clip(0.5 + (w - 0.5) * k, 0.0, 1.0).astype(np.float32) for w in weights]
    s = np.sum(out, axis=0)
    return [np.where(s > 0, w / np.maximum(s, 1e-9), 0).astype(np.float32) for w in out]


def black_lift(masks: list[np.ndarray], screen_mask: np.ndarray | None,
               blur_px: float = 25.0):
    n = np.sum([m.astype(np.float32) for m in masks], axis=0)
    region = n > 0
    if screen_mask is not None:
        region &= screen_mask
    nmax = n[region].max() if region.any() else 1
    with np.errstate(invalid="ignore", divide="ignore"):
        L = np.where(n > 0, (nmax - n) / np.maximum(n, 1), 0).astype(np.float32)
    if blur_px > 0:
        L = cv2.GaussianBlur(L, (0, 0), blur_px)
    return L, int(nmax)


def sample(img: np.ndarray, pmap: np.ndarray, border=0.0) -> np.ndarray:
    """Muestrea una imagen de cámara en las posiciones del mapa del proyector."""
    return cv2.remap(img.astype(np.float32), pmap[..., 0], pmap[..., 1],
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                     borderValue=border)


def _norm_blur(img: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    """Desenfoque gaussiano normalizado: no oscurece cerca del borde de la máscara."""
    if sigma <= 0:
        return img
    m = mask.astype(np.float32)
    num = cv2.GaussianBlur(img * m, (0, 0), sigma)
    den = cv2.GaussianBlur(m, (0, 0), sigma)
    return np.where(den > 1e-4, num / np.maximum(den, 1e-6), 0).astype(np.float32)


def uniformity_gain(whites, blacks, masks, weights, screen_mask, cam_gamma=2.2,
                    blur_px=10.0, percentile=5.0, smooth_px=30.0):
    """
    Compensación de brillo medida con la cámara.
    En superficies convexas los lados reciben la luz de forma rasante y se
    ven más oscuros; además cada proyector puede tener distinto brillo.

    Luz que llega en cada punto (en luz lineal):  S(x) = sum_i w_i(x) * L_i(x)
    con L_i = blanco_i - negro_i, linealizado con la gamma de la cámara.
    Ganancia = clip(T / S, 0, 1), con T = percentil bajo de S en la pantalla,
    o sea, se atenúan las zonas brillantes hasta igualar las oscuras.

    Devuelve (gain HxW en [0,1], estadísticas).
    OJO: la viñeta de la webcam también se mide como "oscuro"; por eso en
    TouchDesigner la intensidad (Uniformity) se ajusta en vivo de 0 a 1.
    """
    S = np.zeros(masks[0].shape, np.float32)
    for wh, bk, m, w in zip(whites, blacks, masks, weights):
        lw = np.power(np.clip(wh / 255.0, 0, 1), cam_gamma)
        lb = np.power(np.clip(bk / 255.0, 0, 1), cam_gamma)
        L = _norm_blur(np.maximum(lw - lb, 0).astype(np.float32), m, blur_px)
        S += w * L
    region = screen_mask & (S > 0)
    if not region.any():
        return np.ones_like(S), {}
    T = float(np.percentile(S[region], percentile))
    gain = np.where(region, np.clip(T / np.maximum(S, 1e-9), 0, 1), 1).astype(np.float32)
    gain = np.where(region, _norm_blur(gain, region, smooth_px), 1).astype(np.float32)
    s = S[region]
    stats = {"brightness_min_over_max": round(float(np.percentile(s, 1) / np.percentile(s, 99)), 3),
             "light_kept_avg": round(float(np.mean(gain[region])), 3)}
    return gain, stats
