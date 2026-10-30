"""
Generación y decodificación de patrones Gray code por "celdas".

En lugar de codificar cada píxel del proyector (las líneas de 1 px no las
resuelve una webcam), codificamos celdas de `cell` x `cell` píxeles.
Cada píxel de cámara recibe el índice (cx, cy) de la celda de proyector que
lo ilumina. Después promediamos las posiciones de cámara de cada celda
(centroide) para obtener precisión sub-píxel.

Memoria: `make_sequence` entrega los patrones de uno en uno (generador) y
`Decoder` los decodifica conforme llegan, guardando sólo acumuladores. Antes se
retenían las 46 capturas completas en RAM (>1,5 GB con cámara 4K).
"""
from __future__ import annotations

import math
import numpy as np
import cv2


def n_bits(n_cells: int) -> int:
    return max(1, math.ceil(math.log2(max(2, n_cells))))


def grid_size(width: int, height: int, cell: int) -> tuple[int, int]:
    return math.ceil(width / cell), math.ceil(height / cell)


def sequence_length(width: int, height: int, cell: int, phase_steps: int = 8) -> int:
    """Nº de patrones que entregará make_sequence, sin construirlos."""
    gw, gh = grid_size(width, height, cell)
    n = 2 + 2 * (n_bits(gw) + n_bits(gh))
    if phase_steps and phase_steps >= 3:
        n += 2 * phase_steps
    return n


def make_sequence(width: int, height: int, cell: int, intensity: int = 255,
                  phase_steps: int = 8, phase_period: int = 32, color: bool = True):
    """
    Generador de (nombre, imagen); sólo materializa un patrón a la vez.

    Con `color=True` (por defecto) los patrones son BGR con dos tonos de
    luminancia muy distinta: amarillo (claro) y azul (oscuro). El código Gray
    se codifica en esos dos tonos, de modo que la cámara ve el mismo patrón de
    luminancia que con blanco/negro pero proyectado en color (amarillo/azul),
    con contraste de sobra y sin saturar el blanco. Las sinusoides de fase van
    tintadas: amarillas en X y cian en Y.

    Con `color=False` se entregan patrones en escala de grises de 1 canal
    (comportamiento anterior).

    Orden: black, white, luego para cada bit de columnas y filas el patrón
    y su inverso (decodificación robusta comparando pos vs neg) y, si
    phase_steps > 0, N sinusoides desplazadas por eje (phase shifting) que dan
    una coordenada de proyector continua (sub-píxel) en cada píxel de cámara.
    """
    gw, gh = grid_size(width, height, cell)
    bx, by = n_bits(gw), n_bits(gh)

    cx = (np.arange(width) // cell).astype(np.int64)
    cy = (np.arange(height) // cell).astype(np.int64)
    gx = cx ^ (cx >> 1)
    gy = cy ^ (cy >> 1)

    if color:
        # BGR. Amarillo ~226 de luminancia, azul ~29: contraste ~197.
        bright = np.array((0, intensity, intensity), np.uint8)   # amarillo
        dark = np.array((intensity, 0, 0), np.uint8)             # azul

        def two_tone(mask: np.ndarray) -> np.ndarray:
            """mask True -> amarillo, False -> azul (imagen BGR HxWx3)."""
            out = np.empty(mask.shape + (3,), np.uint8)
            for c in range(3):
                out[..., c] = np.where(mask, bright[c], dark[c])
            return out

        yield "black", np.zeros((height, width, 3), np.uint8)
        yield "white", np.full((height, width, 3), intensity, np.uint8)

        for k in range(bx):
            bit = ((gx >> (bx - 1 - k)) & 1).astype(bool)
            mask = np.broadcast_to(bit[None, :], (height, width))
            yield f"col{k:02d}p", two_tone(mask)
            yield f"col{k:02d}n", two_tone(~mask)

        for k in range(by):
            bit = ((gy >> (by - 1 - k)) & 1).astype(bool)
            mask = np.broadcast_to(bit[:, None], (height, width))
            yield f"row{k:02d}p", two_tone(mask)
            yield f"row{k:02d}n", two_tone(~mask)

        if phase_steps and phase_steps >= 3:
            xs = np.arange(width, dtype=np.float64)
            ys = np.arange(height, dtype=np.float64)
            for k in range(phase_steps):
                sh = 2 * np.pi * k / phase_steps
                v = np.round((0.5 + 0.5 * np.cos(2 * np.pi * xs / phase_period - sh))
                             * intensity).astype(np.uint8)
                img = np.zeros((height, width, 3), np.uint8)
                img[..., 1] = np.broadcast_to(v[None, :], (height, width))  # G
                img[..., 2] = np.broadcast_to(v[None, :], (height, width))  # R
                yield f"phx{k:02d}", img                                  # amarillo
            for k in range(phase_steps):
                sh = 2 * np.pi * k / phase_steps
                v = np.round((0.5 + 0.5 * np.cos(2 * np.pi * ys / phase_period - sh))
                             * intensity).astype(np.uint8)
                img = np.zeros((height, width, 3), np.uint8)
                img[..., 0] = np.broadcast_to(v[:, None], (height, width))  # B
                img[..., 1] = np.broadcast_to(v[:, None], (height, width))  # G
                yield f"phy{k:02d}", img                                  # cian
        return

    # --- escala de grises (comportamiento anterior) ---
    yield "black", np.zeros((height, width), np.uint8)
    yield "white", np.full((height, width), intensity, np.uint8)

    for k in range(bx):
        bit = ((gx >> (bx - 1 - k)) & 1).astype(np.uint8)
        img = np.broadcast_to((bit * intensity)[None, :], (height, width)).copy()
        yield f"col{k:02d}p", img
        yield f"col{k:02d}n", (intensity - img).astype(np.uint8)

    for k in range(by):
        bit = ((gy >> (by - 1 - k)) & 1).astype(np.uint8)
        img = np.broadcast_to((bit * intensity)[:, None], (height, width)).copy()
        yield f"row{k:02d}p", img
        yield f"row{k:02d}n", (intensity - img).astype(np.uint8)

    if phase_steps and phase_steps >= 3:
        xs = np.arange(width, dtype=np.float64)
        ys = np.arange(height, dtype=np.float64)
        for k in range(phase_steps):
            sh = 2 * np.pi * k / phase_steps
            row = 0.5 + 0.5 * np.cos(2 * np.pi * xs / phase_period - sh)
            yield (f"phx{k:02d}",
                   np.broadcast_to(np.round(row * intensity).astype(np.uint8)[None, :],
                                   (height, width)).copy())
        for k in range(phase_steps):
            sh = 2 * np.pi * k / phase_steps
            col = 0.5 + 0.5 * np.cos(2 * np.pi * ys / phase_period - sh)
            yield (f"phy{k:02d}",
                   np.broadcast_to(np.round(col * intensity).astype(np.uint8)[:, None],
                                   (height, width)).copy())


class Decoder:
    """
    Decodifica los patrones conforme llegan, con memoria O(1) en el nº de
    patrones: sólo acumula el Gray code y las componentes seno/coseno de la
    fase, en lugar de guardar las 46 capturas.

        d = Decoder(width, height, cell, phase_steps=8)
        for name, img in make_sequence(...):
            d.add(name, img)
        dec = d.result()          # mismo dict que devolvía decode()

    Los patrones de un eje deben llegar en orden (bit 0 primero y cada
    positivo seguido de su inverso), que es justo el orden de make_sequence.
    """

    def __init__(self, width: int, height: int, cell: int, min_contrast: float = 10.0,
                 white_threshold: float = 25.0, phase_period: int = 32,
                 phase_steps: int = 8):
        self.width, self.height, self.cell = int(width), int(height), int(cell)
        self.min_contrast = float(min_contrast)
        self.white_threshold = float(white_threshold)
        self.phase_period = float(phase_period)
        self.phase_steps = int(phase_steps or 0)
        self.gw, self.gh = grid_size(self.width, self.height, self.cell)
        self.bx, self.by = n_bits(self.gw), n_bits(self.gh)
        self.white = None
        self.black = None
        self.n_added = 0
        self._g = {}          # eje -> acumulador Gray
        self._bad = {}        # eje -> OR de bits con poco contraste
        self._bit = {"col": 0, "row": 0}
        self._pend = {}       # (eje, k) -> patrón positivo aún sin su inverso
        self._S = {}
        self._C = {}
        self._phase = set()

    # -- entrada ----------------------------------------------------------
    def add(self, name: str, img: np.ndarray):
        # img puede ser HxW (escala de grises) o HxWx3 (BGR). Para los patrones
        # de color usamos la luminancia (mismos pesos que cv2.COLOR_BGR2GRAY),
        # así el decodificado es idéntico al de los patrones blanco/negro.
        a = np.asarray(img, dtype=np.float32)
        if a.ndim == 3:
            b, g, r = a[..., 0], a[..., 1], a[..., 2]
            a = 0.114 * b + 0.587 * g + 0.299 * r
        axis = name[:3]
        if name == "black":
            self.black = a
        elif name == "white":
            self.white = a
        elif axis in ("col", "row"):
            k = int(name[3:5])
            if name.endswith("p"):
                self._pend[(axis, k)] = a
            else:
                p = self._pend.pop((axis, k), None)
                if p is None:
                    raise RuntimeError(f"llegó '{name}' sin su patrón positivo '{name[:-1]}p'")
                self._gray_bit(axis, k, p - a)
        elif axis in ("phx", "phy"):
            self._phase_step(axis, int(name[3:5]), a)
        else:
            raise ValueError(f"patrón desconocido: '{name}'")
        self.n_added += 1
        return self

    def _gray_bit(self, axis: str, k: int, d: np.ndarray):
        if k != self._bit[axis]:
            raise RuntimeError(f"los patrones de '{axis}' llegaron desordenados: "
                               f"esperaba el bit {self._bit[axis]} y llegó el {k}")
        self._bit[axis] += 1
        if axis not in self._g:
            self._g[axis] = np.zeros(d.shape, np.int32)
            self._bad[axis] = np.zeros(d.shape, np.uint8)
        self._g[axis] = (self._g[axis] << 1) | (d > 0).astype(np.int32)
        self._bad[axis] |= (np.abs(d) <= self.min_contrast).astype(np.uint8)

    def _phase_step(self, axis: str, k: int, a: np.ndarray):
        if self.phase_steps < 3:
            return
        ang = 2.0 * np.pi * k / self.phase_steps
        if axis not in self._S:
            self._S[axis] = np.zeros(a.shape, np.float32)
            self._C[axis] = np.zeros(a.shape, np.float32)
        self._S[axis] += a * np.float32(np.sin(ang))
        self._C[axis] += a * np.float32(np.cos(ang))
        self._phase.add(axis)

    # -- salida -----------------------------------------------------------
    def result(self) -> dict:
        if self.white is None or self.black is None:
            raise RuntimeError("El decodificador necesita los patrones 'black' y 'white'.")
        white, black = self.white, self.black
        lit = (white - black) > self.white_threshold
        valid = lit.copy()

        cxi = gray_to_binary(self._gray("col", self.bx))
        cyi = gray_to_binary(self._gray("row", self.by))
        valid &= (self._bad["col"] == 0) & (self._bad["row"] == 0)
        valid &= (cxi < self.gw) & (cyi < self.gh)

        # Para el caso sin fase (phase_steps == 0), no hay coherencia de fase.
        # `valid` ya es la zona iluminada por el proyector: usamos `pv = valid`
        # para que no se pierda el 90% o más del encuadre. El decodificado de
        # las celdas Gray sigue estando en cell_x/cell_y, y la imagen blanca/negra
        # en white/black.
        pv = valid

        cell_x = np.where(valid, cxi, -1).astype(np.int32)
        cell_y = np.where(valid, cyi, -1).astype(np.int32)
        out = dict(cell_x=cell_x, cell_y=cell_y, valid=valid, lit=lit,
                   white=white, black=black)

        if "phx" in self._phase:
            fx, okx = self._phase_result("phx")
            fy, oky = self._phase_result("phy")
            xp = unwrap(fx, cxi, self.cell, self.phase_period)
            yp = unwrap(fy, cyi, self.cell, self.phase_period)
            # coherencia: la posición continua debe caer en (o junto a) la celda decodificada
            coh = (np.abs(xp - (cxi * self.cell + self.cell / 2 - 0.5)) <= self.cell * 0.75) & \
                  (np.abs(yp - (cyi * self.cell + self.cell / 2 - 0.5)) <= self.cell * 0.75)
            pv = valid & okx & oky & coh
            out["proj_x"] = np.where(pv, xp, np.nan).astype(np.float32)
            out["proj_y"] = np.where(pv, yp, np.nan).astype(np.float32)
        return out

    def _gray(self, axis: str, nb: int) -> np.ndarray:
        if axis not in self._g or self._bit[axis] < nb:
            raise RuntimeError(f"faltan patrones de '{axis}': llegaron {self._bit[axis]} "
                               f"bits y hacen falta {nb}")
        return self._g[axis]

    def _phase_result(self, axis: str):
        if axis not in self._S:
            raise RuntimeError(f"faltan los patrones de fase '{axis}'")
        S, C = self._S[axis], self._C[axis]
        phi = np.mod(np.arctan2(S, C), 2 * np.pi)
        mod = 2.0 / self.phase_steps * np.hypot(S, C)
        return phi / (2 * np.pi) * self.phase_period, mod > self.min_contrast / 2


def unwrap(frac: np.ndarray, cell_idx: np.ndarray, cell: int, period: float):
    """Elige el periodo cuya posición queda más cerca del centro de la celda del Gray code."""
    center = cell_idx * cell + cell / 2.0 - 0.5
    n = np.round((center - frac) / period)
    return frac + n * period


def gray_to_binary(g: np.ndarray) -> np.ndarray:
    b = g.copy()
    s = g >> 1
    while np.any(s):
        b ^= s
        s >>= 1
    return b


def _ordered_names(captures: dict, bx: int, by: int, phase_steps: int):
    """Nombres en el orden canónico de make_sequence (sólo los que estén presentes)."""
    if "black" in captures:
        yield "black"
    if "white" in captures:
        yield "white"
    for axis, nb in (("col", bx), ("row", by)):
        for k in range(nb):
            for suf in ("p", "n"):
                nm = f"{axis}{k:02d}{suf}"
                if nm in captures:
                    yield nm
    if phase_steps and phase_steps >= 3:
        for axis in ("phx", "phy"):
            for k in range(phase_steps):
                nm = f"{axis}{k:02d}"
                if nm in captures:
                    yield nm


def decode(captures: dict, width: int, height: int, cell: int,
           min_contrast: float = 10.0, white_threshold: float = 25.0,
           phase_period: int = 32, phase_steps: int | None = None):
    """
    Versión por lotes de Decoder (comodidad y compatibilidad). Si no se indica
    `phase_steps` se deduce del nº de patrones 'phx' presentes.

    Devuelve dict con:
      cell_x, cell_y : int32 HxW (índice de celda, -1 si inválido)
      valid          : bool HxW
      lit            : bool HxW (píxel iluminado por el proyector: white-black)
      white, black   : float32
      proj_x, proj_y : float32 HxW (coordenada continua de proyector, si hay fase)
    """
    gw, gh = grid_size(width, height, cell)
    bx, by = n_bits(gw), n_bits(gh)
    if phase_steps is None:
        phase_steps = sum(1 for k in captures if k.startswith("phx"))
    dec = Decoder(width, height, cell, min_contrast, white_threshold,
                  phase_period, phase_steps)
    for name in _ordered_names(captures, bx, by, dec.phase_steps):
        dec.add(name, captures[name])
    return dec.result()
