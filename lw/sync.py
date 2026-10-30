"""
Sincronización proyector -> cámara robusta a latencia variable (NDI, Wi-Fi, webcams lentas).

Con una espera fija ("settle_ms") se corre el riesgo de capturar el patrón ANTERIOR
cuando la red se retrasa. En modo "auto", después de mostrar cada patrón:

  1. se esperan settle_ms (mínimo),
  2. se leen frames NUEVOS hasta ver que la imagen CAMBIÓ respecto a la captura anterior
     (fracción de píxeles que cambian más que el umbral de ruido),
  3. se espera a que la imagen sea ESTABLE (stable_frames frames seguidos casi iguales:
     descarta frames a medio cambiar, rolling shutter y la "recuperación" del códec),
  4. se promedian avg_frames frames nuevos.

Si en sync_timeout_ms no se detecta el cambio se avisa y se usa lo último recibido.

Config (bloque "camera"):
  "sync": "auto" | "fixed"        (por defecto "auto" en NDI, "fixed" en webcam)
  "sync_timeout_ms": 4000
  "sync_stable_frames": 2
  "sync_change_frac": 0.003       fracción mínima de píxeles que deben cambiar
"""
from __future__ import annotations

import time

import numpy as np
import cv2


class SyncedCapture:
    def __init__(self, cam, cfg: dict, log=print):
        self.cam = cam
        self.cfg = cfg
        self.log = log
        self.timeout = cfg.get("sync_timeout_ms", 4000) / 1000.0
        self.stable_frames = max(1, int(cfg.get("sync_stable_frames", 2)))
        self.change_frac = float(cfg.get("sync_change_frac", 0.003))
        self.avg = max(1, int(cfg.get("avg_frames", 3)))
        self.prev_small = None
        self.pix_thr = 6.0
        self.latencies: list[float] = []
        self.timeouts = 0
        self.t_show = None

    # -- utilidades --------------------------------------------------------
    @staticmethod
    def small(g: np.ndarray) -> np.ndarray:
        h, w = g.shape[:2]
        s = cv2.resize(g, (max(8, w // 4), max(8, h // 4)), interpolation=cv2.INTER_AREA)
        return cv2.GaussianBlur(s, (3, 3), 0)

    def frac(self, a, b) -> float:
        return float(np.mean(np.abs(a - b) > self.pix_thr))

    def _new(self, timeout_s):
        return self.cam.read_gray_new(timeout_s)

    # -- calibración del ruido con la escena quieta ----------------------
    def calibrate(self, n: int = 12):
        frames = []
        t0 = time.time()
        while len(frames) < n and time.time() - t0 < 5:
            g = self._new(1.0)
            if g is not None:
                frames.append(self.small(g))
        if len(frames) < 3:
            raise RuntimeError("La cámara no entrega frames nuevos (¿está congelada la imagen?)")
        st = np.stack(frames)
        per_pair = [float(np.percentile(np.abs(st[k + 1] - st[k]), 99.5)) for k in range(len(st) - 1)]
        sigma = float(np.median(per_pair))           # la mediana ignora un frame de transición
        self.pix_thr = max(5.0, 2.0 * sigma)
        self.prev_small = np.median(st[len(st) // 2:], axis=0)
        self.log(f"Sincronía auto: ruido p99.5={sigma:.1f} niveles -> umbral por píxel "
                 f"{self.pix_thr:.1f}, mínimo {self.change_frac*100:.2f}% de píxeles")

    def mark_shown(self):
        self.t_show = time.time()

    # -- captura sincronizada --------------------------------------------
    def capture(self) -> np.ndarray:
        t_show = self.t_show or time.time()
        deadline = t_show + self.timeout
        changed = self.prev_small is None
        last, n_stable, got = None, 0, None
        while time.time() < deadline:
            g = self._new(max(0.05, deadline - time.time()))
            if g is None:
                break
            s = self.small(g)
            if not changed:
                if self.frac(s, self.prev_small) >= self.change_frac:
                    changed = True
                    self.latencies.append(time.time() - t_show)
                    last, n_stable = s, 0
                continue
            if last is not None and self.frac(s, last) < self.change_frac:
                n_stable += 1
                if n_stable >= self.stable_frames - 1:
                    got = g
                    break
            else:
                n_stable = 0
            last = s
        if got is None:
            self.timeouts += 1
            why = "no se detectó el cambio de patrón" if not changed else "la imagen no se estabiliza"
            self.log(f"\n  AVISO sincronía: {why} en {self.timeout:.1f}s; se usa la última imagen. "
                     "(¿cámara movida, auto-exposición activa o proyector fuera de cuadro?)")
        # promedio de frames nuevos ya estables
        acc, k = (got.astype(np.float32) if got is not None else None), (1 if got is not None else 0)
        while k < self.avg:
            g = self._new(1.0)
            if g is None:
                break
            acc = g if acc is None else acc + g
            k += 1
        if acc is None:
            raise RuntimeError("Sin imagen de la cámara")
        img = acc / k
        self.prev_small = self.small(img)
        return img

    def summary(self) -> str:
        if not self.latencies:
            return ""
        a = np.array(self.latencies) * 1000
        return (f"latencia patrón->cámara: mediana {np.median(a):.0f} ms, máx {a.max():.0f} ms"
                + (f", {self.timeouts} avisos" if self.timeouts else ""))
