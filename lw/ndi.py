"""
Cámara NDI (p. ej. el teléfono con la app "NDI HX Camera" / "NDI Camera").

Usa cyndilib (pip install cyndilib), que trae el runtime de NDI incluido.

Config (bloque "camera"):
    "type": "ndi",
    "ndi_source": "IPHONE",      # parte del nombre de la fuente (sin distinguir mayúsculas);
                                 # vacío = la primera que aparezca
    "ndi_ips": "192.168.1.35",   # opcional: IP(s) del teléfono separadas por coma, para
                                 # cuando la red bloquea el descubrimiento automático (mDNS)
    "ndi_timeout_s": 15,         # espera máxima para encontrar la fuente
    "ndi_color": "uyvy"          # "uyvy" (luminancia directa, recomendado) o "bgr"
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import cv2

_NDI = None


def _prepare_env(ips: str | None):
    """Si hay IPs extra, crea un ndi-config.v1.json propio y apunta NDI_CONFIG_DIR a él.
    Debe llamarse ANTES de importar cyndilib (el runtime lee la config al iniciar)."""
    if not ips or "NDI_CONFIG_DIR" in os.environ:
        return
    d = Path(tempfile.gettempdir()) / "librewarp_ndi"
    d.mkdir(exist_ok=True)
    ips = ",".join(s.strip() for s in str(ips).replace(";", ",").split(",") if s.strip())
    (d / "ndi-config.v1.json").write_text(
        json.dumps({"ndi": {"networks": {"ips": ips, "discovery": ""}}}), encoding="utf-8")
    os.environ["NDI_CONFIG_DIR"] = str(d)
    print(f"NDI: buscando también en las IPs {ips}")


def _lib(ips: str | None = None):
    global _NDI
    if _NDI is None:
        _prepare_env(ips)
        try:
            from cyndilib.finder import Finder
            from cyndilib.receiver import Receiver
            from cyndilib.video_frame import VideoFrameSync
            from cyndilib.wrapper.ndi_recv import RecvColorFormat, RecvBandwidth
        except ImportError as e:
            raise RuntimeError(
                "Falta la librería NDI. Instálala con:\n"
                "    .venv\\Scripts\\python.exe -m pip install cyndilib\n"
                f"(error original: {e})") from e
        _NDI = dict(Finder=Finder, Receiver=Receiver, VideoFrameSync=VideoFrameSync,
                    RecvColorFormat=RecvColorFormat, RecvBandwidth=RecvBandwidth)
    return _NDI


def list_sources(timeout_s: float = 5.0, ips: str | None = None) -> list[str]:
    L = _lib(ips)
    f = L["Finder"]()
    f.open()
    try:
        t0 = time.time()
        names: list[str] = []
        while time.time() - t0 < timeout_s:
            f.wait_for_sources(0.5)
            names = f.get_source_names()
            if names and time.time() - t0 > 1.5:   # deja aparecer al resto
                break
        return list(names)
    finally:
        f.close()


def _pick(names: list[str], wanted: str) -> str | None:
    if not names:
        return None
    if not wanted:
        return names[0]
    w = wanted.lower()
    for n in names:
        if n.lower() == w:
            return n
    for n in names:
        if w in n.lower():
            return n
    return None


class NDICamera:
    """Misma interfaz que Webcam: shape, read_bgr, read_gray_new, capture_gray,
    set_exposure, open_driver_settings, release."""

    is_network = True

    def __init__(self, cfg: dict):
        self.cfg = cfg
        L = _lib(cfg.get("ndi_ips"))
        wanted = cfg.get("ndi_source", "") or ""
        timeout = float(cfg.get("ndi_timeout_s", 15))
        self.finder = L["Finder"]()
        self.finder.open()
        print(f"NDI: buscando fuente '{wanted or '(cualquiera)'}' ...")
        t0, name, names = time.time(), None, []
        while time.time() - t0 < timeout and name is None:
            self.finder.wait_for_sources(0.5)
            names = self.finder.get_source_names()
            name = _pick(names, wanted)
        if name is None:
            self.finder.close()
            raise RuntimeError(
                f"No se encontró la fuente NDI '{wanted}'. Fuentes vistas: {names or 'ninguna'}.\n"
                "  - ¿El teléfono y el PC están en la misma red? ¿La app está emitiendo?\n"
                "  - Permite python.exe en el Firewall de Windows (redes privadas).\n"
                "  - Si la Wi-Fi bloquea el descubrimiento, pon la IP del teléfono en \"ndi_ips\".")
        with self.finder.notify:
            src = self.finder.get_source(name)
        self.name = name
        color = str(cfg.get("ndi_color", "uyvy")).lower()
        cf = L["RecvColorFormat"].UYVY_BGRA if color == "uyvy" else L["RecvColorFormat"].BGRX_BGRA
        self.recv = L["Receiver"](color_format=cf, bandwidth=L["RecvBandwidth"].highest,
                                  recv_name="LibreWarp")
        self.vf = L["VideoFrameSync"]()
        self.recv.frame_sync.set_video_frame(self.vf)
        self.recv.connect_to(src)
        self._last_ts = None
        self._last_sig = None
        self._ptz = None
        # primer frame
        t0 = time.time()
        frame = None
        while time.time() - t0 < timeout:
            frame = self._grab()
            if frame is not None:
                break
            time.sleep(0.02)
        if frame is None:
            self.release()
            raise RuntimeError(f"Conectado a '{name}' pero no llegan imágenes.")
        self.shape = frame.shape[:2]
        fr = self._fps()
        print(f"Cámara NDI: '{name}'  {self.shape[1]}x{self.shape[0]}"
              + (f"  {fr:.2f} fps" if fr else "") + f"  formato={self._fmt}")
        if self.shape[0] < 1080:
            print("  AVISO: resolución < 1080p. En la app del teléfono elige 1080p o 4K "
                  "para más precisión.")

    # ------------------------------------------------------------------
    def _fps(self):
        try:
            fr = self.vf.get_frame_rate()
            return float(fr.numerator) / float(fr.denominator) if fr.denominator else None
        except Exception:
            return None

    def _grab(self):
        """Último frame disponible como (H,W,C) uint8 o (H,W,2) UYVY; None si aún no hay."""
        self.recv.frame_sync.capture_video()
        xres, yres = int(self.vf.xres), int(self.vf.yres)
        if min(xres, yres) == 0:
            return None
        buf = self.vf.get_array()
        stride = int(self.vf.get_line_stride()) or buf.size // yres
        if stride * yres > buf.size:
            stride = buf.size // yres
        bpp = max(1, min(4, stride // xres))
        img = buf[:stride * yres].reshape(yres, stride)[:, :xres * bpp].reshape(yres, xres, bpp)
        self._fmt = {2: "UYVY", 4: "BGRX"}.get(bpp, f"{bpp}bpp")
        self._bpp = bpp
        self._ts = self.vf.get_timestamp_posix()
        return img

    def _is_new(self, img) -> bool:
        ts = self._ts
        sig = img[::37, ::41].tobytes()
        new = (ts != self._last_ts) or (ts in (0, None) and sig != self._last_sig)
        if new:
            self._last_ts, self._last_sig = ts, sig
        return new

    def _to_bgr(self, img):
        if self._bpp == 2:
            return cv2.cvtColor(img, cv2.COLOR_YUV2BGR_UYVY)
        return np.ascontiguousarray(img[..., :3])

    def _to_gray(self, img):
        if self._bpp == 2:          # luminancia Y directa (sin pasar por RGB)
            return img[..., 1].astype(np.float32)
        return cv2.cvtColor(np.ascontiguousarray(img[..., :3]), cv2.COLOR_BGR2GRAY).astype(np.float32)

    def _wait_new(self, timeout_s: float):
        t0 = time.time()
        while True:
            img = self._grab()
            if img is not None and self._is_new(img):
                return img
            if time.time() - t0 > timeout_s:
                return None
            time.sleep(0.003)

    # ------------------------------------------------------------------
    def read_bgr(self):
        img = self._wait_new(0.25)
        if img is None:
            img = self._grab()
        if img is None:
            raise RuntimeError("NDI: sin imagen")
        return self._to_bgr(img)

    def read_gray_new(self, timeout_s: float = 1.0):
        img = self._wait_new(timeout_s)
        return None if img is None else self._to_gray(img)

    def capture_gray(self) -> np.ndarray:
        """Modo de sincronía 'fixed': descarta flush_frames y promedia avg_frames nuevos."""
        for _ in range(self.cfg.get("flush_frames", 4)):
            self._wait_new(0.5)
        n = max(1, self.cfg.get("avg_frames", 3))
        acc = None
        for _ in range(n):
            g = self.read_gray_new(1.0)
            if g is None:
                g = self._to_gray(self._grab())
            acc = g if acc is None else acc + g
        return acc / n

    # ---- controles de cámara: sólo si la fuente acepta PTZ (casi ninguna app de móvil) ----
    def ptz_supported(self) -> bool:
        if self._ptz is None:
            try:
                self._ptz = bool(self.recv.is_ptz_supported())
            except Exception:
                self._ptz = False
        return self._ptz

    def set_exposure(self, v):
        """v en la escala de webcam (-13..0) -> nivel 0..1 de NDI PTZ."""
        if not self.ptz_supported():
            print("NDI: esta fuente no acepta control remoto; ajusta la exposición en la app "
                  "del teléfono (bloquéala con AE/AF lock).")
            return
        level = float(np.clip((v + 13) / 13.0, 0, 1))
        self.recv.ptz.set_exposure_coarse(level)
        print(f"NDI PTZ: exposición {level:.2f}")

    def open_driver_settings(self):
        print("NDI: los ajustes de cámara se cambian en la app del teléfono.")

    def stats(self) -> str:
        try:
            return str(self.recv.video_stats)
        except Exception:
            return ""

    def release(self):
        for fn in (lambda: self.recv.disconnect(), lambda: self.finder.close()):
            try:
                fn()
            except Exception:
                pass
