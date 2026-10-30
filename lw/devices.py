"""
Hardware real: una ventana a pantalla completa por DISPLAY de Windows (con un
QuadHead2Go hay un solo display grande que contiene los 3-4 proyectores como
regiones) y la webcam. También define la interfaz "Rig" del escaneo.
"""
from __future__ import annotations

import numpy as np
import cv2


def list_monitors():
    try:
        from screeninfo import get_monitors
    except ImportError:
        print("Instala 'screeninfo' (pip install screeninfo) para listar monitores.")
        return []
    mons = get_monitors()
    for i, m in enumerate(mons):
        print(f"[{i}] {m.name}: x={m.x} y={m.y} {m.width}x{m.height} "
              f"primary={getattr(m, 'is_primary', '?')}")
    print("\nCon un QuadHead2Go verás UN monitor grande: 7680x1080 (layout 4x1),") 
    print("5760x1080 (layout 3x1) o 3840x2160 (layout 2x2). El layout es nº de")
    print("cabezas x filas: 7680/1920 = 4 columnas, no 3.")
    print("Copia su x, y, width, height en \"displays\"[0] de config.json. Los proyectores van")
    print("todos con \"display\": 0 y \"output\": 1, 2, 3 (el número [N] de arriba NO va en config).")
    return mons


def dshow_device_names() -> list[str]:
    """Nombres amigables de las cámaras DirectShow ("Camo", "NDI Webcam Video 1"...).

    Usa pygrabber; si no está instalado o falla, devuelve [] y el menú muestra
    solo índice/backend/resolución. El orden coincide con los índices DirectShow
    de OpenCV en Windows.
    """
    try:
        from pygrabber.dshow_graph import FilterGraph
        return [str(n) for n in FilterGraph().get_input_devices()]
    except Exception:
        return []


def probe_video_devices(max_index: int = 8) -> list[dict]:
    """Prueba los índices 0..max_index-1 y devuelve los que abren.

    Cada elemento: {index, backend ("dshow"|"any"), width, height, ok}. No
    imprime nada: lo usan el comando `list_devices` y el menú del wizard. No se
    detiene en el primer hueco (si el 2 no abre y el 3 sí, el 3 aparece).
    """
    found = []
    for idx in range(int(max_index)):
        for bname, backend in (("dshow", cv2.CAP_DSHOW), ("any", cv2.CAP_ANY)):
            cap = cv2.VideoCapture(idx, backend)
            if not cap.isOpened():
                cap.release()
                continue
            ok, frame = cap.read()
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or (frame.shape[1] if ok else 0)
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or (frame.shape[0] if ok else 0)
            cap.release()
            if not ok and not (w and h):
                continue      # abre pero no da imagen: no sirve para calibrar
            found.append({"index": idx, "backend": bname, "width": w, "height": h,
                          "ok": bool(ok)})
            break
    names = dshow_device_names()
    for d in found:
        d["name"] = names[d["index"]] if 0 <= d["index"] < len(names) else ""
    return found


def list_video_devices(max_index: int = 8):
    """Lista todos los dispositivos de video conectados (webcams, capturadoras...).

    Recorre los índices 0..max_index-1 probando DirectShow (Windows) y, si no
    abre, el backend automático. No se detiene en el primer hueco: si el índice
    2 no abre pero el 3 sí, el 3 también aparece. Devuelve los índices que
    abren y entregan imagen; el índice es el que va en "camera": {"index": N}.
    """
    try:
        from screeninfo import get_monitors
        print("--- Monitores del sistema ---")
        for i, m in enumerate(get_monitors()):
            print(f"[{i}] {m.name}: {m.width}x{m.height} (pos: {m.x},{m.y})")
    except ImportError:
        print("--- Monitores: instala 'screeninfo' para verlos ---")

    print("\n--- Dispositivos de video ---")
    devs = probe_video_devices(max_index)
    for d in devs:
        bname = "DirectShow" if d["backend"] == "dshow" else "auto"
        label = f" · {d['name']}" if d.get("name") else ""
        print(f"[{d['index']}] {bname}: {d['width']}x{d['height']}{label}"
              + ("" if d["ok"] else " (sin imagen)"))
    print(f"\nDispositivos con imagen: {[d['index'] for d in devs if d['ok']] or 'ninguno'}")
    print("Para usar uno: en config.json, camera.type = \"webcam\" e index = el número [N].")
    return [d["index"] for d in devs]


def compose(bufs: list[np.ndarray], projs: list[dict], i: int, img: np.ndarray):
    """Copia la imagen del proyector i en su región del lienzo de su display."""
    p = projs[i]
    x, y, w, h = p["rect"]
    if img.shape[0] != h or img.shape[1] != w:
        raise ValueError(f"imagen {img.shape[:2]} != proyector {h}x{w}")
    buf = bufs[p["display"]]
    buf[y:y + h, x:x + w] = img[..., None] if img.ndim == 2 else img


class DisplayCanvas:
    """Una ventana sin bordes a pantalla completa por display físico."""

    def __init__(self, cfg: dict, open_windows: bool = True):
        self.displays = cfg["displays"]
        self.projs = cfg["projectors"]
        self.bufs = [np.zeros((d["height"], d["width"], 3), np.uint8) for d in self.displays]
        self.names = [f"LW_display{i}" for i in range(len(self.displays))]
        if open_windows:
            for name, d in zip(self.names, self.displays):
                cv2.namedWindow(name, cv2.WINDOW_NORMAL)
                cv2.moveWindow(name, int(d["x"]), int(d["y"]))
                cv2.resizeWindow(name, int(d["width"]), int(d["height"]))
                cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            self.flush(1)

    def set(self, i: int, img: np.ndarray):
        compose(self.bufs, self.projs, i, img)

    def clear(self, value: int = 0):
        for b in self.bufs:
            b[:] = value

    def flush(self, wait_ms: int = 1):
        """Dibuja y espera. wait_ms<=0 significa "esperar a que se pulse algo".

        OJO: eso no puede resolverse con cv2.waitKey(1), que deja pasar 1 ms y
        devuelve -1: identify/verify se cerrarían al instante sin que se vea nada.
        """
        for name, b in zip(self.names, self.bufs):
            cv2.imshow(name, b)
        wait_ms = int(wait_ms)
        if wait_ms <= 0:
            while True:
                k = cv2.waitKey(200)
                if k >= 0:
                    return k
        return cv2.waitKey(max(1, wait_ms))

    def close(self):
        for n in self.names:
            try:
                cv2.destroyWindow(n)
            except cv2.error:
                pass


def identify_image(p: dict, i: int) -> np.ndarray:
    """Imagen para identificar cada salida: número grande, nombre y marco."""
    w, h = p["width"], p["height"]
    img = np.zeros((h, w, 3), np.uint8)
    cols = [(60, 60, 230), (60, 200, 60), (230, 120, 40), (40, 200, 230)]
    c = cols[i % len(cols)]
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), c, 12)
    cv2.line(img, (0, 0), (w, h), c, 2)
    cv2.line(img, (w, 0), (0, h), c, 2)
    txt = str(i)
    s = h / 120
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, s, int(s * 2))
    cv2.putText(img, txt, ((w - tw) // 2, (h + th) // 2), cv2.FONT_HERSHEY_SIMPLEX, s, c, int(s * 2))
    label = f"proyector {i}: {p['name']}  rect={p['rect']}"
    cv2.putText(img, label, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    return img


class Webcam:
    BACKENDS = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF,
                "v4l2": cv2.CAP_V4L2, "avfoundation": cv2.CAP_AVFOUNDATION,
                "any": cv2.CAP_ANY}

    def __init__(self, cfg: dict):
        self.cfg = cfg
        be = self.BACKENDS.get(cfg.get("backend", "any"), cv2.CAP_ANY)
        self.cap = cv2.VideoCapture(cfg.get("index", 0), be)
        if not self.cap.isOpened():
            raise RuntimeError("No se pudo abrir la cámara")
        c = self.cap
        c.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.get("width", 1920))
        c.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.get("height", 1080))
        c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # lo que el driver trae al abrir: es lo que dejó su diálogo (⚙️) la última vez
        self.driver_before = self._driver_vals()
        self.requested: dict[str, float] = {}
        if not cfg.get("auto_exposure", False):
            # DirectShow: 0.25 = manual en muchos drivers; V4L2: 1 = manual
            c.set(cv2.CAP_PROP_AUTO_EXPOSURE, cfg.get("auto_exposure_off_value", 0.25))
            if cfg.get("exposure") is not None:
                self.requested["exposure"] = float(cfg["exposure"])
                c.set(cv2.CAP_PROP_EXPOSURE, cfg["exposure"])
        if cfg.get("gain") is not None:
            self.requested["gain"] = float(cfg["gain"])
            c.set(cv2.CAP_PROP_GAIN, cfg["gain"])
        if cfg.get("focus") is not None:
            c.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            c.set(cv2.CAP_PROP_FOCUS, cfg["focus"])
        c.set(cv2.CAP_PROP_AUTO_WB, 0)
        for _ in range(10):
            c.read()
        ok, f = c.read()
        if not ok:
            raise RuntimeError("La cámara no entrega imágenes")
        self.shape = f.shape[:2]
        a = self._driver_vals()
        print(f"Cámara: {f.shape[1]}x{f.shape[0]} · exposición {a['exposure']:.1f} (log2) "
              f"· ganancia {a['gain']:.0f} · autoexp. {a['auto_exposure']:.2f}")
        st = self.driver_state()
        if st["ignored"]:
            print("  AVISO exposición: el driver NO aceptó " + ", ".join(
                f"{k}={v:g}" for k, v in st["requested"].items() if k in st["ignored"])
                + " (no es ajustable por software en este driver).")
        if st["overwritten"]:
            print("  AVISO exposición: config.json fuerza " + ", ".join(
                f"{k}={v:g}" for k, v in st["requested"].items() if k in st["overwritten"])
                + " y el driver traía " + ", ".join(
                f"{k}={st['before'][k]:g}" for k in st["overwritten"])
                + " (lo puesto en su diálogo).\n     El escaneo usará lo de config.json.")

    # -- estado del driver de la cámara ------------------------------------
    def _driver_vals(self) -> dict:
        c = self.cap
        return {"exposure": float(c.get(cv2.CAP_PROP_EXPOSURE)),
                "gain": float(c.get(cv2.CAP_PROP_GAIN)),
                "auto_exposure": float(c.get(cv2.CAP_PROP_AUTO_EXPOSURE))}

    def driver_state(self) -> dict:
        """Exposición/ganancia REALES, las que pidió config.json y las que traía el
        driver al abrir (las de su diálogo ⚙️). Sirve para explicar por qué dos
        escaneos salen iguales aunque en el diálogo cambies la exposición: el
        escaneo vuelve a aplicar config.json al abrir la cámara.

        Devuelve {actual, requested, before, ignored, overwritten}.
        """
        act = self._driver_vals()
        req = getattr(self, "requested", {}) or {}
        before = getattr(self, "driver_before", {}) or {}
        tol = {"exposure": 0.5, "gain": 2.0}
        ignored = [k for k, v in req.items() if abs(act.get(k, v) - v) > tol.get(k, 0.5)]
        overwritten = [k for k, v in req.items()
                       if k in before and abs(before[k] - v) > tol.get(k, 0.5)]
        return {"actual": act, "requested": req, "before": before,
                "ignored": ignored, "overwritten": overwritten}

    def set_exposure(self, v):
        self.cap.set(cv2.CAP_PROP_EXPOSURE, v)

    def open_driver_settings(self):
        self.cap.set(cv2.CAP_PROP_SETTINGS, 1)  # sólo DirectShow (Windows)

    def read_bgr(self):
        ok, f = self.cap.read()
        if not ok:
            raise RuntimeError("Fallo de captura")
        return f

    def read_gray_new(self, timeout_s: float = 1.0):
        return cv2.cvtColor(self.read_bgr(), cv2.COLOR_BGR2GRAY).astype(np.float32)

    def capture_gray(self) -> np.ndarray:
        for _ in range(self.cfg.get("flush_frames", 4)):
            self.cap.grab()
        acc = None
        n = max(1, self.cfg.get("avg_frames", 3))
        for _ in range(n):
            f = cv2.cvtColor(self.read_bgr(), cv2.COLOR_BGR2GRAY).astype(np.float32)
            acc = f if acc is None else acc + f
        return acc / n

    def release(self):
        self.cap.release()


def open_camera(cam_cfg: dict):
    """Webcam (USB / DirectShow) o cámara NDI según camera.type."""
    if str(cam_cfg.get("type", "webcam")).lower() == "ndi":
        from .ndi import NDICamera
        return NDICamera(cam_cfg)
    return Webcam(cam_cfg)


class RealRig:
    """Interfaz común: n_proj, proj_size(i), cam_shape, show_only, show_all, capture()."""

    def __init__(self, cfg: dict, canvas=None, cam=None, log=print):
        self.cfg = cfg
        cc = cfg["camera"]
        self.projs = cfg["projectors"]
        self.canvas = canvas if canvas is not None else DisplayCanvas(cfg)
        self.cam = cam if cam is not None else open_camera(cc)
        self.cam_shape = self.cam.shape
        is_net = getattr(self.cam, "is_network", False)
        self.sync_mode = str(cc.get("sync") or ("auto" if is_net else "fixed")).lower()
        if self.sync_mode not in ("auto", "fixed"):
            raise ValueError(f"camera.sync debe ser 'auto' o 'fixed' (no '{self.sync_mode}')")
        # fixed: espera fija settle_ms | auto: espera mínima sync_min_ms y luego detección
        self.settle = int(cc.get("sync_min_ms", 100) if self.sync_mode == "auto"
                          else cc.get("settle_ms", 400))
        self.sync = None
        if self.sync_mode == "auto":
            from .sync import SyncedCapture
            self.sync = SyncedCapture(self.cam, cc, log=log)
            self.canvas.clear(0)          # referencia: todo negro
            self.canvas.flush(1500)
            self.sync.calibrate()

    @property
    def n_proj(self):
        return len(self.projs)

    def proj_size(self, i):
        return self.projs[i]["width"], self.projs[i]["height"]

    def _flush(self):
        if self.sync is not None:
            self.canvas.flush(1)
            self.sync.mark_shown()          # la latencia se mide desde que se muestra
            if self.settle > 1:
                self.canvas.flush(self.settle)
        else:
            self.canvas.flush(self.settle)

    def show_only(self, i, img):
        self.canvas.clear(0)
        self.canvas.set(i, img)
        self._flush()

    def show_all(self, imgs):
        self.canvas.clear(0)
        for i, im in enumerate(imgs):
            self.canvas.set(i, im)
        self._flush()

    def capture(self):
        if self.sync is not None:
            return self.sync.capture()
        return self.cam.capture_gray()

    def camera_state(self) -> dict:
        """Estado del driver (exposición/ganancia) para registrarlo con el escaneo."""
        fn = getattr(self.cam, "driver_state", None)
        return fn() if callable(fn) else {}

    def sync_summary(self):
        return self.sync.summary() if self.sync is not None else ""

    def close(self):
        self.canvas.close()
        self.cam.release()
