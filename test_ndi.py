"""
Prueba end-to-end por NDI REAL (sin hardware):

  "teléfono virtual" --NDI--> NDICamera --> sincronía auto --> scan --> compute --> error

El teléfono virtual renderiza lo que vería la cámara (ciclorama convexo, 3 proyectores
del simulador) y lo emite por NDI a 30 fps con:
  - latencia 250 ms + jitter aleatorio 0-150 ms, y picos ocasionales de +700 ms (Wi-Fi),
  - compresión JPEG (imita el códec H.264 de NDI|HX),
  - un frame "a medio cambiar" en cada cambio de patrón (rolling shutter).

  python test_ndi.py            # escaneo completo con sincronía auto + evaluación
  python test_ndi.py --fixed    # además mide cuántas capturas salen MAL con espera fija

Si el descubrimiento mDNS no funciona en tu red, usa NDI_IPS=127.0.0.1 (por defecto aquí).
"""
import os, sys, json, time, threading, collections
from pathlib import Path
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent))
from lw.sim import VirtualRig
from lw import pipeline, screen, ndi
from lw.config import normalize
from lw.devices import RealRig
import test_simulated

SOURCE = "LW_VIRTUAL_PHONE"


class VirtualPhone(threading.Thread):
    def __init__(self, rig, fps=30, base_ms=250, jitter_ms=150, spike_ms=700, spike_p=0.02,
                 jpeg=80, seed=1):
        super().__init__(daemon=True)
        from fractions import Fraction
        from cyndilib.sender import Sender
        from cyndilib.video_frame import VideoSendFrame
        from cyndilib.wrapper.ndi_structs import FourCC
        self.rig, self.fps = rig, fps
        self.base, self.jit, self.spike, self.spike_p = base_ms / 1e3, jitter_ms / 1e3, spike_ms / 1e3, spike_p
        self.jpeg = jpeg
        self.rng = np.random.default_rng(seed)
        h, w = rig.cam_shape
        self.sender = Sender(SOURCE, clock_video=False)
        vf = VideoSendFrame()
        vf.set_resolution(w, h)
        vf.set_fourcc(FourCC.BGRX)
        vf.set_frame_rate(Fraction(fps, 1))
        self.sender.set_video_frame(vf)
        self.state = (0, [None] * rig.n_proj)
        self.lock = threading.Lock()
        self.rlock = threading.Lock()
        self.stop_flag = False
        self.noise = [rig.rng.normal(0, rig.noise, (h, w)).astype(np.float32) for _ in range(4)]
        self.bgrx = np.full((h, w, 4), 255, np.uint8)

    def publish(self, imgs):
        with self.lock:
            self.state = (self.state[0] + 1, list(imgs))

    def render(self, imgs):
        with self.rlock:
            r = self.rig
            r._imgs = imgs
            light = r.light() * r.vig
            light = cv2.GaussianBlur(light, (0, 0), r.blur) if r.blur else light
        return (255 * np.power(np.clip(light, 0, 1), 1 / 2.2)).astype(np.float32)

    def run(self):
        q = collections.deque()
        ver, clean, prev_clean = -1, None, None
        last_due, k = 0.0, 0
        with self.sender:
            t_next = time.time()
            while not self.stop_flag:
                now = time.time()
                with self.lock:
                    v, imgs = self.state
                transition = False
                if v != ver:
                    prev_clean, clean, ver = clean, self.render(imgs), v
                    transition = prev_clean is not None
                img = clean + self.noise[k % 4]
                if transition:      # rolling shutter: mitad superior nueva, inferior vieja
                    h = img.shape[0]
                    cut = int(h * self.rng.uniform(0.3, 0.7))
                    img[cut:] = prev_clean[cut:] + self.noise[k % 4][cut:]
                g = np.clip(np.round(img), 0, 255).astype(np.uint8)
                if self.jpeg:
                    g = cv2.imdecode(cv2.imencode(".jpg", g, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg])[1], 0)
                lat = self.base + self.rng.uniform(0, self.jit)
                if self.rng.random() < self.spike_p:
                    lat += self.spike
                due = max(last_due, now + lat)       # NDI no reordena frames
                last_due = due
                q.append((due, g))
                while q and q[0][0] <= time.time():
                    _, g2 = q.popleft()
                    self.bgrx[..., 0] = g2
                    self.bgrx[..., 1] = g2
                    self.bgrx[..., 2] = g2
                    self.sender.write_video(self.bgrx.ravel())
                k += 1
                t_next += 1.0 / self.fps
                time.sleep(max(0.0, t_next - time.time()))


class PhoneCanvas:
    """Sustituye a DisplayCanvas: lo que se 'muestra' lo ve el teléfono virtual."""

    def __init__(self, n, phone):
        self.imgs, self.phone = [None] * n, phone

    def clear(self, value=0):
        self.imgs = [None] * len(self.imgs)

    def set(self, i, img):
        self.imgs[i] = img

    def flush(self, wait_ms=1):
        self.phone.publish(self.imgs)
        time.sleep(max(1, wait_ms) / 1000)
        return -1

    def close(self):
        pass


def stale_check(phone, rig_sim, cap, imgs_now, imgs_prev):
    """¿La captura se parece más al patrón actual que al anterior?"""
    a = phone.render(imgs_now)
    b = phone.render(imgs_prev)
    return np.mean(np.abs(cap - a)) < np.mean(np.abs(cap - b))


def main():
    os.environ.setdefault("NDI_IPS", "127.0.0.1")
    here = Path(__file__).parent
    cfg = normalize(json.loads((here / "config.example.json").read_text(encoding="utf-8")))
    cfg["camera"].update(type="ndi", ndi_source=SOURCE, ndi_ips=os.environ["NDI_IPS"],
                         sync="auto", sync_min_ms=100, avg_frames=3, sync_timeout_ms=4000)
    work, out = here / "sim_work_ndi", here / "sim_output_ndi"

    sim = VirtualRig("convex3")
    ndi._lib(cfg["camera"]["ndi_ips"])          # prepara la config NDI antes del emisor
    phone = VirtualPhone(sim)
    phone.start()
    cam = ndi.NDICamera(cfg["camera"])
    canvas = PhoneCanvas(sim.n_proj, phone)

    if "--fixed" in sys.argv:
        print("\n== Comparación: espera fija settle_ms=450 (como una webcam USB) ==")
        cfg_f = dict(cfg["camera"], sync="fixed", settle_ms=450, flush_frames=5)
        rig_f = RealRig(dict(cfg, camera=cfg_f), canvas=canvas, cam=cam)
        from lw import patterns
        seq = patterns.make_sequence(1920, 1080, 16)
        bad, prev = 0, [None] * 3
        for name, img in seq:
            rig_f.show_only(1, img)
            cap = rig_f.capture()
            now = [None, img, None]
            bad += not stale_check(phone, sim, cap, now, prev)
            prev = now
        print(f"   capturas del patrón EQUIVOCADO con espera fija: {bad}/{len(seq)}")

    print("\n== Escaneo por NDI con sincronía auto ==")
    t0 = time.time()
    rig = RealRig(cfg, canvas=canvas, cam=cam)
    rig.n_proj  # noqa
    pipeline.scan(rig, cfg, work, log=lambda *a, **k: None if k.get("end") == "\r" else print(*a))
    print(f"Escaneo: {time.time()-t0:.0f}s  ({cam.stats()[:120]})")
    phone.stop_flag = True
    phone.join(2)
    cam.release()

    edges = sim.screen_edges(n=9)
    screen.save_edges(work / "screen.json", edges, "uniform")
    print("== Cálculo ==")
    results, _ = pipeline.compute(cfg, work, out)
    return test_simulated.evaluate(sim, results, out)


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
