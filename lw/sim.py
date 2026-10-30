"""
Rig virtual para probar el pipeline sin hardware.

Pantalla cilíndrica (cara interna), una cámara pinhole y N proyectores
pinhole. Renderiza lo que "vería" la webcam (desenfoque, ruido, gamma,
cuantización a 8 bits) y ofrece ground truth para medir el error.

ESCENAS (ver SCENES al final del módulo):
  convex3  - ciclorama convexo 120°, 3 proyectores 1920x1080, webcam 1080p.
              La cámara está FUERA del cilindro: la pantalla se aleja en los extremos.
  concave3 - el MISMO montaje con la pantalla CÓNCAVA: la cámara y los
              proyectores están DENTRO del cilindro, así que la pared se le
              echa encima y hace falta un objetivo más angular.
  concave2 - pantalla cóncava pequeña, 2 proyectores (prueba rápida).
"""
from __future__ import annotations

import numpy as np
import cv2


# --------------------------------------------------------------------------
# Catálogo de escenas simuladas (seleccionables con --scene)
# --------------------------------------------------------------------------
SCENES = {
    "convex3": {
        "titulo": "Ciclorama convexo 120° · 3 proyectores 1920x1080 · webcam 1080p",
        "curvatura": "convexa",
        "proyectores": 3,
        "nota": "La cámara está FUERA del cilindro: la pantalla se aleja en los extremos.",
    },
    "concave3": {
        "titulo": "Pantalla cóncava 72° · 3 proyectores 1920x1080 · webcam 1080p",
        "curvatura": "cóncava",
        "proyectores": 3,
        "nota": "La cámara y los proyectores están DENTRO del cilindro: hace falta "
                "un objetivo más angular (62°) y 3 proyectores cubren menos arco.",
    },
    "concave2": {
        "titulo": "Pantalla cóncava 90° · 2 proyectores 800x600 · webcam 720p",
        "curvatura": "cóncava",
        "proyectores": 2,
        "nota": "Escena mínima para pruebas rápidas.",
    },
}


def list_scenes():
    """[(nombre, título)] de las escenas simuladas disponibles."""
    return [(k, v["titulo"]) for k, v in SCENES.items()]


def look_rot(yaw_deg, pitch_deg=0.0):
    y, p = np.radians(yaw_deg), np.radians(pitch_deg)
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    return Ry @ Rx  # columnas: ejes del dispositivo en mundo


class PinholeDev:
    def __init__(self, pos, yaw, pitch, w, h, hfov_deg):
        self.C = np.asarray(pos, float)
        self.R = look_rot(yaw, pitch)
        self.w, self.h = w, h
        self.f = (w / 2) / np.tan(np.radians(hfov_deg) / 2)

    def rays(self, x, y):
        d = np.stack([(x - self.w / 2 + 0.5) / self.f,
                      -(y - self.h / 2 + 0.5) / self.f,
                      np.ones_like(x, dtype=float)], -1)
        d = d @ self.R.T
        return d / np.linalg.norm(d, axis=-1, keepdims=True)

    def project(self, X):
        P = (X - self.C) @ self.R          # a coords locales
        z = P[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            x = self.f * P[..., 0] / z + self.w / 2 - 0.5
            y = -self.f * P[..., 1] / z + self.h / 2 - 0.5
        ok = (z > 0) & (x >= -0.5) & (x < self.w - 0.5) & (y >= -0.5) & (y < self.h - 0.5)
        return x, y, ok


class CylinderScreen:
    def __init__(self, R=3.0, theta_deg=(-45, 45), y_range=(-1.0, 1.0)):
        self.R = R
        self.t0, self.t1 = np.radians(theta_deg[0]), np.radians(theta_deg[1])
        self.y0, self.y1 = y_range

    def intersect(self, C, d):
        # |(C + t d)_xz| = R, rayo desde dentro -> raíz positiva mayor
        a = d[..., 0] ** 2 + d[..., 2] ** 2
        b = 2 * (C[0] * d[..., 0] + C[2] * d[..., 2])
        c = C[0] ** 2 + C[2] ** 2 - self.R ** 2
        disc = b * b - 4 * a * c
        t = (-b + np.sqrt(np.maximum(disc, 0))) / (2 * a)
        X = C + t[..., None] * d
        th = np.arctan2(X[..., 0], X[..., 2])
        ok = (disc > 0) & (t > 0) & (th >= self.t0) & (th <= self.t1) & \
             (X[..., 1] >= self.y0) & (X[..., 1] <= self.y1)
        return X, ok, th

    def point(self, u, v):
        th = self.t0 + u * (self.t1 - self.t0)
        y = self.y1 - v * (self.y1 - self.y0)
        return np.stack([self.R * np.sin(th), y, self.R * np.cos(th)], -1)

    def uv(self, X):
        th = np.arctan2(X[..., 0], X[..., 2])
        return (th - self.t0) / (self.t1 - self.t0), (self.y1 - X[..., 1]) / (self.y1 - self.y0)


class ConvexCylinder:
    """Cara EXTERIOR de un cilindro vertical (ciclorama convexo), mirando a la cámara.
    Eje en (0, *, zc). theta=0 es el punto más cercano a la cámara."""

    def __init__(self, R=2.5, zc=6.0, theta_deg=(-60, 60), y_range=(-0.6, 0.6)):
        self.R, self.zc = R, zc
        self.t0, self.t1 = np.radians(theta_deg[0]), np.radians(theta_deg[1])
        self.y0, self.y1 = y_range

    def intersect(self, C, d):
        ox, oz = C[0], C[2] - self.zc
        a = d[..., 0] ** 2 + d[..., 2] ** 2
        b = 2 * (ox * d[..., 0] + oz * d[..., 2])
        c = ox ** 2 + oz ** 2 - self.R ** 2
        disc = b * b - 4 * a * c
        t = (-b - np.sqrt(np.maximum(disc, 0))) / (2 * a)   # primera intersección
        X = C + t[..., None] * d
        th = np.arctan2(X[..., 0], self.zc - X[..., 2])
        ok = (disc > 0) & (t > 0) & (th >= self.t0) & (th <= self.t1) & \
             (X[..., 1] >= self.y0) & (X[..., 1] <= self.y1)
        return X, ok, th

    def normal(self, X):
        n = np.stack([X[..., 0], np.zeros_like(X[..., 0]), X[..., 2] - self.zc], -1)
        return n / np.linalg.norm(n, axis=-1, keepdims=True)

    def point(self, u, v):
        th = self.t0 + u * (self.t1 - self.t0)
        y = self.y1 - v * (self.y1 - self.y0)
        return np.stack([self.R * np.sin(th), y, self.zc - self.R * np.cos(th)], -1)

    def uv(self, X):
        th = np.arctan2(X[..., 0], self.zc - X[..., 2])
        return (th - self.t0) / (self.t1 - self.t0), (self.y1 - X[..., 1]) / (self.y1 - self.y0)


def _cyl_normal_concave(screen, X):
    n = np.stack([-X[..., 0], np.zeros_like(X[..., 0]), -X[..., 2]], -1)
    return n / np.linalg.norm(n, axis=-1, keepdims=True)


class VirtualRig:
    def __init__(self, scene="convex3", cam_w=None, cam_h=None, proj_w=None, proj_h=None,
                 noise=2.0, blur=1.0, seed=0, falloff=True, vignette=0.0,
                 cam_z=None, cam_hfov=None, cam_pos=None, cam_yaw=0.0,
                 proj_phis=None, proj_fov=28.0, wall_theta=(-60, 60), wall_y=(-0.6, 0.6)):
        """`cam_pos`/`cam_yaw`/`cam_hfov` permiten mover la cámara: así se simula el
        escaneo por proyector (una pose por proyector, como en una superficie grande)."""
        if scene not in SCENES:
            raise ValueError(f"Escena desconocida '{scene}'. "
                             f"Disponibles: {', '.join(SCENES)}")
        if scene == "concave2":
            cam_w, cam_h = cam_w or 1280, cam_h or 720
            proj_w, proj_h = proj_w or 800, proj_h or 600
            self.screen = CylinderScreen()
            self.cam = PinholeDev((0, 0.15, -0.6), 0, -2, cam_w, cam_h, 92)
            self.projs = [PinholeDev((-0.4, 0.2, -1.2), -18, -4, proj_w, proj_h, 60),
                          PinholeDev((0.4, 0.2, -1.2), 18, -4, proj_w, proj_h, 60)]
            normal = lambda X: _cyl_normal_concave(self.screen, X)
        elif scene == "convex3":
            # ciclorama convexo de 120°, 3 proyectores 16:9, webcam 1080p al frente
            cam_w, cam_h = cam_w or 1920, cam_h or 1080
            proj_w, proj_h = proj_w or 1920, proj_h or 1080
            self.screen = ConvexCylinder(theta_deg=wall_theta, y_range=wall_y)
            cpos = (tuple(cam_pos) if cam_pos is not None
                    else (0, 0.1, cam_z if cam_z is not None else -1.0))
            self.cam = PinholeDev(cpos, cam_yaw, 0, cam_w, cam_h,
                                  cam_hfov if cam_hfov is not None else 48.0)
            zc = self.screen.zc
            self.projs = []
            phis = tuple(proj_phis) if proj_phis is not None else (-46, 0, 46)
            for phi in phis:   # por defecto ~17% de solape
                ph = np.radians(phi)
                d = 7.0
                pos = (d * np.sin(ph), 0.0, zc - d * np.cos(ph))
                self.projs.append(PinholeDev(pos, -phi, 0.0, proj_w, proj_h, proj_fov))
            normal = self.screen.normal
        elif scene == "concave3":
            # El MISMO montaje que 'convex3' pero con la pantalla CÓNCAVA:
            # la cámara y los proyectores están DENTRO del cilindro.
            #   - pared: 72° de arco (5.4 m) x 1.2 m de alto, radio 4.3 m
            #     (mismo tamaño de pantalla que convex3; el radio es mayor
            #      porque 3 proyectores por dentro sólo cubren ~72° de arco)
            #   - la cámara va en el mismo sitio que en convex3 (z = -1.0) pero
            #     necesita un objetivo más angular (62° en vez de 48°): al estar
            #     dentro, la pared se le echa encima en los extremos.
            #   - los proyectores, dentro y a 0.35 m del eje, apuntan radialmente
            #     a la pared con el mismo FOV de 28° que en convex3.
            cam_w, cam_h = cam_w or 1920, cam_h or 1080
            proj_w, proj_h = proj_w or 1920, proj_h or 1080
            self.screen = CylinderScreen(R=4.3, theta_deg=(-36, 36), y_range=(-0.6, 0.6))
            cpos = (tuple(cam_pos) if cam_pos is not None
                    else (0, 0.1, cam_z if cam_z is not None else -1.0))
            self.cam = PinholeDev(cpos, cam_yaw, 0, cam_w, cam_h,
                                  cam_hfov if cam_hfov is not None else 62.0)
            self.projs = []
            for phi, fov in ((-24, 28), (0, 28), (24, 28)):    # ~18% de solape
                ph = np.radians(phi)
                d_p = 0.35
                pos = (d_p * np.sin(ph), 0.0, d_p * np.cos(ph))
                self.projs.append(PinholeDev(pos, phi, 0.0, proj_w, proj_h, fov))
            normal = lambda X: _cyl_normal_concave(self.screen, X)
        else:
            raise ValueError(scene)
        self.cam_shape = (cam_h, cam_w)
        self.noise, self.blur, self.vignette = noise, blur, vignette
        self.rng = np.random.default_rng(seed)
        yy, xx = np.mgrid[0:cam_h, 0:cam_w].astype(float)
        X, ok, _ = self.screen.intersect(self.cam.C, self.cam.rays(xx, yy))
        self.X, self.on_screen = X, ok
        n = normal(X)
        self._normal = normal
        self.pmaps, self.falloffs = [], []
        for p in self.projs:
            px, py, pok = p.project(X)
            facing = np.sum(n * (p.C - X), -1) > 0      # oclusión: cara que mira al proyector
            pok = pok & ok & facing
            self.pmaps.append((px.astype(np.float32), py.astype(np.float32), pok))
            if falloff:   # Lambert: cos(incidencia) / distancia^2 (normalizado)
                v = p.C - X
                dist = np.linalg.norm(v, axis=-1)
                cos = np.clip(np.sum(n * v, -1) / dist, 0, 1)
                f = cos / dist ** 2
                f = f / np.percentile(f[pok], 95) if pok.any() else f
                self.falloffs.append(np.clip(f, 0, 1.5).astype(np.float32))
            else:
                self.falloffs.append(np.ones((cam_h, cam_w), np.float32))
        r2 = ((xx - cam_w / 2) ** 2 + (yy - cam_h / 2) ** 2) / ((cam_w / 2) ** 2 + (cam_h / 2) ** 2)
        self.vig = (1 - vignette * r2).astype(np.float32)
        self._imgs = [None] * len(self.projs)

    # ---- interfaz Rig ----
    @property
    def n_proj(self):
        return len(self.projs)

    def proj_size(self, i):
        return self.projs[i].w, self.projs[i].h

    def show_only(self, i, img):
        self._imgs = [None] * self.n_proj
        self._imgs[i] = img

    def show_all(self, imgs):
        self._imgs = list(imgs)

    def light(self):
        """Luz lineal que llega a cada píxel de cámara (sin ruido ni gamma)."""
        h, w = self.cam_shape
        light = np.full((h, w), 0.02, np.float32)  # luz ambiente
        for (px, py, ok), img, f in zip(self.pmaps, self._imgs, self.falloffs):
            black = 0.015  # nivel de negro del proyector
            if img is None:
                light += np.where(ok, 0.8 * black * f, 0)
                continue
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            v = cv2.remap(img.astype(np.float32) / 255.0, px, py, cv2.INTER_LINEAR)
            lin = black + (1 - black) * np.power(v, 2.2)
            light += np.where(ok, 0.8 * lin * f, 0)
        return light

    def capture(self):
        light = self.light() * self.vig
        light = cv2.GaussianBlur(light, (0, 0), self.blur) if self.blur else light
        cam = 255 * np.power(np.clip(light, 0, 1), 1 / 2.2)
        cam += self.rng.normal(0, self.noise, cam.shape)
        return np.clip(np.round(cam), 0, 255).astype(np.float32)

    def close(self):
        pass

    # ---- ground truth ----
    def cam_to_proj(self, i, cx, cy):
        d = self.cam.rays(np.asarray(cx, float), np.asarray(cy, float))
        X, ok, _ = self.screen.intersect(self.cam.C, d)
        px, py, pok = self.projs[i].project(X)
        facing = np.sum(self._normal(X) * (self.projs[i].C - X), -1) > 0
        return px, py, ok & pok & facing

    def cam_to_uv(self, cx, cy):
        d = self.cam.rays(np.asarray(cx, float), np.asarray(cy, float))
        X, ok, _ = self.screen.intersect(self.cam.C, d)
        u, v = self.screen.uv(X)
        return u, v, ok

    def screen_edges(self, n=9):
        """Bordes de la pantalla en cámara con marcas físicamente equiespaciadas."""
        t = np.linspace(0, 1, n)
        def cam_pts(u, v):
            X = self.screen.point(u, v)
            x, y, _ = self.cam.project(X)
            return np.stack([x, y], -1).tolist()
        return {"top": cam_pts(t, 0 * t), "right": cam_pts(1 + 0 * t, t),
                "bottom": cam_pts(t, 1 + 0 * t), "left": cam_pts(0 * t, t)}
