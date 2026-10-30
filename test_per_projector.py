"""
Prueba end-to-end del ESCANEO POR PROYECTOR (sin hardware).

Simula el caso real de una superficie grande: la cámara no cabe delante de toda
la pared, así que se coloca una vez por proyector (viendo ese proyector completo
y un trozo de los vecinos), se escanea, se para, y se mueve a la siguiente pose.
Después se marcan los bordes de cada proyector y se calcula UN solo lienzo.

Comprueba contra el ground truth del simulador:
  - error geométrico de cada mapa (proyector -> cámara de SU pose)
  - que las coordenadas de contenido de todos los proyectores sean la MISMA
    recta respecto a la posición real en la pared (ni saltos ni estirones)
  - que en la franja de solape los dos proyectores asignen el mismo contenido
    (la costura queda clavada sin haber visto los dos a la vez)
  - que los pesos de blend sumen 1 sobre la pared

Uso:
  python test_per_projector.py
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent))
from lw.sim import VirtualRig
from lw import pipeline, screen, correspond
from lw.config import normalize

PHIS = (-36.0, 0.0, 36.0)          # proyectores dentro de la pared (huellas completas)
DIST = 7.0                          # distancia proyector-pared (la de la escena)
ZC = 6.0                            # centro del cilindro en convex3
CAM_FOV = 40.0                      # el proyector abre 28°: se ve entero + un trozo de vecinos


def pose_rig(i, **kw):
    """Rig con la cámara colocada delante del proyector i (cámara = su misma pose)."""
    ph = np.radians(PHIS[i])
    pos = (DIST * np.sin(ph), 0.10, ZC - DIST * np.cos(ph))
    return VirtualRig("convex3", cam_pos=pos, cam_yaw=-PHIS[i], cam_hfov=CAM_FOV,
                      proj_phis=PHIS, wall_theta=(-80, 80), wall_y=(-1.4, 1.4), **kw)


def footprint_edges(rig, i, dec, npts=40):
    """Bordes del área iluminada, marcados EQUIESPACIADOS FÍSICAMENTE.

    Es lo que el wizard llama modo 'uniform' (marcas de cinta cada X cm): los
    puntos se reparten por longitud real sobre la pared, así el contenido no se
    estira y las dos poses comparten la misma parametrización física. Los bordes
    salen del área que realmente se ilumina en la captura (lo que marcaría el
    usuario a mano).
    """
    from lw import blend
    m = blend.clean_mask(dec["lit"], k=7)
    if not m.any():
        raise RuntimeError("el proyector no ilumina nada visible")
    ys, xs = np.nonzero(m)
    H, W = m.shape
    cols = np.arange(int(xs.min()), int(xs.max()) + 1)
    rows = np.arange(int(ys.min()), int(ys.max()) + 1)
    top_y = np.array([np.argmax(m[:, x]) for x in cols], float)
    bot_y = np.array([H - 1 - np.argmax(m[::-1, x]) for x in cols], float)
    left_x = np.array([np.argmax(m[y, :]) for y in rows], float)
    right_x = np.array([W - 1 - np.argmax(m[y, ::-1]) for y in rows], float)

    def uniform_pts(px, py, by_u=True):
        px, py = np.asarray(px, float), np.asarray(py, float)
        u, v, ok = rig.cam_to_uv(px, py)
        t = u if by_u else v
        keep = ok & np.isfinite(t)
        if keep.sum() < 2:
            raise RuntimeError(f"borde sin puntos válidos (proyector {i})")
        px, py, t = px[keep], py[keep], t[keep]
        o = np.argsort(t)
        px, py, t = px[o], py[o], t[o]
        grid = np.linspace(t[0], t[-1], npts)
        return np.stack([np.interp(grid, t, px), np.interp(grid, t, py)], -1).tolist()

    return {"top": uniform_pts(cols, top_y, True),
            "right": uniform_pts(right_x, rows, False),
            "bottom": uniform_pts(cols, bot_y, True),
            "left": uniform_pts(left_x, rows, False)}


def main():
    here = Path(__file__).parent
    cfg = normalize(json.loads((here / "config.example.json").read_text(encoding="utf-8")))
    n = len(cfg["projectors"])
    work = here / "sim_work" / "per_projector"
    out = here / "sim_output" / "per_projector"
    for d in (work, out):
        d.mkdir(parents=True, exist_ok=True)
    rigs = [pose_rig(i, cam_w=960, cam_h=540, proj_w=960, proj_h=540) for i in range(n)]
    for i, r in enumerate(rigs):
        vis = float(np.mean(r.pmaps[i][2]))
        nb = [float(np.mean(r.pmaps[j][2])) for j in range(n) if j != i]
        print(f"Pose {i}: proyector {i} visible al {vis*100:.0f}% del encuadre; "
              f"vecinos {[round(x*100) for x in nb]}%")

    t0 = time.time()
    print("\n== Escaneo por proyector (con pausa entre poses) ==")
    for i in range(n):
        print(f"-- proyector {i}: cámara colocada, escaneando --")
        pipeline.scan_projector(cfg, i, work, rig=rigs[i])
        print("   pausa: la cámara se movería al siguiente proyector\n")
    print(f"Escaneos listos en {time.time()-t0:.1f}s")

    print("== Marcado de bordes por proyector ==")
    for i in range(n):
        d = dict(np.load(work / f"scan_proj{i}.npz"))
        w, h, cell = int(d["width"]), int(d["height"]), int(d["cell"])
        screen.save_edges(work / f"screen_proj{i}.json",
                          footprint_edges(rigs[i], i, d), "uniform")
        ov = pipeline.scan_overlaps(work, i)
        gt = _overlap_gt(rigs[i], i)
        print(f"   P{i}: solape medido " +
              ", ".join(f"{k.replace('overlap_', '')}={v*100:.1f}%"
                        for k, v in ov.items() if v == v)
              + "   · real " + ", ".join(f"{k}={v*100:.1f}%" for k, v in gt.items()))

    print("== Cálculo en el espacio de contenido ==")
    results, info = pipeline.compute(cfg, work, out)
    print(f"Tiempo total: {time.time()-t0:.1f}s\n")
    ok = evaluate(rigs, cfg, results, info, out)
    print("\nRESULTADO:", "OK" if ok else "REVISAR")
    return ok


def _sample(img, x, y):
    return cv2.remap(img.astype(np.float32), np.asarray(x, np.float32), np.asarray(y, np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def evaluate(rigs, cfg, results, info, out):
    n = len(results)
    ok = True

    # 1. error geométrico contra el ground truth de cada pose
    for i, r in enumerate(results):
        h, w = r["u"].shape
        py, px = np.mgrid[0:h, 0:w]
        m = r["valid"]
        gx, gy, gok = rigs[i].cam_to_proj(i, r["pmap"][..., 0][m], r["pmap"][..., 1][m])
        err = np.hypot(gx - px[m], gy - py[m])[gok]
        print(f"Proyector {i}: cobertura {m.mean()*100:.1f}%  ventana de lienzo "
              f"{[round(x, 3) for x in info['windows'][i]]}")
        print(f"   error geométrico  mediana {np.median(err):.3f}px  p95 "
              f"{np.percentile(err, 95):.3f}px")
        ok &= np.percentile(err, 95) < 1.5

    # 2. contenido vs posición real en la pared (misma recta para todos)
    su, cu = [], []
    for i, r in enumerate(results):
        h, w = r["u"].shape
        ys, xs = np.mgrid[0:h:18, 0:w:18]
        X, uu, vv, on = _proj_wall(rigs[i], i, xs, ys)
        cont = r["u"][ys, xs]
        good = on & (cont >= 0)
        su.append(uu[good])
        cu.append(cont[good])
    su, cu = np.concatenate(su), np.concatenate(cu)
    A = np.stack([np.ones_like(su), su], 1)
    coef, *_ = np.linalg.lstsq(A, cu, rcond=None)
    resid = cu - A @ coef
    print(f"\nContenido vs pared: u = {coef[0]:.4f} + {coef[1]:.4f} * u_pared")
    print(f"   residuo  mediana {np.median(np.abs(resid))*1000:.2f} (x1000)  "
          f"p95 {np.percentile(np.abs(resid), 95)*1000:.2f}  máx {np.abs(resid).max()*1000:.1f}")
    ok &= np.percentile(np.abs(resid), 95) < 0.02

    # 3. costuras: mismo punto de pared visto por dos proyectores
    print("\nCosturas (el mismo punto físico debe tener el mismo contenido):")
    t = np.linspace(0.02, 0.98, 60)
    U, V = np.meshgrid(t, t)
    X = rigs[0].screen.point(U, V)
    vals, cover = [], np.zeros(U.shape, bool)
    normal = rigs[0]._normal(X)
    for i in range(n):
        px, py, pok = rigs[i].projs[i].project(X)
        facing = np.sum(normal * (rigs[i].projs[i].C - X), axis=-1) > 0
        vals.append((_sample(results[i]["u"], px, py), pok))
        cover |= pok & facing     # zona que ese proyector ilumina de verdad
    for i in range(n - 1):
        a, oka = vals[i]
        b, okb = vals[i + 1]
        m = oka & okb & (a >= 0) & (b >= 0)
        if not m.any():
            print(f"   P{i}-P{i+1}: sin zona de solape visible")
            continue
        d = np.abs(a - b)[m]
        print(f"   P{i}-P{i+1}: solape {m.mean()*100:.1f}% del lienzo · diferencia de contenido "
              f"mediana {np.median(d)*1000:.2f} (x1000)  p95 {np.percentile(d, 95)*1000:.2f}")
        ok &= np.percentile(d, 95) < 0.02

    # 4. suma de pesos de blend sobre la zona realmente iluminada
    total = np.zeros(U.shape, np.float32)
    normal = rigs[0]._normal(X)
    for i in range(n):
        px, py, pok = rigs[i].projs[i].project(X)
        facing = np.sum(normal * (rigs[i].projs[i].C - X), axis=-1) > 0
        wi = _sample(results[i]["w"], px, py)
        total += np.where(pok & facing, wi, 0)
    inner = (U > 0.03) & (U < 0.97) & (V > 0.03) & (V < 0.97) & cover
    s = total[inner]
    print(f"\nSuma de pesos en la pared iluminada: mediana {np.median(s):.4f}  "
          f"p5 {np.percentile(s, 5):.4f}  p95 {np.percentile(s, 95):.4f}  "
          f"(cobertura del lienzo {cover.mean()*100:.0f}%)")
    ok &= abs(float(np.median(s)) - 1) < 0.03

    # 5. cómo se ve (una rejilla proyectada por los mapas, visto desde la pose del centro)
    rig = rigs[n // 2]
    rig.show_all([pipeline.render_projector(r, None, uniformity=0.0) for r in results])
    cv2.imwrite(str(out / "per_projector_result.png"), rig.capture().astype(np.uint8))
    print(f"\nImagen: {out / 'per_projector_result.png'}")
    return ok


def _overlap_gt(rig, i, stride=8):
    """Solape real (ground truth) de cada vecino, como fracción del ancho del proyector i."""
    h, w = rig.proj_size(i)[1], rig.proj_size(i)[0]
    ys, xs = np.mgrid[0:h:stride, 0:w:stride]
    X, uu, vv, on = _proj_wall(rig, i, xs, ys)
    on = on & (uu >= 0) & (uu <= 1) & (vv >= 0) & (vv <= 1)
    out = {}
    for side, j in (("left", i - 1), ("right", i + 1)):
        if not 0 <= j < len(rig.projs):
            continue
        _, _, okj = rig.projs[j].project(X)
        okj = okj & on
        cols = np.nonzero(on.any(axis=0))[0]
        if not len(cols):
            continue
        frac = okj.sum(axis=0) / np.maximum(on.sum(axis=0), 1)
        litc = on.any(axis=0) & (frac > 0.5)
        edge = cols[-1] if side == "right" else cols[0]
        span, k = 0, edge
        while 0 <= k < litc.size and litc[k]:
            span += 1
            k += -1 if side == "right" else 1
        out[side] = span / float(len(cols))
    return out


def _proj_wall(rig, i, px, py):
    """Punto de pared -> (X, u, v, ok) para píxeles del proyector i."""
    d = rig.projs[i].rays(np.asarray(px, np.float64), np.asarray(py, np.float64))
    X, ok, _ = rig.screen.intersect(rig.projs[i].C, d)
    u, v = rig.screen.uv(X)
    return X, u, v, ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
