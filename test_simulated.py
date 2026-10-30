"""
Prueba end-to-end con el rig virtual (sin hardware):
  python test_simulated.py              # ciclorama convexo, 3 proyectores 1920x1080, webcam 1080p
  python test_simulated.py concave3     # pantalla cóncava, 3 proyectores 1920x1080, webcam 1080p
  python test_simulated.py concave2     # pantalla cóncava, 2 proyectores
  python test_simulated.py concave3 --figure ../docs/sim_concavo_3proy.jpg
Mide el error del mapa geométrico, de las coordenadas de contenido, la suma de
blending y la uniformidad de brillo contra el ground truth del simulador.
"""
import os, sys, json, time, argparse
from pathlib import Path
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent))
from lw.sim import VirtualRig, SCENES
from lw import pipeline, screen
from lw.config import normalize


def main(scene="convex3", figure=None):
    here = Path(__file__).parent
    if scene not in SCENES:
        raise SystemExit(f"Escena desconocida '{scene}'. Disponibles: {', '.join(SCENES)}")
    cfg = normalize(json.loads((here / "config.example.json").read_text(encoding="utf-8")))
    if scene == "concave2":
        cfg["scan"]["cell_size"] = 8
        cfg["projectors"] = cfg["projectors"][:2]
    work, out = here / "sim_work" / scene, here / "sim_output" / scene
    rig = VirtualRig(scene)
    info = SCENES[scene]
    print(f"Escena '{scene}': {info['titulo']}")
    print(f"   {info['nota']}")
    print(f"   {rig.n_proj} proyectores {rig.proj_size(0)}, cámara {rig.cam_shape[::-1]}")

    t0 = time.time()
    print("== Escaneo simulado ==")
    pipeline.scan(rig, cfg, work, log=lambda *a, **k: None if k.get("end") == "\r" else print(*a))
    edges = rig.screen_edges(n=9)           # marcas físicamente equiespaciadas
    screen.save_edges(work / "screen.json", edges, "uniform")
    print("== Cálculo ==")
    results, cam = pipeline.compute(cfg, work, out)
    print(f"Tiempo total: {time.time()-t0:.1f}s\n")
    ok = evaluate(rig, results, out)
    if figure:
        p = build_figure(out / "camera_overview.png", out / "sim_camera_result.png",
                         figure, edges)
        print(f"Figura escrita: {p}")
    return ok


def _screen_bbox(edges, shape, margin=0.10):
    """Caja que contiene la pantalla en la imagen de cámara, con margen."""
    pts = np.vstack([np.asarray(v, np.float64) for v in edges.values()])
    h, w = shape[:2]
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    mx, my = (x1 - x0) * margin, (y1 - y0) * margin
    return (max(0, int(x0 - mx)), max(0, int(y0 - my)),
            min(w - 1, int(x1 + mx)), min(h - 1, int(y1 + my)))


def build_figure(overview_png, result_png, out_path, edges=None, width=1032, margin=0.10):
    """Figura de documentación en dos paneles:
         arriba  = vista de la cámara con las huellas de cada proyector
                   (output/camera_overview.png)
         abajo   = el resultado: el contenido ya deformado sobre la pantalla
    Se recorta a la pantalla (con un poco de margen) y se apilan.
    """
    ov = cv2.imread(str(overview_png))
    rs = cv2.imread(str(result_png))
    if ov is None or rs is None:
        raise FileNotFoundError(f"Faltan las imágenes: {overview_png} o {result_png}")
    if edges:
        x0, y0, x1, y1 = _screen_bbox(edges, ov.shape, margin)
        ov = ov[y0:y1 + 1, x0:x1 + 1]
        rs = rs[y0:y1 + 1, x0:x1 + 1]
    w = width or ov.shape[1]
    if ov.shape[1] != w:
        s = w / ov.shape[1]
        ov = cv2.resize(ov, (w, max(1, int(round(ov.shape[0] * s)))), interpolation=cv2.INTER_AREA)
        rs = cv2.resize(rs, (w, max(1, int(round(rs.shape[0] * s)))), interpolation=cv2.INTER_AREA)
    gap = 6
    canvas = np.full((ov.shape[0] + gap + rs.shape[0], w, 3), 43, np.uint8)
    canvas[:ov.shape[0]] = ov
    canvas[ov.shape[0] + gap:] = rs
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    return out_path


def evaluate(rig, results, out):
    """Compara el resultado con el ground truth del rig virtual."""
    ok_all = True
    for i, r in enumerate(results):
        h, w = r["u"].shape
        py, px = np.mgrid[0:h, 0:w]
        m = r["valid"]
        cx, cy = r["pmap"][..., 0][m], r["pmap"][..., 1][m]
        gx, gy, gok = rig.cam_to_proj(i, cx, cy)
        err = np.hypot(gx - px[m], gy - py[m])[gok]
        uu, vv, uok = rig.cam_to_uv(cx, cy)
        uerr = np.hypot(uu - r["u"][m], vv - (1 - r["v"][m]))[uok]
        print(f"Proyector {i}: cobertura válida {m.mean()*100:.1f}%")
        print(f"   error geométrico  mediana {np.median(err):.3f}px  p95 {np.percentile(err,95):.3f}px  máx {err.max():.2f}px")
        print(f"   error UV (x1000)  mediana {np.median(uerr)*1000:.2f}  p95 {np.percentile(uerr,95)*1000:.2f}")
        ok_all &= np.percentile(err, 95) < 1.5

    # suma de blending en la pantalla (vista desde la cámara)
    hc, wc = rig.cam_shape
    yy, xx = np.mgrid[0:hc, 0:wc]
    uu, vv, on = rig.cam_to_uv(xx.astype(float), yy.astype(float))
    inner = on & (uu > 0.01) & (uu < 0.99) & (vv > 0.01) & (vv < 0.99)
    total = np.zeros((hc, wc), np.float32)
    cover = np.zeros((hc, wc), bool)
    for i, r in enumerate(results):
        px, py, pok = rig.pmaps[i]
        wi = cv2.remap(r["w"], px, py, cv2.INTER_LINEAR)
        total += np.where(pok, wi, 0)
        cover |= pok
    sel = inner & cover
    s = total[sel]
    print(f"\nSuma de pesos de blend sobre la pantalla: media {s.mean():.4f}  "
          f"p1 {np.percentile(s,1):.4f}  p99 {np.percentile(s,99):.4f}")
    ok_all &= abs(np.median(s) - 1) < 0.02

    # uniformidad de brillo: blanco total con Uniformity = 0 y = 1
    rig.show_all([np.zeros((p.h, p.w), np.uint8) for p in rig.projs])
    dark = rig.light()
    print("\nUniformidad de brillo en pantalla con blanco completo (luz real, p1/p99):")
    for k in (0.0, 1.0):
        rig.show_all([pipeline.render_projector(r, None, uniformity=k, white=True) for r in results])
        L = (rig.light() - dark)[sel]
        print(f"   Uniformity={k:.0f}:  mín/máx = {np.percentile(L,1)/np.percentile(L,99):.3f}   "
              f"variación (CV) = {L.std()/L.mean()*100:.1f}%   brillo medio = {L.mean():.3f}")

    # imagen "cómo se ve" desde la cámara con la rejilla de prueba
    rig.show_all([pipeline.render_projector(r, None, uniformity=1.0) for r in results])
    cv2.imwrite(str(out / "sim_camera_result.png"), rig.capture().astype(np.uint8))
    print("\nRESULTADO:", "OK" if ok_all else "REVISAR")
    return ok_all


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Simulación de LibreWarp (sin hardware)")
    ap.add_argument("scene", nargs="?", default="convex3", choices=list(SCENES),
                    help="escena simulada a ejecutar")
    ap.add_argument("--figure", help="ruta del JPG de documentación a generar "
                                     "(arriba: vista de cámara; abajo: resultado)")
    a = ap.parse_args()
    main(a.scene, figure=a.figure)
