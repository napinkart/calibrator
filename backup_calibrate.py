#!/usr/bin/env python3
"""
LibreWarp - calibrador automático de warp & blend con cámara.

Uso:
  python calibrate.py monitors                 lista monitores (para config.json)
  python calibrate.py list_devices             lista dispositivos de captura DirectShow
  python calibrate.py ndi      [--ip 192.168.1.35]  lista fuentes NDI (cámara del teléfono)
  python calibrate.py identify -c config.json  muestra el número de cada proyector (verifica el QuadHead2Go)
  python calibrate.py adjust   -c config.json  ajustar cámara (exposición/foco)
  python calibrate.py scan     -c config.json  proyecta patrones y captura
  python calibrate.py screen   -c config.json  [--mode arclength|uniform] marcar bordes
  python calibrate.py obsidian_screen -c config.json  genera nota en Obsidian para marcado
  python calibrate.py guide    -c config.json  obtener instrucciones de colocación de cámara
  python calibrate.py compute  -c config.json  calcula mapas -> output/*.exr
  python calibrate.py diagnose -c config.json  explica por qué un escaneo falló (work/diagnostico.png)
  python calibrate.py verify   -c config.json  [--image foto.jpg] previsualiza en los proyectores
  python calibrate.py clean    -c config.json  [--yes] [--edges] [--maps] borra escaneos viejos
  python calibrate.py all      -c config.json  scan + screen + compute
  python calibrate.py simulate [--scene convex3|concave3|concave2] [--figure salida.jpg]
                                               prueba completa con un rig virtual
"""
import os
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent))
from lw import pipeline, screen, patterns  # noqa: E402
from lw.config import normalize  # noqa: E402
from lw import guidance  # noqa: E402

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def load_cfg(path):
    p = Path(path)
    if not p.exists():
        sys.exit(f"No existe {p}. Copia config.example.json a config.json y edítalo.")
    try:
        cfg = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        sys.exit(f"\nconfig.json tiene un error de formato en la línea {e.lineno}, columna {e.colno}:\n"
                 f"  {e.msg}\n  (revisa comas y comillas cerca de esa línea)")
    base = p.parent
    cfg["_path"] = str(p)
    cfg["_work"] = base / cfg.get("work_dir", "work")
    cfg["_out"] = base / cfg.get("output_dir", "output")
    try:
        return normalize(cfg)
    except (ValueError, KeyError) as e:
        sys.exit(f"\nError en {p.name}:\n  {e}")


def save_cfg(cfg):
    # guarda sólo la exposición en el archivo original (no la config normalizada)
    raw = json.loads(Path(cfg["_path"]).read_text(encoding="utf-8"))
    raw.setdefault("camera", {})["exposure"] = cfg["camera"]["exposure"]
    Path(cfg["_path"]).write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")


def _log_status(cfg: dict, step_name: str, metrics: dict = None):
    """Helper para registrar estado en Obsidian de forma segura si el vault existe."""
    try:
        guidance.generate_status_note(cfg, cfg.get("_work"), cfg.get("_out"), step_name, metrics)
    except Exception:
        pass


# --------------------------------------------------------------------------
def cmd_monitors(args):
    from lw.devices import list_monitors
    list_monitors()


def cmd_list_devices(args):
    """Lista todos los dispositivos de video disponibles en la PC."""
    from lw.devices import list_video_devices
    list_video_devices()


def cmd_ndi(args):
    """Lista las fuentes NDI visibles (p. ej. la app NDI HX Camera del teléfono)."""
    from lw import ndi
    ips = args.ip
    if not ips and Path(args.config).exists():
        try:
            ips = json.loads(Path(args.config).read_text(encoding="utf-8")).get("camera", {}).get("ndi_ips")
        except Exception:
            pass
    print("Buscando fuentes NDI (8 s)...")
    try:
        names = ndi.list_sources(8.0, ips)
    except RuntimeError as e:
        sys.exit(str(e))
    if not names:
        print("\nNo se encontró ninguna fuente NDI.")
        print("  - Abre la app NDI en el teléfono y pulsa emitir; misma red Wi-Fi/LAN que el PC.")
        print("  - Permite python.exe en el Firewall de Windows (redes privadas).")
        print("  - Si la Wi-Fi aísla clientes o bloquea multicast, usa:  calibrate.py ndi --ip <IP del teléfono>")
        return
    print("\nFuentes NDI encontradas:")
    for i, n in enumerate(names):
        print(f"  [{i}] {n}")
    print("\nEn config.json, bloque \"camera\":")
    print(f'    "type": "ndi",\n    "ndi_source": "{names[0]}",')
    if ips:
        print(f'    "ndi_ips": "{ips}",')


def cmd_identify(args):
    """Muestra número y nombre en cada salida para comprobar el orden del splitter."""
    from lw.devices import DisplayCanvas, identify_image
    cfg = load_cfg(args.config)
    cv_ = DisplayCanvas(cfg)
    for i, p in enumerate(cfg["projectors"]):
        cv_.set(i, identify_image(p, i))
        print(f"proyector {i}: '{p['name']}' display={p['display']} rect={p['rect']} "
              f"escritorio=({p['desktop_x']},{p['desktop_y']})")
    print("Cada proyector debe mostrar su número completo, con el marco entero visible.")
    print("Pulsa cualquier tecla en la ventana para salir.")
    cv_.flush(0)
    cv_.close()


def cmd_adjust(args):
    """Vista en vivo de la cámara con los proyectores en blanco."""
    from lw.devices import DisplayCanvas, open_camera
    cfg = load_cfg(args.config)
    cv_ = DisplayCanvas(cfg)
    cam = open_camera(cfg["camera"])
    if getattr(cam, "is_network", False) and not cam.ptz_supported():
        print("Cámara NDI: exposición, foco y balance se ajustan EN LA APP del teléfono y se "
              "BLOQUEAN (AE/AF lock). Objetivo: blanco sin rojo (saturación) y rayas nítidas.")
    exp = cfg["camera"].get("exposure")
    exp = -6 if exp is None else int(exp)   # ojo: 0 es una exposición válida, no la sustituyas
    modes = ["blanco", "negro", "rayas finas", "por proyector"]
    mode, which = 0, 0
    cell = cfg["scan"].get("cell_size", 16)
    help_txt = ("+/- exposición | m modo | n siguiente proyector | c ajustes del driver | "
                "s guardar | q salir")
    print(help_txt)

    def refresh():
        cv_.clear(0)
        for i, p in enumerate(cfg["projectors"]):
            w, h = p["width"], p["height"]
            if mode == 0:
                img = np.full((h, w), 255, np.uint8)
            elif mode == 1:
                img = np.zeros((h, w), np.uint8)
            elif mode == 2:  # el patrón más fino que se usará: debe verse nítido
                seq = patterns.make_sequence(w, h, cell)
                img = [s for n, s in seq if n.startswith("col")][-2]
            else:
                img = np.full((h, w), 255 if i == which else 0, np.uint8)
            cv_.set(i, img)
        cv_.flush(1)
    refresh()
    while True:
        f = cam.read_bgr()
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        vis = f.copy()
        vis[g >= 250] = (0, 0, 255)  # saturado en rojo
        s = min(1.0, 1280 / vis.shape[1])
        vis = cv2.resize(vis, None, fx=s, fy=s)
        txt = f"exp={exp}  modo={modes[mode]}  max={g.max()}  saturado={np.mean(g>=250)*100:.2f}%"
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 50), (0, 0, 0), -1)
        cv2.putText(vis, txt, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        cv2.putText(vis, help_txt, (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.imshow("LibreWarp - camara", vis)
        k = cv2.waitKey(30) & 0xFF
        if k in (ord("q"), 27):
            break
        elif k in (ord("+"), ord("=")):
            exp += 1; cam.set_exposure(exp)
        elif k in (ord("-"), ord("_")):
            exp -= 1; cam.set_exposure(exp)
        elif k == ord("m"):
            mode = (mode + 1) % len(modes); refresh()
        elif k == ord("n"):
            which = (which + 1) % len(cfg["projectors"]); refresh()
        elif k == ord("c"):
            cam.open_driver_settings()
        elif k == ord("s"):
            cfg["camera"]["exposure"] = exp
            save_cfg(cfg)
            print(f"Exposición {exp} guardada en {cfg['_path']}")
            _log_status(cfg, "camera_adjust", {"exposure": exp})

    cv2.destroyAllWindows()
    cam.release()


def cmd_scan(args):
    from lw.devices import RealRig
    from lw import diagnose
    cfg = load_cfg(args.config)
    try:
        rig = RealRig(cfg)
    except RuntimeError as e:
        sys.exit(f"\nERROR al abrir la cámara: {e}")
    try:
        pipeline.scan(rig, cfg, cfg["_work"])
    except RuntimeError as e:
        rig.close()
        sys.exit(str(e))
    rig.close()

    print("\n== Diagnóstico rápido ==")
    worst, verdicts = diagnose.run(cfg["_work"], cfg)
    if worst >= 2:
        print("\nHay problemas: corrígelos y repita el escaneo antes de calcular (opción 7).")

    stats = {"worst_severity": worst}
    scans = sorted(cfg["_work"].glob("scan_proj*.npz"), key=lambda p: int(p.stem[9:]))
    stats["projectors_scanned"] = len(scans)
    for f in scans:
        try:
            d = np.load(f)
            stats[f"proj{f.stem[9:]}_valid_pct"] = f"{float(d['valid'].mean() * 100):.1f}%"
        except Exception:
            pass
    _log_status(cfg, "scan_complete", stats)


def cmd_screen(args):
    cfg = load_cfg(args.config)
    work = cfg["_work"]
    img = cv2.imread(str(work / "all_white.png"))
    if img is None:
        sys.exit("Falta work/all_white.png: ejecute 'scan' primero.")
    existing = None
    if (work / "screen.json").exists() and not args.reset:
        existing, _ = screen.load_edges(work / "screen.json")
    mode = args.mode or cfg.get("screen", {}).get("edge_param", "arclength")
    print("Marque los bordes de la pantalla.")
    if mode == "uniform":
        print("  Modo 'uniform': cada punto cuenta como equiespaciado FÍSICAMENTE.")
        print("  -> Clic sobre marcas de cinta cada X cm (incluidas las 2 esquinas).")
        print("     Si marca puntos libres, use --mode arclength o el contenido saldrá estirado.")
    else:
        print("  Modo 'arclength': los puntos se reparten por distancia EN LA IMAGEN.")
        print("  -> Marque puntos libres donde le convenga (con muchos puntos la curva clava la forma).")
        print("     Para una pantalla curva y contenido sin estirar: cinta equiespaciada + --mode uniform.")
    edges = screen.click_edges(img, existing)
    if edges is None:
        print("Cancelado.")
        return
    puntos = {k: len(v) for k, v in edges.items()}
    print(f"Puntos por borde: {puntos} (todos se usan: la curva pasa por cada uno)")
    screen.save_edges(work / "screen.json", edges, mode)
    print(f"Guardado {work / 'screen.json'}  (edge_param = {mode})")
    _log_status(cfg, "screen_marking", {"edge_param": mode, "points_per_edge": puntos})


def cmd_obsidian_screen(args):
    """Genera la nota de Obsidian con la imagen de referencia para marcado de bordes."""
    cfg = load_cfg(args.config)
    work = cfg["_work"]
    all_white_path = work / "all_white.png"
    if not all_white_path.exists():
        sys.exit("Falta work/all_white.png: ejecute 'scan' primero.")

    try:
        note_path = guidance.generate_edge_marking_note(cfg, work)
        print(f"[OK] Nota de Obsidian para marcado de bordes creada en:\n  {note_path}")
        print("\nInstrucciones:")
        print("1. Abra Obsidian para visualizar la nota con la captura de pantalla.")
        print("2. Para marcar los bordes de forma rápida e interactiva, use:")
        print("   python calibrate.py screen -c config.json")
    except Exception as e:
        print(f"[AVISO] No se pudo crear la nota de Obsidian: {e}")
        print("Use la herramienta nativa: python calibrate.py screen -c config.json")


def cmd_guide(args):
    """Obtener instrucciones de colocación y enfoque de cámara."""
    cfg = load_cfg(args.config)
    guidance_text = guidance.generate_camera_placement_guidance(cfg, cfg["_work"])
    _log_status(cfg, "camera_guidance", {"details": "Instrucciones de cámara generadas"})
    print("\n" + "=" * 60)
    print(guidance_text)
    print("=" * 60)


def cmd_compute(args):
    cfg = load_cfg(args.config)
    try:
        pipeline.compute(cfg, cfg["_work"], cfg["_out"])
    except RuntimeError as e:
        sys.exit(f"\nERROR: {e}")

    cal_file = cfg["_out"] / "calibration.json"
    metrics = {}
    if cal_file.exists():
        try:
            cal_data = json.loads(cal_file.read_text(encoding="utf-8"))
            unif = cal_data.get("uniformity", {})
            if unif:
                metrics["uniformity_min_max"] = unif.get("brightness_min_over_max", "N/A")
                metrics["light_conserved_pct"] = f"{unif.get('light_kept_avg', 0) * 100:.1f}%"
            metrics["projectors"] = len(cal_data.get("projectors", []))
        except Exception:
            pass
    _log_status(cfg, "compute_complete", metrics)


def cmd_diagnose(args):
    from lw import diagnose
    cfg = load_cfg(args.config)
    worst = diagnose.run(cfg["_work"], cfg)
    if worst == 0:
        print("\nEl escaneo parece correcto. Si 'compute' falla, revisa la marca de bordes (opción 6).")
    _log_status(cfg, "diagnose_complete", {"diagnosis_level": worst})


def load_maps(out: Path):
    meta_path = out / "calibration.json"
    if not meta_path.exists():
        sys.exit(f"Falta {meta_path}: ejecute 'compute' primero (opción 7).")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not pipeline.exr_available():
        sys.exit("\n" + pipeline.exr_error_msg())
    res = []
    for p in meta["projectors"]:
        m = cv2.imread(str(out / p["file"]), cv2.IMREAD_UNCHANGED)
        b = cv2.imread(str(out / p["black_file"]), cv2.IMREAD_UNCHANGED)
        if m is None or b is None:
            sys.exit(f"No se pudieron leer {p['file']} / {p['black_file']} en {out}.\n"
                     "  Ejecute 'compute' de nuevo para regenerar los mapas.")
        res.append(dict(u=m[..., 2], v=m[..., 1], w=m[..., 0], lift=b[..., 2], gain=b[..., 1]))
    return meta, res


def cmd_verify(args):
    from lw.devices import DisplayCanvas
    cfg = load_cfg(args.config)
    meta, res = load_maps(cfg["_out"])
    content = cv2.imread(args.image) if args.image else None
    cv_ = DisplayCanvas(cfg)
    gamma, black, unif = 2.2, 0.0, 0.0
    modes = ["rejilla", "blanco"] + (["imagen"] if content is not None else [])
    mode = 0
    print("g/G gamma | b/B negro | u/U uniformidad | m modo (rejilla/blanco/imagen) | q salir")
    while True:
        for i, r in enumerate(res):
            img = pipeline.render_projector(r, content if modes[mode] == "imagen" else None,
                                            gamma, black, unif, white=modes[mode] == "blanco")
            cv_.set(i, img)
        k = cv_.flush(0) & 0xFF
        if k in (ord("q"), 27):
            break
        gamma += {ord("g"): -0.1, ord("G"): 0.1}.get(k, 0)
        black = max(0.0, black + {ord("b"): -0.005, ord("B"): 0.005}.get(k, 0))
        unif = min(1.0, max(0.0, unif + {ord("u"): -0.1, ord("U"): 0.1}.get(k, 0)))
        if k == ord("m"):
            mode = (mode + 1) % len(modes)
        print(f"modo={modes[mode]} gamma={gamma:.2f} negro={black:.3f} uniformidad={unif:.1f}")
    cv_.close()
    _log_status(cfg, "verify_complete", {"verified": True})


def cmd_clean(args):
    """Lista (y con --yes borra) los escaneos viejos. Sin --yes no toca nada."""
    from lw import clean as cl
    cfg = load_cfg(args.config)
    work, out = cfg["_work"], cfg["_out"]
    files = cl.scan_files(work, out)
    if args.maps:
        files += cl.map_files(out)
    if args.edges:
        files.append(work / "screen.json")
    files = [p for p in files if p.exists()]
    if not files:
        print("No hay nada que borrar en", work, "(ni en", out, ")")
        return
    print("Se borrarían:")
    for p in files:
        print("  ", p)
    print(f"Total: {cl.count_files(files)} archivos, {cl.size_mb(files):.0f} MB")
    if not args.yes:
        print("\nNada borrado. Repita con --yes para borrar de verdad.")
        return
    n, mb = cl.clean(work, out, edges=args.edges, maps=args.maps)
    print(f"Borrados {n} archivos ({mb:.0f} MB).")


def cmd_all(args):
    cmd_scan(args)
    cmd_screen(args)
    cmd_compute(args)


def cmd_simulate(args):
    import test_simulated
    from lw.sim import SCENES
    scene = args.scene or "convex3"
    if scene not in SCENES:
        print(f"Escena desconocida '{scene}'. Disponibles:")
        for k, v in SCENES.items():
            print(f"  {k:10s} {v['titulo']}")
        sys.exit(1)
    test_simulated.main(scene, figure=args.figure)


def main():
    ap = argparse.ArgumentParser(description="LibreWarp - calibrador warp & blend")
    ap.add_argument("command", choices=["monitors", "list_devices", "ndi", "identify", "adjust", "scan", "screen",
                                        "obsidian_screen", "guide", "compute", "diagnose", "verify", "all", "simulate", "clean"])
    ap.add_argument("-c", "--config", default="config.json")
    ap.add_argument("--image", help="imagen de contenido para 'verify'")
    ap.add_argument("--reset", action="store_true", help="'screen': empezar de cero")
    ap.add_argument("--mode", choices=["arclength", "uniform"],
                    help="'screen': cómo repartir los puntos. arclength = clics libres (defecto). "
                         "uniform = puntos equiespaciados FÍSICAMENTE (cinta cada X cm); es lo "
                         "correcto para pantallas curvas y contenido sin estirar")
    ap.add_argument("--ip", help="'ndi': IP(s) del teléfono si el descubrimiento automático no funciona")
    ap.add_argument("--scene", help="'simulate': escena simulada (convex3, concave3, concave2)")
    ap.add_argument("--figure", help="'simulate': ruta del JPG de documentación a generar "
                                     "(arriba: vista de cámara; abajo: resultado)")
    ap.add_argument("--yes", action="store_true", help="'clean': borrar de verdad (sin esto sólo lista)")
    ap.add_argument("--edges", action="store_true", help="'clean': borrar también work/screen.json")
    ap.add_argument("--maps", action="store_true", help="'clean': borrar también los mapas de output/")
    args = ap.parse_args()
    globals()[f"cmd_{args.command}"](args)


if __name__ == "__main__":
    main()