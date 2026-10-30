"""
Diagnóstico de un escaneo: explica POR QUÉ un proyector no se decodificó.

Lee work/scan_projN.npz (imágenes negro/blanco capturadas y píxeles válidos) y
work/all_white.png, imprime un veredicto por proyector y guarda work/diagnostico.png
(negro | blanco | diferencia | píxeles decodificados) para mirarlo o enviarlo.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import cv2


def _tile(img, w=480, label=""):
    img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    s = w / img.shape[1]
    t = cv2.resize(img, (w, max(1, int(img.shape[0] * s))), interpolation=cv2.INTER_AREA)
    cv2.rectangle(t, (0, 0), (w, 22), (0, 0, 0), -1)
    cv2.putText(t, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return t


def analyze(white, black, valid, lit, white_threshold=25.0):
    diff = white - black
    st = dict(
        black_mean=float(black.mean()), white_mean=float(white.mean()),
        white_p99=float(np.percentile(white, 99)),
        sat=float(np.mean(white >= 250) * 100),
        diff_p999=float(np.percentile(diff, 99.9)),
        neg=float(np.mean(diff < -white_threshold) * 100),
        identical=float(np.mean(np.abs(diff) < 0.5) * 100),
        lit=float(lit.mean() * 100), valid=float(valid.mean() * 100),
    )
    st["ratio"] = st["valid"] / st["lit"] * 100 if st["lit"] > 0 else 0.0
    v = []   # (gravedad, mensaje)
    if st["identical"] > 99.0:
        v.append((3, "La imagen en NEGRO y en BLANCO es EXACTAMENTE la misma: la cámara está "
                     "CONGELADA. La fuente NDI no se actualiza, o no es una cámara (¿estás "
                     "enviando una imagen fija o el contenido de TouchDesigner?)."))
    elif st["neg"] > 1.0:
        v.append((3, f"En el {st['neg']:.1f}% de la imagen el 'negro' es MÁS claro que el 'blanco': las "
                     "capturas están DESFASADAS: la cámara llega con retraso y se capturó el patrón ANTERIOR.\n"
                     "      Usa \"sync\": \"auto\" en \"camera\" (o sube \"settle_ms\" a 1500 si usas \"fixed\"), y\n"
                     "      sube \"sync_timeout_ms\" a 6000. Pasar el NDI por TouchDesigner añade retraso."))
    elif st["diff_p999"] < white_threshold:
        v.append((3, "La cámara NO ve cambiar nada cuando el proyector pasa de negro a blanco.\n"
                     "      Causas típicas, en orden:\n"
                     "       a) La fuente NDI no es la cámara que mira el ciclorama (p. ej. el NDI Out de\n"
                     "          TouchDesigner envía el contenido, no la cámara).\n"
                     "       b) Los patrones salen en OTRO monitor: revisa x/y de \"displays\" con la opción 1\n"
                     "          y compruébalo con la opción 3 (identificar).\n"
                     "       c) Una ventana de TouchDesigner (Perform/Window COMP) tapa la salida del\n"
                     "          QuadHead2Go durante el escaneo: ciérrala.\n"
                     "       d) La cámara no apunta a la zona de ese proyector."))
    if st["diff_p999"] >= white_threshold:
        if st["lit"] < 0.5:
            v.append((2, f"El proyector sólo aparece en el {st['lit']:.2f}% del encuadre: está casi "
                         "fuera de cuadro o muy oscuro. Encuadra todo el ciclorama / sube la exposición."))
        if st["sat"] > 3.0:
            v.append((2, f"SOBREEXPUESTA: {st['sat']:.1f}% de píxeles saturados en blanco. Baja la "
                         "exposición/EV en la app hasta que no haya rojo en 'Ajustar cámara'."))
        if st["white_p99"] < 60:
            v.append((2, f"SUBEXPUESTA: el blanco más brillante apenas llega a {st['white_p99']:.0f}/255. "
                         "Sube la exposición/EV o baja \"scan.white_threshold\" a 12."))
        if st["lit"] >= 0.5 and st["ratio"] < 30:
            v.append((2, f"El proyector se ve ({st['lit']:.1f}% del encuadre) pero sólo se decodifica el "
                         f"{st['ratio']:.0f}% de esa zona: las rayas no se distinguen (desenfoque, "
                         "compresión NDI o cámara movida).\n"
                         "      Prueba: \"cell_size\": 24, \"phase_period\": 48, \"min_contrast\": 6; enfoca y "
                         "bloquea el foco; máxima calidad en la app; no muevas la cámara."))
    if not v:
        v.append((0, "OK"))
    return st, v


def scan_ok(work: Path, min_valid: float = 1.0) -> list[dict]:
    """Resumen de lo que hay en work/scan_proj*.npz, para avisar ANTES de calcular.

    `valid` es la fracción del encuadre que se decodificó: si es ~0 el escaneo no
    sirve y 'compute' fallará. Devuelve [{proj, lit, valid, black, white, ok}].
    """
    work = Path(work)
    out = []
    for f in sorted(work.glob("scan_proj*.npz"), key=lambda p: int(p.stem[9:])):
        try:
            d = np.load(f)
        except Exception:
            continue
        val = float(d["valid"].mean() * 100)
        out.append(dict(proj=int(f.stem[9:]),
                        lit=float(d["lit"].mean() * 100),
                        valid=val,
                        black=float(d["black"].mean()),
                        white=float(d["white"].mean()),
                        exposure=(float(d["exposure"]) if "exposure" in d.files else None),
                        when=(float(d["scan_unix_time"]) if "scan_unix_time" in d.files else None),
                        ok=val >= min_valid))
    return out


def run(work: Path, cfg: dict, log=print) -> tuple[int, list]:
    work = Path(work)
    scans = sorted(work.glob("scan_proj*.npz"), key=lambda p: int(p.stem[9:]))
    wt = float(cfg.get("scan", {}).get("white_threshold", 25))
    rows, worst = [], 0
    all_verdicts = []  # Collect all verdicts across projectors
    aw = cv2.imread(str(work / "all_white.png"), cv2.IMREAD_GRAYSCALE)
    if not scans:
        log(f"No hay escaneos en {work}: ejecuta primero 'Escanear' (opción 5).")
        return 3, []
    if aw is not None:
        log(f"all_white.png: media {aw.mean():.0f}, máx {aw.max()}, "
            f"saturado {np.mean(aw >= 250)*100:.1f}%")
    for f in scans:
        i = int(f.stem[9:])
        d = np.load(f)
        white, black = d["white"].astype(np.float32), d["black"].astype(np.float32)
        valid, lit = d["valid"].astype(bool), d["lit"].astype(bool)
        st, verdicts = analyze(white, black, valid, lit, wt)
        exp = (f"  exposición {float(d['exposure']):.0f}"
               if "exposure" in d.files and np.isfinite(float(d["exposure"])) else "")
        when = ("  escaneado " + dt.datetime.fromtimestamp(float(d["scan_unix_time"]))
                .strftime("%H:%M:%S") if "scan_unix_time" in d.files else "")
        log(f"\nProyector {i}:  negro {st['black_mean']:.0f}  blanco {st['white_mean']:.0f} "
            f"(p99 {st['white_p99']:.0f})  saturado {st['sat']:.1f}%  "
            f"iluminado {st['lit']:.1f}%  decodificado {st['valid']:.1f}% del encuadre"
            f"{exp}{when}")
        for sev, msg in verdicts:
            log(("   OK" if sev == 0 else "   PROBLEMA: " if sev == 3 else "   AVISO: ") +
                ("" if sev == 0 else msg))
            worst = max(worst, sev)
            all_verdicts.append((i, sev, msg))
        diff = np.clip((white - black) * 2, 0, 255)
        vis = cv2.cvtColor(np.clip(white, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR) // 2
        vis[valid] = (0, 200, 0)
        rows.append(np.hstack([_tile(black, label=f"P{i} negro"), _tile(white, label=f"P{i} blanco"),
                               _tile(diff, label="blanco - negro (x2)"),
                               _tile(vis, label="verde = decodificado")]))
    img = np.vstack(rows)
    if aw is not None:
        top = _tile(aw, w=img.shape[1] // 2, label="todos en blanco")
        top = np.hstack([top, np.zeros((top.shape[0], img.shape[1] - top.shape[1], 3), np.uint8)])
        img = np.vstack([top, img])
    out = work / "diagnostico.png"
    cv2.imwrite(str(out), img)
    log(f"\nImagen de diagnóstico: {out}")
    return worst, all_verdicts
