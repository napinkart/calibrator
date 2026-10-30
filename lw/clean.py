"""
Limpieza de escaneos viejos para empezar de cero sin tocar nada más.

Qué borra (sólo estos patrones, dentro de work/ y output/):
  work/scan_proj*.npz      datos decodificados del escaneo (lo que usa compute)
  work/scan_history.json   resumen de los últimos escaneos (para avisar de repetidos)
  work/seam_geometry.npz   huellas cacheadas para la vista previa de la costura
  work/decoded_proj*.png   vista previa de lo decodificado
  work/preflight_proj*.png negro/blanco con que se comprueba cada proyector
  work/captures_proj*.png  hoja de contacto de las capturas
  work/raw_proj*/          todas las capturas crudas (la mayor parte del espacio)
  work/all_white.png       fondo con todos los proyectores en blanco
  work/all_white_proj*.png fondo de cada proyector (escaneo por proyector)
  work/overlap_proj*_vecino*.png  capturas para medir el solape
  work/diagnostico.png     imagen del diagnóstico
Opcional:
  work/screen.json         los bordes marcados (edges=True)
  work/screen_proj*.json   los bordes de cada proyector (edges=True)
  output/                  mapas ya calculados (maps=True)

Nunca borra config.json, la config de ejemplo ni archivos que no encajen en
esos patrones.
"""
from __future__ import annotations

import shutil
from pathlib import Path

SCAN_PATTERNS = ("scan_proj*.npz", "scan_history.json", "seam_geometry.npz", "decoded_proj*.png",
                 "preflight_proj*.png", "captures_proj*.png", "diagnostico.png",
                 "all_white.png", "all_white_proj*.png", "overlap_proj*_vecino*.png")
MAP_PATTERNS = ("proj*_map.exr", "proj*_black.exr", "proj*_preview.png",
                "calibration.json", "camera_overview.png", "content_overview.png")
EDGE_PATTERNS = ("screen.json", "screen_proj*.json")


def scan_files(work, out=None) -> list[Path]:
    """Archivos y carpetas del escaneo que hay ahora en `work/`."""
    work = Path(work)
    if not work.exists():
        return []
    fs: list[Path] = []
    for pat in SCAN_PATTERNS:
        fs += sorted(work.glob(pat))
    fs += sorted(p for p in work.glob("raw_proj*") if p.is_dir())
    return fs


def map_files(out) -> list[Path]:
    """Archivos que escribe 'compute' en `output/`."""
    out = Path(out)
    if not out.exists():
        return []
    fs: list[Path] = []
    for pat in MAP_PATTERNS:
        fs += sorted(out.glob(pat))
    return fs


def size_mb(paths) -> float:
    tot = 0
    for p in paths:
        if p.is_dir():
            tot += sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
        elif p.is_file():
            tot += p.stat().st_size
    return tot / 1e6


def count_files(paths) -> int:
    n = 0
    for p in paths:
        if p.is_dir():
            n += sum(1 for f in p.rglob("*") if f.is_file())
        elif p.is_file():
            n += 1
    return n


def clean(work, out, edges: bool = False, maps: bool = False) -> tuple[int, float]:
    """Borra los escaneos (y opcionalmente bordes/mapas). Devuelve (n_archivos, MB)."""
    work, out = Path(work), Path(out)
    targets = scan_files(work, out) + (map_files(out) if maps else [])
    if edges:
        for pat in EDGE_PATTERNS:
            targets += sorted(work.glob(pat))
    targets = [p for p in targets if p.exists()]
    n, mb = count_files(targets), size_mb(targets)
    for p in targets:
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        else:
            try:
                p.unlink()
            except OSError:
                pass
    return n, mb
