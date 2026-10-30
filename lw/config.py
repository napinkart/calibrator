"""
Normalización de la configuración de salidas.

Soporta dos formas de describir los proyectores:

A) Splitter tipo Matrox QuadHead2Go (Windows ve UN solo monitor grande):
   "displays":   [{"name": "QuadHead2Go", "x": 1920, "y": 0,
                   "width": 5760, "height": 1080, "layout": "3x1"}],
   "projectors": [{"name": "izquierda", "display": 0, "output": 1}, ...]
   - layout "3x1": output 1..3 de izquierda a derecha (5760x1080)
   - layout "2x2": output 1=arriba-izq, 2=arriba-der, 3=abajo-izq, 4=abajo-der (3840x2160)
   - o bien "rect": [x, y, w, h] explícito dentro del display (layouts personalizados)

B) Un monitor de Windows por proyector (formato antiguo):
   "projectors": [{"name": "p1", "x": 1920, "y": 0, "width": 1920, "height": 1080}, ...]

Después de normalize() cada proyector tiene:
   display, rect=[x,y,w,h] (dentro del display), width, height, desktop_x, desktop_y
"""
from __future__ import annotations

import re


def layout_rect(display: dict, output: int) -> list[int]:
    lay = display.get("layout", "3x1").lower()
    m = re.fullmatch(r"(\d+)x(\d+)", lay)
    if not m:
        raise ValueError(f"layout desconocido '{lay}' (usa '3x1', '2x2', '4x1'... o 'rect')")
    cols, rows = int(m.group(1)), int(m.group(2))
    w, h = display["width"] // cols, display["height"] // rows
    k = int(output) - 1
    if not 0 <= k < cols * rows:
        raise ValueError(f"output {output} fuera del layout {lay}")
    return [(k % cols) * w, (k // cols) * h, w, h]


def normalize(cfg: dict) -> dict:
    projs = cfg["projectors"]
    displays = cfg.get("displays")
    if not displays:
        displays = []
        for i, p in enumerate(projs):
            displays.append({"name": p.get("name", f"display{i}"), "x": p["x"], "y": p["y"],
                             "width": p["width"], "height": p["height"]})
            p["display"] = i
            p["rect"] = [0, 0, p["width"], p["height"]]
    for i, p in enumerate(projs):
        di = int(p.get("display", 0))
        if di < 0 or di >= len(displays):
            names = ", ".join(f"{k} = '{d.get('name', '?')}' {d.get('width')}x{d.get('height')}"
                              for k, d in enumerate(displays))
            raise ValueError(
                f"proyector {i} ('{p.get('name', '')}'): \"display\": {di} no existe.\n"
                f"  En \"displays\" sólo hay {len(displays)}: {names}.\n"
                "  \"display\" es la posición dentro de la lista \"displays\" de config.json (empieza en 0),\n"
                "  NO el número de monitor de Windows que muestra 'monitors'.\n"
                "  Con un QuadHead2Go TODOS los proyectores van en \"display\": 0 y se distinguen por\n"
                "  \"output\": 1, 2, 3 (izquierda, centro, derecha).")
        d = displays[di]
        if "rect" not in p:
            if "output" not in p:
                raise ValueError(f"proyector {i}: indica 'output' (1..4) o 'rect'")
            p["rect"] = layout_rect(d, p["output"])
        x, y, w, h = map(int, p["rect"])
        if x < 0 or y < 0 or x + w > d["width"] or y + h > d["height"]:
            raise ValueError(f"proyector {i}: rect {p['rect']} se sale del display {d['name']}")
        p.update(display=di, rect=[x, y, w, h], width=w, height=h,
                 desktop_x=d["x"] + x, desktop_y=d["y"] + y)
        p.setdefault("name", f"proj{i}")
    cfg["displays"] = displays

    # Validar y normalizar surface_type
    if "surface_type" not in cfg:
        cfg["surface_type"] = "convex"  # Valor por defecto (compatibilidad con versiones antiguas)
    else:
        valid_types = {"convex", "concave", "flat"}
        if cfg["surface_type"] not in valid_types:
            raise ValueError(
                f"surface_type debe ser 'convex', 'concave' o 'flat'. "
                f"Valor dado: '{cfg['surface_type']}'"
            )

    if "surface_radius" not in cfg:
        cfg["surface_radius"] = 1.0  # Valor por defecto en metros

    return cfg
