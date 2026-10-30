"""
LibreWarp — Wizard web (interfaz amigable, guía paso a paso).

Ejecutar (dentro de calibrator/, con el venv creado por librewarp.bat):
    .venv\\Scripts\\python.exe -m streamlit run wizard.py

Qué hace:
  - HOME tipo dashboard (se abre ahí): estado real de TODO con tus archivos work/, output/
    y config.json, con el «Siguiente paso» y detección rápida (sólo lista, no elige).
  - Checklist automático de progreso (detecta qué paso falta con tus archivos work/ y output/).
  - Detección de fuentes de video: webcams USB y fuentes NDI; Monitores detectados con
    screeninfo. Lo elegido se ESCRIBE en config.json (explicítamente, siempre reversible).
  - Config genérica: con Matrox/otro splitter (un monitor grande, layout NxM o rects
    personalizados) o SIN splitter (un monitor de Windows por proyector), 2..N proyectores.
  - Vista de cámara en vivo en el navegador + foto fija para marcar bordes.
  - Marcar los 4 bordes de la pantalla HACIENDO CLIC sobre la imagen.
  - Botones para scan / compute / diagnóstico con la salida en vivo del proceso.
  - La cámara NUNCA se elige sola: se detecta y EL USUARIO la selecciona explícitamente.
  - Guía visual de encuadre sobre la vista en vivo (margen 5%, tercios, zona iluminada).
  - Feed de la cámara durante el escaneo (capturas raw mostradas al momento).
  - Registro de la sesión en la bóveda Obsidian (ObsidianVault/LibreWarp) para que el
    agente (fcc-claude) aprenda de los fallos y setups.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import cv2
import streamlit as st

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

st.set_page_config(page_title="LibreWarp — Calibrador", page_icon="🎯", layout="wide")

WORK = HERE / "work"
OUT = HERE / "output"
SIM_PY = HERE.parent / ".venv-sim" / "Scripts" / "python.exe"
VENV_PY = HERE / ".venv" / "Scripts" / "python.exe"

def _is_python_working(path: Path) -> bool:
    if not path.exists(): return False
    try:
        # Intenta ejecutar una versión rápida para ver si el binario es válido en este PC
        subprocess.run([str(path), "--version"], capture_output=True, timeout=1, check=True)
        return True
    except Exception:
        return False

# Prioridad: 1. venv-sim (raíz), 2. .venv (local), 3. system python
if _is_python_working(SIM_PY):
    PY = str(SIM_PY)
elif _is_python_working(VENV_PY):
    PY = str(VENV_PY)
else:
    PY = sys.executable

VAULT = HERE.parent / "ObsidianVault" / "LibreWarp"
CONFIG_PATH = HERE / "config.json"

EDGES_ORDER = ["top", "right", "bottom", "left"]
EDGE_HELP = {
    "top": "Borde SUPERIOR: clics de IZQUIERDA a DERECHA",
    "right": "Borde DERECHO: de ARRIBA a ABAJO",
    "bottom": "Borde INFERIOR: de IZQUIERDA a DERECHA",
    "left": "Borde IZQUIERDO: de ARRIBA a ABAJO",
}

try:
    from streamlit_image_coordinates import streamlit_image_coordinates
    HAS_CLICKS = True
except ImportError:
    HAS_CLICKS = False


# ==========================================================================
# utilidades: config, cámaras, monitores, obsidian, subprocess
# ==========================================================================
def load_config():
    p = CONFIG_PATH
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def save_config(cfg):
    """Guarda la configuración actual en config.json."""
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")


def config_section():
    """Muestra la sección de configuración inicial donde se selecciona el tipo de superficie."""
    st.header("📝 Configuración Inicial")
    config = load_config() or {}

    # Tipo de superficie
    surface_type = st.selectbox(
        "Tipo de superficie:",
        ["convex", "concave", "flat"],
        index=["convex", "concave", "flat"].index(config.get("surface_type", "convex"))
    )

    # Radio de la superficie (solo para convex/concave)
    surface_radius = st.number_input(
        "Radio de la superficie (metros):",
        min_value=0.1,
        max_value=10.0,
        value=config.get("surface_radius", 1.0),
        help="Radio para cálculos de warp en superficies curvas. Para superficies planas, usa 1.0."
    )

    # Guardar configuración
    if st.button("Guardar Configuración", type="primary"):
        cfg = {"surface_type": surface_type, "surface_radius": surface_radius}
        # Mantener campos existentes de config.json si ya existen
        existing = load_config() or {}
        cfg.update({k: v for k, v in existing.items() if k not in ["surface_type", "surface_radius"]})
        save_config(cfg)
        st.success("✅ Configuración guardada correctamente en config.json")
        st.rerun()

    return {"surface_type": surface_type, "surface_radius": surface_radius}


def scan_section(config):
    """Sección para escaneo por proyector individual."""
    st.header("🔍 Escaneo por Proyector")
    projectors = config.get("projectors", [])

    if not projectors:
        st.warning("No hay proyectores configurados en config.json. Configura primero la sección anterior.")
        return config

    # Seleccionar proyector para escanear
    projector_names = [p.get("name", f"proyector{i}") for i, p in enumerate(projectors)]
    projector_index = st.selectbox(
        "Selecciona el proyector a escanear:",
        range(len(projectors)),
        format_func=lambda i: projector_names[i]
    )

    # Selector para cell_size (basado en sugerencia automática)
    opt_cell = calculate_optimal_cell_size(1280, 720, projectors[0].get('width', 1920), projectors[0].get('height', 1080), current_cell=16)

    col_a, col_b = st.columns(2)
    with col_a:
        cell_size = st.number_input(
            "Cell Size (px):",
            min_value=8,
            max_value=32,
            value=config.get("scan", {}).get("cell_size", opt_cell),
            help="Tamaño de celda. Sugerido para tu cámara: " + str(opt_cell)
        )
    with col_b:
        if st.button("Aplicar Cell Size"):
            config["scan"]["cell_size"] = int(cell_size)
            save_config(config)
            st.rerun()

    st.markdown(f"**Proyector seleccionado:** {projector_names[projector_index]}")
    st.info(f"📋 Pasos: 1) Asegúrate de que el proyector {projector_index + 1} esté activo. 2) Mueve la cámara para cubrir esa área. 3) Haz clic en 'Iniciar Escaneo'.")

    # Agregar importación segura
    try:
        from lw.pipeline import scan_overlaps, scan_projector
    except ImportError:
        st.error("Error al importar scan_overlaps o scan_projector de lw.pipeline")
        return config

    # ... (resto del código de scan_section)

    if st.button("📷 Iniciar Escaneo", type="primary", disabled=not HAS_CLICKS):
        try:
            # Asegurar que se use el cell_size del config actual
            scan_projector(config, projector_index)
            st.success(f"✅ Escaneo completado para proyector {projector_index + 1}.")
            st.info(f"📁 Archivos generados: work/all_white_proj{projector_index}.png y work/scan_proj{projector_index}.npz")
        except Exception as e:
            st.error(f"❌ Error durante el escaneo: {str(e)[:200]}")

    # Mostrar imagen de referencia si existe
    ref_image = f"work/all_white_proj{projector_index}.png"
    if Path(ref_image).exists():
        st.image(ref_image, caption=f"Imagen blanca para proyector {projector_index + 1}", use_column_width=True)

    return config


@st.cache_data(ttl=2)
def load_config_cached():
    return load_config()


def release_camera():
    """Cierra la cámara que haya abierta la wizard (para dejar el dispositivo libre)."""
    cur = st.session_state.pop("_camera", None)
    if cur is not None:
        try:
            cur[1].release()
        except Exception:
            pass


def get_camera(camera: dict):
    """Abre la cámara UNA vez y la reutiliza mientras no cambie la config."""
    sig = json.dumps(camera or {}, sort_keys=True)
    cur = st.session_state.get("_camera")
    if cur is not None and cur[0] == sig:
        return cur[1]
    release_camera()
    from lw.devices import open_camera
    cam = open_camera(camera)
    st.session_state["_camera"] = (sig, cam)
    return cam


def reset_cache():
    release_camera()
    load_config_cached.clear()
    st.cache_resource.clear()


def edit_config(mutate, create_from_example=True):
    """Carga config.json (creándola desde el ejemplo si falta), aplica mutate(cfg) y guarda."""
    p = CONFIG_PATH
    if not p.exists():
        ex = HERE / "config.example.json"
        if not ex.exists() or not create_from_example:
            return None
        p.write_text(ex.read_text(encoding="utf-8"), encoding="utf-8")
        _t = json.loads(p.read_text(encoding="utf-8"))
        _t.pop("camera", None)
        p.write_text(json.dumps(_t, indent=2, ensure_ascii=False), encoding="utf-8")
    cfg = json.loads(p.read_text(encoding="utf-8"))
    cfg = mutate(cfg) or cfg
    p.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    reset_cache()
    return cfg


def detect_webcams(max_index: int = 6) -> list[dict]:
    """Probe cámara ÍNDICES 0..max_index con DirectShow, sin tocar config."""
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        try:
            if cap.isOpened():
                ret, f = cap.read()
                if ret and f is not None:
                    found.append({"tipo": "webcam", "index": i, "res": f"{f.shape[1]}x{f.shape[0]}"})
        except Exception:
            pass
        finally:
            cap.release()
    return found


def detect_video_devices(max_index: int = 8) -> list[dict]:
    """Lista dispositivos con nombres reales y verifica si están activos (campo 'ok')."""
    from pygrabber.dshow_graph import FilterGraph
    graph = FilterGraph()
    devices = graph.get_input_devices()

    dev_list = []
    for i, name in enumerate(devices):
        # Verificar si la cámara abre (probando brevemente)
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        is_ok = False
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                is_ok = True
            cap.release()

        dev_list.append({
            "tipo": "webcam",
            "index": i,
            "name": name,
            "backend": "dshow",
            "width": 1280,
            "height": 720,
            "ok": is_ok
        })
    return dev_list


def detect_ndi(ips: str | None, timeout: float = 8.0) -> list[str]:
    from lw import ndi
    return ndi.list_sources(timeout, ips)


def detect_monitors() -> list[dict]:
    try:
        from screeninfo import get_monitors
    except ImportError:
        st.error(
            "Falta el paquete `screeninfo`: estás corriendo la wizard con un Python que NO "
            "es el del proyecto. Lánzala con el venv: doble clic en `start_streamlit.bat`, o "
            "`calibrator\\.venv\\Scripts\\python.exe -m streamlit run wizard.py`. "
            "(Alternativa: `pip install -r requirements.txt` en ese Python.)")
        return []
    out = []
    for i, m in enumerate(get_monitors()):
        out.append({"idx": i, "name": str(getattr(m, "name", f"monitor{i}")),
                    "x": int(m.x), "y": int(m.y), "w": int(m.width), "h": int(m.height)})
    return out


def apply_camera_webcam(index: int, backend: str = "dshow"):
    def mut(cfg):
        c = cfg.setdefault("camera", {})
        c["type"] = "webcam"
        c["index"] = int(index)
        c["backend"] = backend
    return edit_config(mut)


def apply_camera_ndi(source: str, ips: str = ""):
    def mut(cfg):
        c = cfg.setdefault("camera", {})
        c["type"] = "ndi"
        c["ndi_source"] = source
        if ips:
            c["ndi_ips"] = ips
    return edit_config(mut)


def apply_config_splitter(mon: dict, layout: str, n_proj: int, cam: dict | None):
    """Un monitor grande (splitter) + layout NxM o rect personalizado."""
    def mut(cfg):
        cols, rows = layout.lower().split("x")
        cfg["displays"] = [{"name": f"Splitter/{mon['name']}", "x": mon["x"], "y": mon["y"],
                            "width": mon["w"], "height": mon["h"], "layout": layout}]
        cfg["projectors"] = [{"name": f"proyector{i + 1}", "display": 0, "output": i + 1}
                             for i in range(n_proj)]
        if cam:
            cfg["camera"] = cam
        return cfg
    return edit_config(mut)


def suggest_layout(w: int, h: int) -> str:
    """Sugiere 'cols x filas' asumiendo una salida de 1920x1080 por cabeza."""
    cols = max(1, int(w) // 1920)
    rows = max(1, int(h) // 1080)
    return f"{cols}x{rows}"


def apply_config_no_splitter(mons: list[dict], cam: dict | None):
    """Sin splitter: un monitor de Windows por proyector (formato B, sin 'displays')."""
    def mut(cfg):
        cfg.pop("displays", None)
        cfg["projectors"] = [{"name": f"proyector{i + 1}", "x": m["x"], "y": m["y"],
                              "width": m["w"], "height": m["h"]}
                             for i, m in enumerate(mons)]
        if cam:
            cfg["camera"] = cam
        return cfg
    return edit_config(mut)


def apply_scan_save_raw(v: bool):
    """Guarda todas las capturas crudas del escaneo en work/raw_projN/."""
    def mut(cfg):
        cfg.setdefault("scan", {})["save_raw"] = bool(v)
    return edit_config(mut)


# ---- bóveda Obsidian ------------------------------------------------------
def _registro_insert_index(lines: list[str]) -> int:
    for i, l in enumerate(lines):
        s = l.strip()
        body = s.replace("|", "").replace(" ", "")
        if s.startswith("|") and body and set(body) <= {"-", ":"}:
            return i + 1
    return len(lines)


def obsidian_log_session(meta: dict) -> str:
    """Crea la nota de sesión y agrega una fila a Registro de calibraciones.md."""
    if not VAULT.exists():
        return "La bóveda Obsidian no existe: crea ObsidianVault/LibreWarp/ primero."
    try:
        today = dt.date.today().isoformat()
        projs = meta.get("projectors", [])
        ustats = meta.get("uniformity", {})
        cov = [float(p.get("coverage") or 0) for p in projs] or [0.0]
        status = "OK" if min(cov) >= 0.5 else "revisar"
        body = (
            "---\n"
            "type: sesion\n"
            f"fecha: {today}\n"
            f"proyectores: {len(projs)}\n"
            f"uniformidad: {ustats.get('brightness_min_over_max', '-')}\n"
            f"status: {status}\n"
            "---\n"
            "# Sesión de calibración\n\n"
            f"- Aspecto estimado de pantalla: {meta.get('aspect_hint_from_camera', '-')}\n"
            f"- Máx. solape de proyectores: {meta.get('max_overlap', '-')}\n"
            f"- Uniformidad (min/max sin compensar): {ustats.get('brightness_min_over_max', '-')}\n"
            f"- Luz conservada con Uniformity=1: "
            f"{round(ustats.get('light_kept_avg', 0) * 100)}%\n\n"
            "## Problemas → buscar en [[Fallos y soluciones]]\n"
        )
        note = VAULT / f"Sesion {today}.md"
        if note.exists():
            note = VAULT / f"Sesion {today} {dt.datetime.now().strftime('%H%M')}.md"
        note.write_text(body, encoding="utf-8")

        reg = VAULT / "Registro de calibraciones.md"
        row = (f"| {today} | {len(projs)} proyectores | mapas OK "
               f"(aspe {meta.get('aspect_hint_from_camera', '-')}) "
               f"| {ustats.get('brightness_min_over_max', '-')} "
               f"| [[{note.name[:-3]}]] |\n")
        txt = reg.read_text(encoding="utf-8")
        lines = txt.splitlines(keepends=True)
        lines.insert(_registro_insert_index(lines), row)
        reg.write_text("".join(lines), encoding="utf-8")
        return f"Bóveda actualizada: {note.name} + fila en Registro de calibraciones"
    except Exception as e:
        return f"No se pudo escribir en la bóveda: {e}"


# ---- procesos -------------------------------------------------------------
def run_streaming(args: list[str], key_prefix="proc", py: str | None = None,
                  on_line=None):
    code = st.container()
    log_lines = st.session_state.get(key_prefix + "_log", [])
    st.session_state[key_prefix + "_log"] = log_lines
    try:
        proc = subprocess.Popen([py or PY] + args, cwd=str(HERE),
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace",
                                env={**os.environ, "PYTHONUNBUFFERED": "1"})
    except Exception as e:
        code.error(f"No se pudo lanzar el proceso: {e}")
        return 1
    with code:
        box = st.empty()
        for line in proc.stdout:
            log_lines.append(line.rstrip())
            box.code("\n".join(log_lines[-400:]), language="console")
            if on_line is not None:
                try:
                    on_line(line)
                except Exception:
                    pass
    proc.wait()
    return proc.returncode


def calibrate_cli(args: list[str], key_prefix="proc", on_line=None):
    return run_streaming(["calibrate.py"] + args, key_prefix, on_line=on_line)


# ---- agente copiloto (fcc-claude / fcc-server) -----------------------------
FCC_PORT = 8082


def _port_open(port: int, host: str = "127.0.0.1", timeout: float = 0.3) -> bool:
    import socket
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _open_console(cmd: str, cwd: Path, title: str):
    if os.name != "nt":
        raise RuntimeError("Estos botones sólo funcionan en Windows.")
    if shutil.which(cmd) is None:
        raise RuntimeError(f"'{cmd}' no está en el PATH.")
    subprocess.Popen(f'start "{title}" cmd /k {cmd}', shell=True, cwd=str(cwd))


# ---- dibujo de bordes -----------------------------------------------------
EDGE_COLORS = {"top": (0, 0, 255), "right": (0, 200, 0),
               "bottom": (255, 90, 0), "left": (255, 0, 255)}


def _bgr(img: np.ndarray) -> np.ndarray:
    v = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if v.dtype != np.uint8:
        v = np.clip(v, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(v)


def _draw_marks(vis, edges, active, to_disp, fine=None):
    for k, pts in edges.items():
        if not pts:
            continue
        col = EDGE_COLORS.get(k, (255, 255, 255))
        qs = [to_disp(p) for p in pts]
        if len(qs) > 1:
            cv2.polylines(vis, [np.array(qs, np.int32).reshape(-1, 1, 2)], False,
                          col, 2 if k == active else 1, cv2.LINE_AA)
        for j, q in enumerate(qs):
            r = 5 if k == active else 3
            cv2.circle(vis, q, r, col, -1, cv2.LINE_AA)
            cv2.circle(vis, q, r + 1, (255, 255, 255), 1, cv2.LINE_AA)
            if k == active:
                for th, cc in ((2, (0, 0, 0)), (1, (255, 255, 255))):
                    cv2.putText(vis, str(j + 1), (q[0] + 7, q[1] - 7),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, cc, th, cv2.LINE_AA)
    if fine is not None and edges.get(active) and 0 <= fine < len(edges[active]):
        q = to_disp(edges[active][fine])
        cv2.circle(vis, q, 12, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(vis, (q[0] - 17, q[1]), (q[0] - 7, q[1]), (255, 255, 255), 1)
        cv2.line(vis, (q[0] + 7, q[1]), (q[0] + 17, q[1]), (255, 255, 255), 1)
        cv2.line(vis, (q[0], q[1] - 17), (q[0], q[1] - 7), (255, 255, 255), 1)
        cv2.line(vis, (q[0], q[1] + 7), (q[0], q[1] + 17), (255, 255, 255), 1)


def _coons_grid(vis, edges, to_disp, n: int = 7):
    if any(len(edges.get(k, [])) < 2 for k in EDGES_ORDER):
        return vis
    try:
        from lw.screen import coons
        s = np.linspace(0, 1, 80)
        for a in np.linspace(0, 1, n + 2)[1:-1]:
            for P in (coons(edges, np.full_like(s, a), s), coons(edges, s, np.full_like(s, a))):
                cv2.polylines(vis, [np.array([to_disp(p) for p in P], np.int32)],
                              False, (170, 170, 170), 1, cv2.LINE_AA)
    except Exception:
        pass
    return vis


def overview_view(img_bgr, edges, active, dw: int, fine=None):
    H, W = img_bgr.shape[:2]
    dh = max(1, int(round(dw * H / W)))
    vis = cv2.resize(img_bgr, (dw, dh), interpolation=cv2.INTER_AREA)
    sx, sy = dw / W, dh / H

    def to_disp(p):
        return int(round(p[0] * sx)), int(round(p[1] * sy))

    _coons_grid(vis, edges, to_disp)
    _draw_marks(vis, edges, active, to_disp, fine)
    return vis, sx, sy


def zoom_view(img_bgr, edges, active, dw: int, center, zoom: float, fine=None):
    H, W = img_bgr.shape[:2]
    dh = max(1, int(round(dw * H / W)))
    cw = max(8, min(W, int(round(W / zoom))))
    ch = max(8, min(H, int(round(H / zoom))))
    ix = int(round(center[0] - cw / 2))
    iy = int(round(center[1] - ch / 2))
    ix = min(max(0, ix), W - cw)
    iy = min(max(0, iy), H - ch)
    crop = img_bgr[iy:iy + ch, ix:ix + cw]
    zpp = dw / cw
    vis = cv2.resize(crop, (dw, dh),
                     interpolation=cv2.INTER_NEAREST if zpp >= 3 else cv2.INTER_LINEAR)
    if zpp >= 4:
        for k in range(cw + 1):
            x = int(round(k * dw / cw))
            mm = (ix + k) % 10 == 0
            cv2.line(vis, (x, 0), (x, dh), (150, 150, 150) if mm else (85, 85, 85), 1)
        for k in range(ch + 1):
            y = int(round(k * dh / ch))
            mm = (iy + k) % 10 == 0
            cv2.line(vis, (0, y), (dw, y), (150, 150, 150) if mm else (85, 85, 85), 1)

    def to_disp(p):
        return (int(round((p[0] - ix) * dw / cw)),
                int(round((p[1] - iy) * dh / ch)))

    _draw_marks(vis, edges, active, to_disp, fine)
    cv2.rectangle(vis, (0, 0), (dw - 1, dh - 1), (200, 200, 200), 1)
    return vis, ix, iy, cw, ch, dh


def click_component(img_bgr, width: int, key: str):
    if not HAS_CLICKS:
        return None
    rgb = cv2.cvtColor(_bgr(img_bgr), cv2.COLOR_BGR2RGB)
    val = streamlit_image_coordinates(rgb, width=int(width), key=key,
                                      click_and_drag=False, cursor="crosshair")
    if not val or val.get("x") is None:
        return None
    stamp = val.get("unix_time")
    seen = st.session_state.setdefault("_click_seen", {})
    if seen.get(key) == stamp:
        return None
    seen[key] = stamp
    return float(val["x"]), float(val["y"])


def save_screen(edges: dict, mode: str, path: Path | None = None) -> str:
    from lw.screen import save_edges
    WORK.mkdir(parents=True, exist_ok=True)
    path = Path(path) if path is not None else (WORK / "screen.json")
    save_edges(str(path),
               {k: [[float(p[0]), float(p[1])] for p in edges.get(k, [])] for k in EDGES_ORDER}, mode)
    return f"Guardado {path.name} — modo '{mode}'"


# ==========================================================================
# guía visual de encuadre + feed del escaneo
# ==========================================================================
def lit_region_bbox(img_bgr):
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    thr = max(70.0, float(np.percentile(g, 95)) * 0.75)
    ys, xs = np.nonzero(g >= thr)
    if xs.size < 100:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    H, W = g.shape
    pct = 100.0 * (x1 - x0 + 1) * (y1 - y0 + 1) / (W * H)
    return x0, y0, x1, y1, pct


def _frame_label(img, text, org, color=(0, 230, 0)):
    x, y = org
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(img, (x, y - th - 10), (x + tw + 10, y + 4), (35, 35, 35), -1)
    cv2.putText(img, text, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                color, 1, cv2.LINE_AA)


def blend_band_image(w: int, h: int, left: float, right: float) -> np.ndarray:
    """Imagen de guía: blanco con la rampa de solape en los bordes del proyector.

    Se PROYECTA, así que la franja aparece en la pared en su sitio real y no hay
    que adivinar una geometría de cámara. El borde del frame del proyector es el
    borde de su ventana de lienzo, o sea donde el calibrador aplica el blend: la
    franja que se ve en la pared es exactamente la zona de solape con el vecino.
    `left`/`right` = fracción del ancho (0 = sin vecino de ese lado).
    """
    img = np.full((h, w), 255, np.uint8)
    ya, yb = int(0.22 * h), int(0.78 * h)      # la marca vertical no llega a los bordes
    for frac, to_right in ((right, True), (left, False)):
        f = int(round(float(frac) * w))
        if frac <= 0 or f < 4:
            continue
        # La rampa baja a 200 y no a negro: el veredicto de encuadre mide la zona
        # \"iluminada\" con un umbral de 0.75*p95 y una franja muy oscura le haría
        # creer que el haz es más chico. La marca vertical sí es oscura, pero sólo
        # en el centro del alto, así no cambia esa medida.
        ramp = np.linspace(255, 200, f).astype(np.uint8)
        if to_right:
            img[:, w - f:] = ramp
            img[ya:yb, w - f:w - f + 2] = 40
        else:
            img[:, :f] = ramp[::-1]
            img[ya:yb, max(0, f - 2):f] = 40
    return img


def seam_overlap_text(i: int, n: int) -> str:
    """Solape ya medido por los escaneos en las costuras de este proyector."""
    from lw.pipeline import scan_overlaps
    parts = []
    for j, side in ((i - 1, "left"), (i + 1, "right")):
        if not (0 <= j < n):
            continue
        vals = []
        for own, other in ((i, j), (j, i)):
            v = scan_overlaps(WORK, own).get(
                f"overlap_{side if own == i else ('right' if side == 'left' else 'left')}")
            if v is not None and np.isfinite(v) and 0.0 < v < 0.45:
                vals.append(float(v))
        parts.append(f"P{i}-P{j} " + (f"{100 * sum(vals) / len(vals):.0f}%"
                                     if vals else "sin medir"))
    return " · ".join(parts)


def framing_overlay(frame_bgr, draw_lit=True, margins=True):
    H, W = frame_bgr.shape[:2]
    m = max(8, int(0.05 * min(H, W)))
    vis = frame_bgr
    if draw_lit:
        r = lit_region_bbox(frame_bgr)
        if r is not None:
            x0, y0, x1, y1, pct = r
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 220, 0), 2)
            _frame_label(vis, f"iluminado {pct:.0f}%", (x0 + 4, y0 + 24))
    if margins:
        cv2.rectangle(vis, (m, m), (W - 1 - m, H - 1 - m), (80, 180, 255), 2)
        for i in (1, 2):
            cv2.line(vis, (W * i // 3, m), (W * i // 3, H - 1 - m), (110, 110, 110), 1)
            cv2.line(vis, (m, H * i // 3), (W - 1 - m, H * i // 3), (110, 110, 110), 1)
        cv2.line(vis, (W // 2 - 16, H // 2), (W // 2 + 16, H // 2), (80, 180, 255), 1)
        cv2.line(vis, (W // 2, H // 2 - 16), (W // 2, H // 2 + 16), (80, 180, 255), 1)
    return vis


def framing_verdict_text(g) -> str:
    """Veredicto de encuadre desde un frame en escala de grises (las mismas
    reglas que coverage_verdict, pero para usarlo sobre la vista en vivo)."""
    if g is None:
        return "no se pudo leer la imagen"
    r = lit_region_bbox(_bgr(g))
    if r is None:
        return ("❌ Casi nada iluminado: ¿es una CÁMARA mirando la pared? Revisa el "
                "tipo de fuente, el encuadre y la exposición (sin rojo en el paso 3).")
    x0, y0, x1, y1, pct = r
    H, W = g.shape[:2]
    m = max(8, int(0.05 * min(H, W)))
    inside = x0 >= m and y0 >= m and x1 < W - m and y1 < H - m
    if pct >= 80 and inside:
        return (f"✅ Encuadre correcto: la pared llena la cámara "
                f"({pct:.0f}% del encuadre, dentro del margen del 5%).")
    why = []
    if pct < 80:
        why.append("la pared es chiquita en el cuadro")
    if not inside:
        why.append("la pared SE SALE del margen del 5%")
    return (f"⚠️ {', '.join(why)} ({pct:.0f}% del encuadre): recoloca la cámara — "
            "centrada, lejos y con zoom, hasta que lo iluminado llene el marco ámbar.")


def projector_canvas(cfg):
    """DisplayCanvas con la config YA NORMALIZADA → (canvas, cfg_norm, error).

    load_config() devuelve config.json tal cual, sin pasar por lw.config.normalize():
    entonces no existen 'rect'/'width'/'height', compose() lanzaba KeyError y el
    except lo tragaba — la vista de proyector NO PINTABA NADA (ni el blanco ni los
    patrones de escaneo salían en la pared, aunque los pidieras).
    """
    if not cfg or not cfg.get("projectors"):
        return None, None, "no hay proyectores en config.json"
    import copy as _copy
    from lw.config import normalize
    c = _copy.deepcopy(cfg)
    if not c.get("displays"):
        c.pop("displays", None)          # normalize() los crea (formato B)
    try:
        normalize(c)
        from lw.devices import DisplayCanvas
        return DisplayCanvas(c), c, None
    except Exception as e:
        return None, None, str(e)


def close_preview_canvas():
    """Cierra la ventana de proyector que haya abierto la vista previa."""
    got = st.session_state.pop("_prev_cv", None)
    if got is None:
        return
    try:
        got[0].close()
    except Exception:
        pass


def pattern_cycle(cfg):
    """Infinito {(w,h): (nombre, imagen)} con la secuencia REAL de escaneo.

    Cada proyector muestra el patrón de su propio tamaño y todos van en el mismo
    paso: sirve para ver los patrones de escaneo en la pared a la exposición con
    la que vas a escanear (antes sólo se veía un patrón fijo y mal generado).
    """
    from lw import patterns
    sc = (cfg or {}).get("scan", {})
    cell = int(sc.get("cell_size", 24))
    nbits = int(sc.get("phase_steps", 8))
    per = int(sc.get("phase_period", 32))
    ints = int(sc.get("pattern_intensity", 255))
    color = bool(sc.get("color_patterns", True))
    keys = {(int(p["width"]), int(p["height"]))
            for p in (cfg or {}).get("projectors", [])} or {(1920, 1080)}

    def _new(k):
        return patterns.make_sequence(k[0], k[1], cell, ints, nbits, per, color)

    gens = {k: _new(k) for k in keys}
    while True:
        out = {}
        for k, g in gens.items():
            try:
                out[k] = next(g)
            except StopIteration:
                gens[k] = g = _new(k)
                out[k] = next(g)
        yield out


def coverage_verdict(path: Path):
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if g is None:
        return None, "no se pudo leer la imagen"
    return framing_overlay(_bgr(g), draw_lit=False), framing_verdict_text(g)


def scan_caption(work: Path, i: int) -> str:
    """«Último escaneo: 13:02:30 · exposición -6 · decodificado 48 %» del .npz.

    La exposición es la que usó DE VERDAD la cámara al capturar: si siempre sale
    el mismo número, el escaneo no está viendo lo que ajustas en el diálogo del
    driver (config.json la vuelve a fijar al abrir la cámara).
    """
    p = Path(work) / f"scan_proj{i}.npz"
    if not p.exists():
        return ""
    try:
        with np.load(p) as d:
            exp = float(d["exposure"]) if "exposure" in d.files else float("nan")
            when = float(d["scan_unix_time"]) if "scan_unix_time" in d.files else None
            dec = float(d["valid"].mean() * 100) if "valid" in d.files else float("nan")
    except Exception:
        return ""
    bits = []
    if when:
        bits.append("escaneado " + dt.datetime.fromtimestamp(when).strftime("%H:%M:%S"))
    if exp == exp:
        bits.append(f"exposición {exp:.0f}")
    if dec == dec:
        bits.append(f"decodificado {dec:.0f}%")
    return "Último escaneo: " + " · ".join(bits) if bits else ""


def file_stamp(path: Path | None) -> tuple[int, int] | None:
    """Huella (mtime_ns, tamaño) de un archivo, o None si no existe.

    Sirve para saber si la imagen que se está marcando en el paso 5 es la de
    ESTE escaneo o la de uno anterior: si cambia, la imagen es nueva y los
    puntos viejos ya no valen (y, además, la vista debe redibujarse).
    """
    if path is None:
        return None
    try:
        s = Path(path).stat()
    except OSError:
        return None
    return int(s.st_mtime_ns), int(s.st_size)


def stamp_label(stamp) -> str:
    """Texto legible de una huella: hora de la captura."""
    if not stamp:
        return "sin imagen"
    import datetime as _dt
    t = _dt.datetime.fromtimestamp(stamp[0] / 1e9)
    return t.strftime("%d/%m %H:%M:%S")


def points_match_scan(screen_path: Path, bg_path: Path | None) -> bool:
    """¿Los puntos guardados se marcaron sobre la imagen que hay ahora?

    Se comparan mtimes: si el escaneo del proyector es más nuevo que
    screen_projN.json, esos bordes son de una imagen anterior (otra pose de
    cámara) y no sirven. Sin imagen de referencia no hay con qué comparar y se
    devuelve False (se avisa en pantalla).
    """
    if bg_path is None or not Path(bg_path).exists() or not Path(screen_path).exists():
        return False
    try:
        # 0.75 s de margen por granularidad de timestamps de algunos sistemas de
        # archivos; un re-escaneo real tarda bastante más que eso.
        return Path(screen_path).stat().st_mtime >= Path(bg_path).stat().st_mtime - 0.75
    except OSError:
        return False


def newest_raw_capture(work: Path):
    work = Path(work)
    best, tbest = None, 0.0
    pattern = "*.png" if work.name.startswith("raw_proj") else "raw_proj*/*.png"
    for f in work.glob(pattern):
        try:
            t = f.stat().st_mtime
        except OSError:
            continue
        if t > tbest:
            best, tbest = f, t
    return best, tbest


# ==========================================================================
# estados de los pasos (checar disco)
# ==========================================================================
def stage_status() -> dict:
    c = load_config()
    scans = list(WORK.glob("scan_proj*.npz")) if WORK.exists() else []
    n_proj = len(c["projectors"]) if c and "projectors" in c else 0
    per_proj = n_proj > 0 and any((WORK / f"all_white_proj{i}.png").exists()
                                  for i in range(n_proj))
    single = (WORK / "all_white.png").exists() and len(scans) > 0
    scanned_all = (all((WORK / f"scan_proj{i}.npz").exists() for i in range(n_proj))
                   if per_proj else single)
    edges_any = (WORK / "screen.json").exists() or bool(list(WORK.glob("screen_proj*.json")))
    edges_all = ((WORK / "screen.json").exists() or
                 (n_proj > 0 and all((WORK / f"screen_proj{i}.json").exists()
                                     for i in range(n_proj))))
    return {
        "venv": VENV_PY.exists(),
        "config": c is not None,
        "scanned": bool(scans) and (single or per_proj),
        "scanned_all": bool(scanned_all),
        "per_projector": bool(per_proj),
        "edges": edges_all,          # listo para calcular
        "edges_any": bool(edges_any),
        "computed": (OUT / "calibration.json").exists(),
        "n_scans": len(scans),
        "n_proj": n_proj,
    }


STEPS = ["🏠 Home", "1 · Config pantallas", "2 · Identificar", "3 · Elegir/ajustar cámara",
         "4 · Escanear", "5 · Marcar bordes", "6 · Calcular", "7 · Verificar",
         "8 · TouchDesigner", "9 · Agente + Obsidian"]


def auto_step(ss: dict):
    if not ss["venv"]:
        return STEPS[0]
    if not ss["config"]:
        return STEPS[1]
    if not ss["scanned"]:
        return STEPS[2]
    if not ss.get("scanned_all", True):
        return STEPS[4]              # quedan proyectores por escanear
    if not ss["edges"]:
        return STEPS[5]
    if not ss["computed"]:
        return STEPS[6]
    return STEPS[7]


# ==========================================================================
# sidebar
# ==========================================================================
ss = stage_status()
with st.sidebar:
    st.title("🎯 LibreWarp")
    st.caption("Calibración warp & blend con cámara")
    ss["identify_ok"] = st.checkbox("2 · Identificó proyectores bien", key="ck_identify",
                                    value=st.session_state.get("ck_identify", False))
    ss["adjust_ok"] = st.checkbox("3 · Guardó ajuste de cámara (s)", key="ck_adjust",
                                  value=st.session_state.get("ck_adjust", False))
    st.divider()
    st.subheader("Progreso")

    # MODO SECCIONES: Panel de control en la barra lateral
    st.markdown("---")
    st.subheader("Modo Secciones (escaneo por partes)")
    section_mode = st.checkbox("Activar", value=st.session_state.get("section_mode", False), key="ck_section_mode")
    st.session_state["section_mode"] = section_mode
    if section_mode:
        if st.button("Definir secciones"):
            st.session_state["show_section_ui"] = True
    st.markdown("---")

    st.markdown(
        ("✅ " if ss["venv"] else "⬜ ") + "Instalación\n\n"
        + ("✅ " if ss["config"] else "⬜ ") + "config.json\n\n"
        + ("✅ " if ss["identify_ok"] else "⬜ ") + "Identificar\n\n"
        + ("✅ " if ss["adjust_ok"] else "⬜ ") + "Ajustar cámara\n\n"
        + (f"✅ Escaneo ({ss['n_scans']})\n\n" if ss["scanned"] else "⬜ Escaneo\n\n")
        + ("✅ " if ss["edges"] else "⬜ ") + "Bordes\n\n"
        + ("✅ " if ss["computed"] else "⬜ ") + "Mapas en output/\n\n"
    )
    st.session_state.setdefault("_step_sel", "🏠 Home")
    step = st.radio("Ir al paso", STEPS, key="_step_sel")

_pend_save = st.session_state.pop("_do_save", None)
if _pend_save:
    _edges_s, _mode_s, _path_s, _msgkey_s = _pend_save
    st.session_state[_msgkey_s] = (True, save_screen(_edges_s, _mode_s, _path_s) + " → paso 6.")


# ==========================================================================
# FUNCIONES AUXILIARES PARA VALIDACIÓN DE FRAMES
# ==========================================================================
def aplicar_ajustes_de_imagen(frame, exposure, contrast, saturation):
    """Aplica ajustes DIGITALES de exposición y ganancia al frame"""
    # Ajuste de Exposición/Ganancia Digital: f(x) = alpha * x + beta
    # exposure slider en wizard.py va de -14 a -1, lo mapeamos a ganancia positiva
    # Si la cámara es oscura, aumentamos alpha
    alpha = 1.0 + (exposure + 6) * 0.2  # +6 es el default, aumentamos brillo
    alpha = max(0.1, alpha)

    # Aplicar ganancia (alpha) y brillo base
    result = cv2.convertScaleAbs(frame, alpha=alpha, beta=0)

    # Ajustar saturación si es necesario (opcional)
    if saturation != 1.0:
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype("float32")
        hsv[:,:,1] = np.clip(hsv[:,:,1] * saturation, 0, 255)
        result = cv2.cvtColor(hsv.astype("uint8"), cv2.COLOR_HSV2BGR)

    return result


def agregar_overlay_de_analisis(frame):
    """Añade overlay de sobre/subexposición + histograma"""
    h, w = frame.shape[:2]

    # Crear copia para dibujar overlay
    overlay = frame.copy()

    # Detectar zonas sobrexpuestas (rojo) y subxpuestas (azul)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    over_mask = gray > 250
    under_mask = gray < 15

    # Aplicar colores de alerta (50% transparencia)
    overlay[over_mask] = [0, 0, 255]  # Rojo en BGR
    overlay[under_mask] = [255, 0, 0]  # Azul en BGR
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, overlay)

    # Añadir histograma en la parte inferior
    hist_height = 50
    hist_canvas = np.zeros((hist_height, w, 3), dtype="uint8")
    hist = cv2.calcHist([gray], [0], None, [256], [0,256])
    cv2.normalize(hist, hist, 0, hist_height, cv2.NORM_MINMAX)

    for x in range(256):
        cv2.line(
            hist_canvas,
            (x, hist_height),
            (x, hist_height - int(hist[x])),
            (255, 255, 255),
            1
        )

    # Combinar frame con histograma
    result = np.vstack([overlay, hist_canvas])

    # Añadir líneas de referencia en histograma
    cv2.line(result, (0, h + 25), (w, h + 25), (0, 255, 0), 1)  # Línea central
    cv2.line(result, (0, h + 10), (w, h + 10), (0, 0, 255), 1)  # Límite superior
    cv2.line(result, (0, h + 40), (w, h + 40), (255, 0, 0), 1)  # Límite inferior

    return result


# ==========================================================================
# HOME: dashboard de estado (PANTALLA INICIAL)
# ==========================================================================
if step.startswith("🏠"):
    st.title("🎯 LibreWarp — centro de calibración")
    st.caption("Estado REAL leído de tus archivos (config.json, work/, output/). "
               "No se elige nada automáticamente: tú navegas y eliges.")

    cfg = load_config()
    cam_cfg = (cfg or {}).get("camera")
    cam_desc = ("⏳ sin configurar — tú la eliges en el paso 3"
                if cam_cfg is None
                else (f"NDI «{cam_cfg.get('ndi_source', '(primera fuente)')}»"
                      if str(cam_cfg.get("type", "webcam")).lower() == "ndi"
                      else f"webcam índice {cam_cfg.get('index')} ({cam_cfg.get('backend', '')})"))
    if cfg and cfg.get("displays"):
        scr_desc = " · ".join(
            f"{d.get('name', '?')} {d.get('width')}x{d.get('height')}"
            f" @{d.get('x')},{d.get('y')}" for d in cfg["displays"])
    elif cfg and "projectors" in cfg:
        scr_desc = "sin splitter — un monitor por proyector"
    else:
        scr_desc = "sin config.json"

    cards = [
        ("Entorno Python", ss["venv"],
         "venv listo (.venv/Scripts/python.exe)" if ss["venv"]
         else "FALTA: doble clic en `librewarp.bat` (opción 0)"),
        ("Pantallas / config", ss["config"], f"{scr_desc} — {ss['n_proj']} proyectores"),
        ("Cámara", cam_cfg is not None, cam_desc),
        ("Escaneo", ss["scanned"],
         f"{ss['n_scans']} escaneo(s) en work/" if ss["n_scans"]
         else "sin escaneos en work/"),
        ("Bordes marcados", ss["edges"],
         "todos los proyectores" if ss.get("per_projector") else "work/screen.json existe"
         if ss["edges"] else "faltan bordes (paso 5)"),
        ("Mapas listos", ss["computed"],
         "output/calibration.json + projN_map.exr" if ss["computed"]
         else "output/ sin mapas (paso 6)"),
    ]
    for row in (cards[:3], cards[3:]):
        cc = st.columns(3)
        for (title, ok, detail), col in zip(row, cc):
            with col:
                st.markdown(("✅ " if ok else "⬜ ") + f"**{title}**\n\n{detail}")

    if not ss["venv"]:
        st.error("Falta el venv: corre `librewarp.bat` (opción 0), o crea el entorno aquí.")
        if st.button("🛠 Crear el entorno ahora (≈2–3 min)"):
            run_streaming(["-m", "venv", ".venv"], key_prefix="venv", py=sys.executable)
            if VENV_PY.exists():
                run_streaming(["-m", "pip", "install", "-r", "requirements.txt"],
                              key_prefix="venvdeps", py=str(VENV_PY))
                st.rerun()
            else:
                st.error("No se pudo crear .venv. Instala Python desde python.org "
                         "marcando «Add python.exe to PATH».")

    nxt = auto_step(ss)
    st.divider()
    st.markdown("#### 🧭 Navegación — gestiona todo desde aquí")
    st.caption("✅ ya hecho · ⬜ pendiente · el botón AZUL es el siguiente paso bloqueado.")
    tiles = [
        (STEPS[1], "🖥 Pantallas", ss["config"],
         f"{ss['n_proj']} proyectores configurados" if ss["config"] else "configurar displays"),
        (STEPS[2], "🔢 Identificar", bool(ss.get("identify_ok")),
         "ver SU número en cada proyector"),
        (STEPS[3], "📷 Cámara", cam_cfg is not None,
         cam_desc if cam_cfg else "elegir fuente + exposición"),
        (STEPS[4], "🎞 Escanear", ss["scanned"],
         f"{ss['n_scans']} escaneo(s) en work/" if ss["scanned"] else "con feed en vivo"),
        (STEPS[5], "✏️ Bordes", ss["edges"], "clics sobre all_white.png"),
        (STEPS[6], "🧮 Calcular", ss["computed"], "mapas .exr + calibration.json"),
        (STEPS[7], "✅ Verificar", ss["computed"], "rejilla/blanco en los proyectores"),
        (STEPS[8], "🎯 TouchDesigner", ss["computed"], "COMP librewarp + ventanas"),
        (STEPS[9], "🤖 Agente + Obsidian", True, "fcc-claude y bóveda de fallos"),
    ]

    def _goto_step(name: str) -> None:
        st.session_state["_step_sel"] = name

    for r in range(0, 9, 3):
        cols = st.columns(3)
        for (step_name, title, ok, detail), col in zip(tiles[r:r + 3], cols):
            with col:
                st.button(("✅ " if ok else "⬜ ") + title + "\n" + detail,
                          key=f"tile_{step_name}", use_container_width=True,
                          type="primary" if step_name == nxt else "secondary",
                          on_click=_goto_step, args=(step_name,),
                          help=f"Ir a {step_name}")

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        if st.button("⚡ Detectar monitores + cámaras + NDI (SÓLO LISTA, no elige)",
                     use_container_width=True,
                     help="Lista todo lo conectado para decidir la config. La cámara se "
                          "elige EXPLÍCITAMENTE en el paso 3; la wizard nunca la asigna sola."):
            c1b, c2b = st.columns(2)
            with c1b:
                st.markdown("**Monitores / salidas**")
                for m in detect_monitors():
                    st.write(f"· [{m['idx']}] {m['name']}: ({m['x']},{m['y']}) {m['w']}x{m['h']}")
            with c2b:
                st.markdown("**Cámaras**")
                for c in detect_webcams():
                    st.write(f"· webcam index {c['index']} · {c['res']}")
                try:
                    srcs = detect_ndi(None, 5.0)
                    for sname in srcs:
                        st.write(f"· NDI: {sname}")
                except Exception as e:
                    st.write(f"· NDI: no disponible ({e})")
    with c2:
        from lw.sim import SCENES
        _escenas = list(SCENES)
        _scene = st.selectbox("Escena simulada", _escenas, key="sim_scene",
                               format_func=lambda s: f"{s} — {SCENES[s]['titulo']}")
        st.caption(SCENES[_scene]["nota"])
        if st.button("🧪 Prueba completa sin hardware (~40 s)", use_container_width=True):
            run_streaming(["test_simulated.py", _scene], key_prefix="sim")

    st.caption("Flujo: configurar pantallas → identificar → elegir cámara → ajustar → "
               "escanear (con feed en vivo) → marcar bordes → calcular → verificar → "
               "TouchDesigner. Si te atascas, abre **fcc-claude** y pídele «guiame con "
               "la calibración» (paso 9).")

# ==========================================================================
# paso 1: config pantallas (genérico)
# ==========================================================================
elif step.startswith("1 "):
    st.title("1 · Configurar las pantallas / salidas")
    st.markdown(
        "Detecta lo conectado y elige el modo:\n\n"
        "**A · Con splitter** (Matrox QuadHead2Go u otro): Windows ve UN monitor grande y el "
        "splitter lo reparte. Ese monitor = display con `layout NxM` o `rect` por salida.\n"
        "**B · Sin splitter**: cada proyector es un monitor de Windows distinto (N≤Nº de monitores).")

    # === Sección de configuración de superficie ===
    st.markdown("---")
    st.subheader("📐 Tipo de superficie")
    cfg_current = load_config() or {}
    surface_type = st.selectbox(
        "Tipo de superficie:",
        ["convex", "concave", "flat"],
        index=["convex", "concave", "flat"].index(cfg_current.get("surface_type", "convex")),
        help="convex = ciclorama convexo (domo); concave = cóncavo (cilindro); flat = plano"
    )
    surface_radius = st.number_input(
        "Radio de la superficie (metros):",
        min_value=0.1,
        max_value=10.0,
        value=cfg_current.get("surface_radius", 1.0),
        help="Radio para cálculos de warp en superficies curvas. Para planos, usa 1.0."
    )
    if st.button("💾 Guardar configuración de superficie", type="secondary"):
        def _mut_surface(cfgm):
            cfgm["surface_type"] = surface_type
            cfgm["surface_radius"] = surface_radius
        edit_config(_mut_surface)
        st.success("✅ Configuración de superficie guardada")
        st.rerun()

    if st.button("🔎 Detectar monitores"):
        st.session_state["screens"] = detect_monitors()
        st.rerun()

    screens = st.session_state.get("screens")
    cfg = load_config()
    if cfg:
        st.info("config.json vigente: "
                + (", ".join(f"{d.get('name','?')} {d.get('width')}x{d.get('height')} "
                             f"({d.get('x')},{d.get('y')})" for d in cfg.get("displays", []))
                   if cfg.get("displays")
                   else "sin 'displays' (un monitor por proyector)")
                + f" — {len(cfg['projectors'])} proyectores")
        if st.button("Abrir config.json en el Bloc de notas"):
            os.startfile(str(CONFIG_PATH))

    if screens:
        mode = st.radio("Modo", ["A — Con splitter (un monitor grande)",
                                 "B — Sin splitter (un monitor por proyector)"])
        names = [f"[{m['idx']}] {m['name']} {m['w']}x{m['h']} ({m['x']},{m['y']})" for m in screens]
        if mode.startswith("A"):
            big = st.selectbox("Monitor del splitter (el GRANDE, p. ej. 7680x1080)", names)
            mon = screens[names.index(big)]
            layout = st.text_input("Layout del splitter (cols x filas)",
                                   value=suggest_layout(mon["w"], mon["h"]),
                                   key=f"layout_{mon['idx']}_{mon['w']}x{mon['h']}",
                                   help="Se deduce del tamaño del monitor asumiendo 1920x1080 "
                                        "por salida: 7680x1080 -> 4x1, 5760x1080 -> 3x1, "
                                        "3840x2160 -> 2x2. Puedes cambiarlo o importar rects "
                                        "personalizados a mano en config.json")
            n = st.number_input("Nº de proyectores", 1, 4, 3)
            if st.button("✅ Aplicar config (modo A)", type="primary"):
                apply_config_splitter(mon, layout.strip(), int(n),
                                      cam=(cfg or {}).get("camera"))
                st.success("config.json actualizado (modo A)")
        else:
            sel = st.multiselect("Monitores para los proyectores (en orden izq→der)",
                                 names)
            if sel:
                mons = [screens[names.index(nm)] for nm in sel]
                if st.button("✅ Aplicar config (modo B)", type="primary"):
                    apply_config_no_splitter(mons, cam=(cfg or {}).get("camera"))
                    st.success("config.json actualizado (modo B)")
            else:
                st.warning("Elige al menos un monitor.")
    else:
        st.caption("Pulsa '🔎 Detectar monitores' para poblar esta sección.")

# ==========================================================================
# paso 2: identificar
# ==========================================================================
elif step.startswith("2 "):
    st.title("2 · Identificar proyectores")
    st.markdown(
        "Cada proyector debe mostrar **su número completo** (marco + X + número enteros).\n"
        "Si ves **dos números**, medio marco o marcos cortados → vuelve al paso 1 "
        "(resolución de Windows o bezel/escala del splitter).")
    if st.button("Mostrar identificación en los proyectores", use_container_width=True):
        run_streaming(["calibrate.py", "identify", "-c", "config.json"], key_prefix="ident")

# ==========================================================================
# paso 3.5: validación de frames (NUEVO)
# ==========================================================================
elif step.startswith("3.5 ") or (step.startswith("3 ") and st.session_state.get("_show_validation", False)):
    # If we're on step 3 but validation was triggered, show validation instead
    if step.startswith("3 ") and st.session_state.get("_show_validation", False):
        st.title("3.5 · Validación de Frames para Escaneo Óptimo")
        st.session_state["_show_validation"] = False  # Reset flag
    else:
        st.title("3.5 · Validación de Frames para Escaneo Óptimo")

    st.markdown("""
Valida individualmente CADA frame que usará el escaneo:
▪️ Ajusta exposición/saturación en una **copia del feed** (no afecta hardware)
▪️ Ve guía visual de sobre/subexposición en tiempo real
▪️ Aprobar cada frame antes de pasar al siguiente patrón
▪️ El escaneo usará exclusivamente estos frames aprobados
""")

    # Initialize session state for validation if needed
    if "validation_state" not in st.session_state:
        st.session_state.validation_state = {
            "current_pattern_idx": 0,
            "approved_frames": [],
            "pattern_sequence": [
                {"name": "all_white", "desc": "Blanco puro (exposición)", "color": (255,255,255)},
                {"name": "vertical_stripes", "desc": "Rayas verticales (fase)", "color": (128,128,128)},
                {"name": "horizontal_stripes", "desc": "Rayas horizontales (fase)", "color": (128,128,128)},
                {"name": "checkerboard", "desc": "Tablero de ajedrez (geometría)", "color": (0,0,0)}
            ],
            "video_source": None,
            "frames_needed_per_pattern": 3  # Number of frames to approve per pattern
        }

    state = st.session_state.validation_state

    # Video source selection
    st.subheader("Selección de Fuente de Video")

    # Detect available sources
    @st.cache_data(ttl=5)
    def get_video_sources():
        sources = []

        # USB webcams
        for i in range(6):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    sources.append({
                        "type": "usb",
                        "index": i,
                        "name": f"Webcam USB {i}"
                    })
                cap.release()

        # NDI sources
        try:
            import NDIlib as ndi
            if ndi.initialize():
                ndi_sources = ndi.find_sources()
                for src in ndi_sources:
                    sources.append({
                        "type": "ndi",
                        "name": src.ndi_name,
                        "source": src
                    })
                ndi.destroy()
        except:
            pass

        return sources if sources else [{"type": "usb", "index": 0, "name": "Predeterminado"}]

    video_sources = get_video_sources()

    # Current source selection
    col1, col2 = st.columns([3, 1])
    with col1:
        source_options = [f"{s['name']} ({'USB' if s['type']=='usb' else 'NDI'})" for s in video_sources]
        selected_idx = st.selectbox(
            "Fuente de video para validación",
            options=range(len(source_options)),
            format_func=lambda x: source_options[x],
            index=0 if state["video_source"] is None else
                  next((i for i, s in enumerate(video_sources)
                        if s["index"] == state["video_source"].get("index") and
                         s["type"] == state["video_source"].get("type")), 0),
            key="validation_source_select"
        )
        state["video_source"] = video_sources[selected_idx]

    with col2:
        if st.button("🔄 Refrescar Fuentes"):
            get_video_sources.clear()
            st.rerun()

    # Current pattern info
    current_pattern = state["pattern_sequence"][state["current_pattern_idx"]]
    progress = (state["current_pattern_idx"] + 1) / len(state["pattern_sequence"])

    st.progress(progress)
    st.info(f"""
    **Patrón actual**: {current_pattern['desc']}
    **Progreso**: {state["current_pattern_idx"] + 1}/{len(state["pattern_sequence"])} patrones
    **Frames aprobados para este patrón**: {len(state["approved_frames"])}/{state["frames_needed_per_pattern"]}
    """)

    # Special instruction for first pattern
    if state["current_pattern_idx"] == 0 and len(state["approved_frames"]) == 0:
        st.warning("""
        🔧 **Instrucciones para el patrón de Blanco Puro**:
        1. Ajusta los controles hasta que **NO veas ROJO** en la imagen procesada
        2. El blanco debe verse puro (sin tinte amarillento o azulado)
        3. Aproba 3 frames consistentes antes de continuar
        """)

    # Video feed panels
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Feed RAW (referencia)")
        raw_placeholder = st.empty()
    with col2:
        st.subheader("Feed Procesado (con ajustes)")
        processed_placeholder = st.empty()

    # Adjustment controls
    st.subheader("Ajustes de Imagen")
    exp_col, cont_col, sat_col = st.columns(3)
    with exp_col:
        exposure = st.slider("Exposición", -2.0, 2.0, 0.0, 0.1, key="exp_slider_val")
    with cont_col:
        contrast = st.slider("Contraste", 0.5, 3.0, 1.0, 0.1, key="cont_slider_val")
    with sat_col:
        saturation = st.slider("Saturación", 0.0, 2.0, 1.0, 0.1, key="sat_slider_val")

    # Action buttons
    btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)

    with btn_col1:
        if st.button("📸 Capturar y Procesar Frame", use_container_width=True):
            try:
                # Capture frame
                cap = None
                if state["video_source"]["type"] == "usb":
                    cap = cv2.VideoCapture(state["video_source"]["index"], cv2.CAP_DSHOW)
                elif state["video_source"]["type"] == "ndi":
                    import NDIlib as ndi
                    recv_create = ndi.RecvCreateV3()
                    recv_create.source_to_connect_to = state["video_source"]["source"]
                    cap = ndi.recv_create_v3(recv_create)

                if state["video_source"]["type"] == "usb":
                    ret, frame = cap.read()
                    cap.release()
                    if not ret:
                        st.error("No se pudo capturar frame desde cámara USB")
                else:  # NDI
                    import NDIlib as ndi
                    t, v, a, _ = ndi.recv_capture_v2(cap, 5000)
                    if t == ndi.FRAME_TYPE_VIDEO:
                        frame = ndi.recv_get_video_data_v2(cap, v)
                        ndi.recv_free_video_v2(cap, v)
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    else:
                        st.error("No se recibió frame de video NDI")
                        frame = None
                    ndi.recv_destroy_v3(cap)

                if frame is not None and ret if state["video_source"]["type"] == "usb" else t == ndi.FRAME_TYPE_VIDEO:
                    # Show RAW
                    raw_placeholder.image(frame, caption="Feed RAW", channels="BGR")

                    # Process copy
                    processed = aplicar_ajustes_de_imagen(frame, exposure, contrast, saturation)

                    # Show processed + analysis
                    analyzed = agregar_overlay_de_analisis(processed)
                    processed_placeholder.image(analyzed, caption="Feed Procesado + Análisis", channels="BGR")

                    # Store for possible acceptance
                    st.session_state.last_captured_frame = frame
                    st.session_state.last_processed = processed
                    st.session_state.last_exposure = exposure
                    st.session_state.last_contrast = contrast
                    st.session_state.last_saturation = saturation
                else:
                    st.error("Error al capturar frame")

            except Exception as e:
                st.error(f"Error al capturar frame: {str(e)}")

    with btn_col2:
        if st.button("✅ Aceptar Frame", use_container_width=True, type="primary"):
            if 'last_captured_frame' not in st.session_state:
                st.error("Primero captura un frame")
            else:
                # Save approved frame
                pattern_name = current_pattern['name']
                proj_id = 0  # Assuming single projector for validation; could be extended

                # Create validation directory
                validation_dir = HERE / "work" / "validation_frames"
                validation_dir.mkdir(exist_ok=True)

                # Generate filename
                timestamp = int(time.time() * 1000)
                frame_path = validation_dir / f"proj{proj_id}_{pattern_name}_{len(state['approved_frames']):03d}_{timestamp}.png"

                # Save RAW frame (not processed)
                cv2.imwrite(str(frame_path), st.session_state.last_captured_frame)

                # Record approval
                state["approved_frames"].append({
                    "pattern": pattern_name,
                    "path": str(frame_path),
                    "exposure": st.session_state.last_exposure,
                    "contrast": st.session_state.last_contrast,
                    "saturation": st.session_state.last_saturation,
                    "timestamp": timestamp
                })

                st.success(f"✔ Frame aprobado: {frame_path.name}")

                # Check if we have enough frames for this pattern
                if len(state["approved_frames"]) >= state["frames_needed_per_pattern"]:
                    # Move to next pattern
                    state["current_pattern_idx"] += 1
                    state["approved_frames"] = []  # Reset for next pattern

                    if state["current_pattern_idx"] >= len(state["pattern_sequence"]):
                        # All patterns validated!
                        st.balloons()
                        st.success("🎉 Todos los patrones validados! Preparando para continuar...")

                        # Update config.json to use validation frames
                        def update_config_for_validation(cfg):
                            cfg["validation"] = {
                                "enabled": True,
                                "frame_dir": str(validation_dir),
                                "pattern_sequence": [p["name"] for p in state["pattern_sequence"]],
                                "frames_needed_per_pattern": state["frames_needed_per_pattern"]
                            }
                            return cfg

                        edit_config(update_config_for_validation)
                        st.info("✅ Configuración actualizada. El escaneo usará frames validados.")

                        # Auto-continue to next step after a delay
                        if st.button("▶️ Continuar a Paso 4: Escanear", use_container_width=True):
                            # Clear validation state and go to step 4
                            if "validation_state" in st.session_state:
                                del st.session_state.validation_state
                            st.session_state["_step_sel"] = "4 "
                            st.rerun()
                    else:
                        st.info(f"Pasando al patrón {state['current_pattern_idx'] + 1}: {state['pattern_sequence'][state['current_pattern_idx']]['desc']}")
                        st.rerun()

    with btn_col3:
        if st.button("🔄 Rechazar y Reintentar", use_container_width=True):
            if 'last_captured_frame' in st.session_state:
                del st.session_state.last_captured_frame
                st.warning("Frame rechazado. Captura uno nuevo.")
            else:
                st.info("No hay frame para rechazar")

    with btn_col4:
        if st.button("⏭️ Saltar Validación (usar configuración anterior)", use_container_width=True):
            # Clear validation state and go to step 3 normally
            if "validation_state" in st.session_state:
                del st.session_state.validation_state
            st.session_state["_step_sel"] = "3 "
            st.rerun()

    # Analysis guide
    with st.expander("📖 Guía de Interpretación del Overlay", expanded=True):
        st.markdown("""
        **En el panel de 'Feed Procesado':**
        - 🔴 **Rojo brillante**: Píxeles sobreexpuestos (valor > 250)
          → *Acción*: Reducir exposición o ganancia
        - 🔵 **Azul brillante**: Píxeles subexpuestos (valor < 15)
          → *Acción*: Aumentar exposición o ganancia
        - 🟢 **Verde**: Rango útil de detalle (15-250)
          → *Objetivo*: Maximizar verde, minimizar rojo/azul
        - 📊 **Histograma inferior**: Distribución de luminosidad
          → *Ideal*: Campana centrada sin cortes en extremos
        """)

    # Button to go back to step 3 (camera adjustment) if needed
    if st.button("← Volver a Paso 3: Ajustar Cámara", use_container_width=True):
        # Clear validation state
        if "validation_state" in st.session_state:
            del st.session_state.validation_state
        st.session_state["_step_sel"] = "3 "
        st.rerun()

# ==========================================================================
# paso 3: elegir cámara y ajustar
# ==========================================================================
elif step.startswith("3 ") and not st.session_state.get("_show_validation", False):
    st.title("3 · Elegir la fuente de video y ajustar la cámara")
    st.markdown(
        "Detecta las fuentes activas (webcams USB y NDI), **elige una** y la wizard la fija "
        "en config.json. Después ajusta: blanco **sin rojo** y rayas nítidas en todo el ciclo.")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔎 Detectar TODOS los dispositivos de video (0–7)"):
            release_camera()
            with st.spinner("Probando dispositivos 0–7…"):
                devs = detect_video_devices(8)
            st.session_state["video_devs"] = devs
            st.rerun()
    with c2:
        ips = st.text_input("IPs NDI (opcional, si mDNS falla)", key="ndi_probe_ips")
        if st.button("🔎 Buscar NDI (8 s)"):
            try:
                st.session_state["ndi"] = detect_ndi(ips or None, 8.0)
            except Exception as e:
                st.session_state["ndi"] = [f"(error: {e})"]
            st.rerun()

    devs = st.session_state.get("video_devs", [])
    if devs:
        st.markdown("**Dispositivos de video conectados a la PC**")
        st.dataframe(
            [{"índice": d["index"], "nombre": d.get("name") or "-",
              "backend": d["backend"],
              "resolución": f"{d['width']}x{d['height']}",
              "estado": "con imagen" if d["ok"] else "sin imagen"} for d in devs],
            use_container_width=True, hide_index=True)
        avail = [d for d in devs if d["ok"]] or devs
        labels = [f"[{d['index']}] {d.get('name') or d['backend']} · {d['width']}x{d['height']}"
                  + ("" if d["ok"] else " · SIN IMAGEN") for d in avail]
        sel = st.selectbox("Selecciona el dispositivo que quieras usar", labels,
                           index=None, key="video_dev_sel")
        if sel:
            dev = avail[labels.index(sel)]
            b1, b2 = st.columns(2)
            if b1.button("✅ Usar este dispositivo", type="primary"):
                apply_camera_webcam(dev["index"], dev["backend"])
                st.success(f"config.json → camera.type=webcam, index={dev['index']}, "
                           f"backend={dev['backend']}")
            if b2.button("📸 Ver una captura de prueba"):
                release_camera()
                try:
                    be = cv2.CAP_DSHOW if dev["backend"] == "dshow" else cv2.CAP_ANY
                    cap = cv2.VideoCapture(dev["index"], be)
                    ok, f = cap.read()
                    cap.release()
                    if ok and f is not None:
                        st.session_state["dev_shot"] = f
                    else:
                        st.error(f"El índice {dev['index']} no entrega imagen ahora mismo.")
                except Exception as e:
                    st.error(f"No se pudo abrir: {e}")
        shot = st.session_state.get("dev_shot")
        if shot is not None:
            gvis = framing_overlay(shot) if st.session_state.get("_guide", True) else shot
            st.image(cv2.cvtColor(gvis, cv2.COLOR_BGR2RGB),
                     caption="Captura de prueba del dispositivo elegido (con guía de encuadre: "
                             "la zona iluminada debe llenar el marco ámbar)")
        st.caption("Esta prueba cierra la cámara en caché; si tenías vista en vivo, vuelve a abrirla abajo.")
    else:
        st.caption("Pulsa 'Detectar TODOS los dispositivos' para ver el menú y elegir el disponible.")

    ndi_sources = st.session_state.get("ndi", [])
    if ndi_sources:
        opts_n = [f"NDI: {sname}" for sname in ndi_sources]
        pick_n = st.radio("Fuentes NDI:", opts_n, index=None, key="pick_ndi")
        if pick_n and st.button("✅ Usar esta fuente NDI"):
            apply_camera_ndi(ndi_sources[opts_n.index(pick_n)],
                             st.session_state.get("ndi_probe_ips", ""))
            st.success("config.json → camera.ndi_source actualizado")

    cfg = load_config()
    if cfg is None:
        st.error("No hay config.json: completa el paso 1.")
    elif "camera" not in cfg:
        st.info("⏳ Cámara: **ninguna configurada** — detecta las fuentes de arriba y pulsa "
                "«✅ Usar este dispositivo» o «✅ Usar esta fuente NDI». "
                "La wizard NUNCA elige la cámara sola: la selección es tuya.")
    else:
        cinfo = cfg["camera"]
        if str(cinfo.get("type", "webcam")).lower() == "ndi":
            cam_desc = f"NDI «{cinfo.get('ndi_source', '(primera fuente)')}»"
        else:
            cam_desc = f"webcam índice {cinfo.get('index')} · backend {cinfo.get('backend', '')}"

        def _rm_cam(c):
            c.pop("camera", None)
            return c
        if st.button("✖ Quitar esta cámara de config.json (elegir otra)",
                     help="Siempre reversible: te vuelven a aparecer las fuentes "
                          "detectadas para elegir otra."):
            edit_config(_rm_cam)
            st.success("Cámara quitada de config.json.")
            st.rerun()
        st.success(f"Cámara configurada: {cam_desc} — cambiarla es libre.")
        cam = None
        _err = None
        try:
            cam = get_camera(cinfo)
        except Exception as e:
            _err = e
        if cam is None or _err is not None:
            st.error(f"No se pudo abrir la cámara: {_err}\n\nNDI: ¿la app está emitiendo "
                     "y en la misma red? Firewall: permite python.exe en redes privadas.")
        else:
            is_ndi = str(cinfo.get("type", "webcam")).lower() == "ndi"
            # defaults: los sliders sólo existen en el bloque de webcam, y la vista
            # en vivo de abajo los usa también con NDI (si no, NameError)
            ae_w = bool(cinfo.get("auto_exposure", False))
            ex_w = int(cinfo["exposure"]) if cinfo.get("exposure") is not None else -6
            gn_w = int(cinfo.get("gain") or 0)
            st.success(f"Cámara abierta: {getattr(cam, 'name', 'webcam')} "
                       f"({cam.shape[1]}x{cam.shape[0]}){ ' · NDI' if is_ndi else ''}")
            if is_ndi:
                st.info("Exposición/foco se ajustan EN LA APP del teléfono (bloquea AE/AF, "
                        "estabilización OFF).")
            else:
                with st.expander("☀️ Exposición / brillo de la cámara (webcam)", expanded=True):
                    st.caption(
                        "Reglas: sin ROJO en la pared blanca = sin sobreexposición (si hay >3 % "
                        "de saturados ⇒ baja); blanco por debajo de 60 ⇒ subexpuesta (sube). "
                        "Autoexposición OFF siempre: si se mueve entre fotos, el escaneo se "
                        "arruina. Y si el problema es «el negro sale más claro que el blanco», "
                        "NO es exposición: es LATENCIA → botones de sincronía en el paso 4.")
                    proj_modes = ["Blanco (recomendado)", "Negro", "Patrón de calibración", "Individual"]
                    proj_mode = st.radio("Modo proyector mientras ajustas", proj_modes,
                                        index=0,
                                        help="Selecciona qué muestra el proyector para que la cámara "
                                             "ve un patrón fijo mientras ajustas exposición y ganancia.")
                    _cap = getattr(cam, "cap", None)
                    if _cap is not None:
                        st.write(
                            f"Driver ahora: autoexposición **{_cap.get(cv2.CAP_PROP_AUTO_EXPOSURE):.2f}**"
                            f" · exposición **{_cap.get(cv2.CAP_PROP_EXPOSURE):.1f}** (log2)"
                            f" · ganancia **{_cap.get(cv2.CAP_PROP_GAIN):.0f}**")
                    st.session_state.setdefault("ae_tgl", bool(cinfo.get("auto_exposure", False)))
                    ae_w = st.toggle("Autoexposición (déjala OFF para escanear)", key="ae_tgl")
                    st.session_state.setdefault("ex_slider", int(cinfo.get("exposure", -6)))
                    ex_w = st.slider("Exposición (log2: -13 = 1/8192 s, -1 = ½ s)",
                                     -14, -1, key="ex_slider",
                                     help="Súbela de 1 en 1 hasta que la pared se vea blanca de verdad.")
                    st.session_state.setdefault("gn_slider", int(cinfo.get("gain") or 0))
                    gn_w = st.slider("Ganancia (sube solo si aún oscura: da ruido)",
                                     0, 100, key="gn_slider")
                    if proj_mode == "Blanco (recomendado)":
                        st.info("Los proyectores muestran blanco puro: ajusta exposición hasta que la pared blanca se vea blanca sin rojo (sin saturados).")
                    elif proj_mode == "Negro":
                        st.info("Los proyectores muestran negro: úsalo para comprobar el contraste dinámico.")
                    elif proj_mode == "Patrón de calibración":
                        st.info("Los proyectores muestran la secuencia REAL de escaneo, patrón a patrón "
                                "(actívalo con «👁 Vista en vivo»): si a esta exposición se ven nítidos "
                                "y con contraste, el escaneo va a decodificar bien.")
                    else:
                        st.info("Modo individual: selecciona qué proyector mostrar a continuación.")
                    _sv_ex = int(cinfo["exposure"]) if cinfo.get("exposure") is not None else int(ex_w)
                    _sv_gn = int(cinfo["gain"]) if cinfo.get("gain") is not None else int(gn_w)
                    if (bool(cinfo.get("auto_exposure", False)) != bool(ae_w)
                            or _sv_ex != int(ex_w) or _sv_gn != int(gn_w)):
                        st.warning(
                            f"⚠️ Estos ajustes **aún no están en config.json**: el escaneo (paso 4) "
                            f"usará exposición **{cinfo.get('exposure')}** · ganancia "
                            f"**{cinfo.get('gain')}** · autoexp. **{cinfo.get('auto_exposure', False)}**. "
                            "Pulsa «✅ Aplicar y guardar» para que el escaneo vea exactamente lo que "
                            "tú ves aquí.")
                    bcol1, bcol2 = st.columns(2)
                    if bcol1.button("✅ Aplicar y guardar en config.json",
                                    type="primary", use_container_width=True):
                        def _mut_ex(cfgm):
                            c = cfgm.setdefault("camera", {})
                            c["auto_exposure"] = bool(ae_w)
                            c["exposure"] = int(ex_w)
                            c["gain"] = int(gn_w)
                        edit_config(_mut_ex)
                        release_camera()
                        st.rerun()
                    if bcol2.button("⚙️ Diálogo del driver de la webcam",
                                    use_container_width=True,
                                    help="Abre el panel nativo de Windows de la cámara (si el "
                                         "driver lo permite)."):
                        try:
                            cam.open_driver_settings()
                        except Exception as e:
                            st.error(f"No se pudo abrir el diálogo del driver: {e}")
                    # lo que cambies en el diálogo ⚙️ NO queda en config.json, y el
                    # escaneo (paso 4) vuelve a fijar config.json al abrir la cámara:
                    # así se escanea siempre con la misma exposición pase lo que pase.
                    if hasattr(cam, "driver_state"):
                        _dst = cam.driver_state()
                        _a = _dst["actual"]
                        _cf_ex = cinfo.get("exposure")
                        if _cf_ex is not None and abs(_a["exposure"] - float(_cf_ex)) > 0.5:
                            st.warning(
                                f"El driver está en exposición **{_a['exposure']:.0f}** y ganancia "
                                f"**{_a['gain']:.0f}** (lo que pusiste en su diálogo ⚙️), pero "
                                f"config.json pide **{_cf_ex}**: el escaneo del paso 4 forzará "
                                f"{_cf_ex} al abrir la cámara. Si la del driver es la buena, "
                                "guárdala con el botón de abajo.")
                        # la ganancia sólo se guarda si el driver la soporta (si no, lee -1)
                        _g_drv = int(round(_a["gain"])) if _a["gain"] >= 0 else None
                        if st.button(
                                "📥 Guardar en config.json lo del driver "
                                f"({_a['exposure']:.0f}"
                                + (f" / ganancia {_g_drv}" if _g_drv is not None else "") + ")",
                                use_container_width=True,
                                help="Escribe en config.json la exposición y ganancia que "
                                     "tiene AHORA el driver, para que el escaneo use "
                                     "exactamente lo que ves en la vista en vivo."):
                            def _mut_drv(cfgm, _e=int(round(_a["exposure"])), _g=_g_drv):
                                c = cfgm.setdefault("camera", {})
                                c["auto_exposure"] = False
                                c["exposure"] = _e
                                if _g is not None:
                                    c["gain"] = _g
                            edit_config(_mut_drv)
                            release_camera()
                            st.rerun()
            with st.expander("📐 Guía de posicionamiento de la cámara (léela antes de escanear)"):
                st.markdown(
                    "1. **Cámara centrada con el ciclorama y lo más lejos posible** (usa zoom, "
                    "no gran angular): la pared debe llenar el encuadre.\n"
                    "2. El **marco ámbar** es el margen del 5 %: si la pared lo toca o se sale, "
                    "recoloca la cámara.\n"
                    "3. La **caja verde** es lo que la cámara ve iluminado (con los proyectores "
                    "en blanco): debe cubrir **~90 % del encuadre** y quedarse DENTRO del "
                    "marco ámbar.\n"
                    "4. Fija el trípode y NO la vuelvas a tocar: el escaneo necesita "
                    "EXACTAMENTE ese encuadre.")
            st.session_state.setdefault("_guide", True)
            guide = st.toggle("📐 Guía de encuadre sobre la vista en vivo", key="_guide")
            live = st.toggle("👁 Vista en vivo")
            ph = st.empty()
            met = st.empty()
            last = None

            # Vista de proyector en tiempo real (config NORMALIZADA: ver projector_canvas)
            cfg_current = load_config()
            cv_, _norm_cfg, _cerr = projector_canvas(cfg_current)
            if cv_ is None:
                st.warning(f"No se pudo abrir la vista de proyector: {_cerr}")
            _pat = pattern_cycle(_norm_cfg) if _norm_cfg is not None else None

            def update_projs(mode_str, step_pat=None):
                """Pinta los proyectores; step_pat = {(w,h): (nombre, imagen)} o None."""
                if cv_ is None: return
                for i, p in enumerate(_norm_cfg["projectors"]):
                    w, h = int(p["width"]), int(p["height"])
                    if "Blanco" in mode_str:
                        img_p = np.full((h, w), 255, np.uint8)
                    elif "Negro" in mode_str:
                        img_p = np.zeros((h, w), np.uint8)
                    elif "Patrón" in mode_str and step_pat is not None:
                        _kv = step_pat.get((w, h))
                        img_p = _kv[1] if _kv is not None else np.zeros((h, w), np.uint8)
                    else:
                        img_p = np.zeros((h, w), np.uint8)
                    try:
                        cv_.set(i, img_p)
                    except Exception: pass
                try:
                    cv_.flush(1)
                except Exception: pass

            if live:
                # Lo que ves aquí es EXACTAMENTE lo que capturará el escaneo: sin brillo
                # digital y con exposición/ganancia aplicadas al driver YA, en vivo.
                if not is_ndi and not ae_w:
                    _drv = getattr(cam, "cap", None)
                    if _drv is not None:
                        try:
                            _drv.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
                            _drv.set(cv2.CAP_PROP_EXPOSURE, int(ex_w))
                            _drv.set(cv2.CAP_PROP_GAIN, int(gn_w))
                        except Exception:
                            pass
                update_projs(proj_mode, next(_pat) if ("Patrón" in proj_mode and _pat) else None)
                for k in range(6000):
                    try:
                        f = cam.read_bgr()
                    except Exception as e:
                        ph.error(f"falla de captura: {e}")
                        break


                    last = f
                    if k % 5 == 0:
                        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                        sat = float((g >= 250).mean()) * 100
                        mean = float(g.mean())
                        tip = ("🔴 Sobreexpuesta: BAJA la exposición" if sat > 3 else
                               ("⚪ Subexpuesta: SUBE la exposición" if mean < 60 else "✔ Brillo OK"))
                        _d = getattr(cam, "cap", None)
                        _drv_s = (f" · exposición **{_d.get(cv2.CAP_PROP_EXPOSURE):.1f}**"
                                  f" · ganancia **{_d.get(cv2.CAP_PROP_GAIN):.0f}**"
                                  f" · autoexp. **{_d.get(cv2.CAP_PROP_AUTO_EXPOSURE):.2f}**"
                                  if (_d is not None and not is_ndi) else "")
                        met.caption(f"☀️ Luminancia media **{mean:.0f}**/255 · saturados **{sat:.2f} %** — {tip}"
                                    f"{_drv_s} · config.json: exposición {ex_w}, ganancia {gn_w}")

                    if "Patrón" in proj_mode and _pat is not None and k % 6 == 0:
                        update_projs(proj_mode, next(_pat))

                    vis = framing_overlay(f) if guide else f
                    ph.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), use_container_width=True)
                    time.sleep(1 / 25)
                if cv_ is not None:
                    cv_.close()
            elif cv_ is not None:
                cv_.close()          # sin vista en vivo: no dejes ventanas tapando la pared
            if st.button("📸 Tomar foto fijada para marcar bordes") and not live:
                try:
                    st.session_state["photo"] = cam.read_bgr()
                    st.success("Foto guardada para el paso 5 (mejor con proyectores en blanco).")
                except Exception as e:
                    st.error(str(e))
            if st.button("🖐 Liberar cámara (antes de escanear)"):
                release_camera()
                st.success("Cámara liberada: ya puedes escanear en el paso 4.")

# ==========================================================================
# paso 4: escanear
# ==========================================================================
elif step.startswith("4 "):
    st.title("4 · Escanear los proyectores")
    st.markdown(
        "46 patrones por proyector (~2 min con 3). **No muevas la cámara** ni cambies la "
        "exposición durante el escaneo.\n\n⚠️ Libera la vista previa del paso 3 antes de "
        "escanear (la cámara no puede la usan la wizard y el escaneo a la vez).\n\n"
        "**¿La superficie es más grande que el encuadre?** Usa «🎯 Escaneo por proyector» "
        "(abajo): escaneas uno, el escaneo se detiene solo, mueves la cámara al siguiente y "
        "sigues. Al final se calcula un único lienzo para TouchDesigner.")
    if st.session_state.pop("_clean_done", False):
        st.session_state["clean_yes"] = False
    _mc = st.session_state.pop("_msg_clean", None)
    if _mc:
        st.success(_mc)
    _cfg = load_config() or {}
    keep_raw = st.checkbox(
        "Guardar TODAS las capturas del escaneo (raw_projN/, ~46 por proyector)",
        value=bool(_cfg.get("scan", {}).get("save_raw", False)),
        help="Así puedes revisar cada patrón capturado, no sólo el blanco/negro. También se "
             "genera captures_projN.png con todas las capturas de ese proyector de un vistazo.")
    feed_on = st.checkbox(
        "🎬 Ver el feed de la cámara mientras escanea (no escanear a ciegas)",
        value=True,
        help="Durante el escaneo se muestran aquí las capturas de la cámara al momento. "
             "Comprueba que la cámara NO se mueve, que ves los patrones proyectados y "
             "que nada bloquea la vista. Requiere guardar capturas temporalmente; si no "
             "marcaste «Guardar TODAS», se borran al terminar el escaneo.")
    _cam_sync = str(((_cfg.get("camera") or {}).get("sync")) or "").lower() or "fixed"
    st.markdown(
        "**Sincronía (latencia patrón → cámara):** si el diagnóstico dice que el negro sale "
        "más claro que el blanco, las capturas llegan cuando el proyector YA cambió. "
        "No es exposición: usa uno de los dos botones.")
    sc1, sc2 = st.columns(2)
    with sc1:
        if st.button("🔄 Activar sync auto (recomendado)", use_container_width=True,
                     help="Espera a ver el cambio de patrón y a que la imagen se quede quieta "
                          "antes de promediar. Sirve con webcam USB y con NDI."):
            def _mut_sync(cfgm):
                cfgm.setdefault("camera", {})["sync"] = "auto"
            edit_config(_mut_sync)
            st.success("camera.sync = auto ✔")
            st.rerun()
    with sc2:
        if st.button("⏱ Espera fija de 900 ms (alternativa)", use_container_width=True,
                     help="Sube settle_ms a 900 sin sync auto. Escanea más lento "
                          "(~7 s extra por proyector) pero tolera más latencia."):
            def _mut_settle(cfgm):
                c = cfgm.setdefault("camera", {})
                c["sync"] = ""
                c["settle_ms"] = max(int(c.get("settle_ms") or 0), 900)
            edit_config(_mut_settle)
            st.success("camera.settle_ms ≥ 900 ✔")
            st.rerun()
    st.caption(f"Estado actual: sync = **{_cam_sync}** · settle_ms = "
               f"**{(_cfg.get('camera') or {}).get('settle_ms')}** · flush_frames = "
               f"**{(_cfg.get('camera') or {}).get('flush_frames')}**.")

    aw_check = WORK / "all_white.png"
    if aw_check.exists():
        with st.expander("📋 Chequeo de encuadre del último all_white.png"):
            _vis, _verdict = coverage_verdict(aw_check)
            if _vis is not None:
                st.image(cv2.cvtColor(_vis, cv2.COLOR_BGR2RGB), use_container_width=True)
            (st.success if _verdict.startswith("✅") else st.warning)(_verdict)
    c1, c2 = st.columns(2)
    with c2:
        st.caption("Esto escanea todos los proyectores seguidos, sin mover la cámara: sirve sólo "
                   "si la cámara ve TODA la superficie de una vez. Si es más grande, usa el "
                   "escaneo por proyector de abajo.")
    with c1:
        if st.button("▶ Escanear de una vez (la cámara ve todo)", use_container_width=True):
            release_camera()
            if bool((load_config() or {}).get("scan", {}).get("save_raw", False)) != (keep_raw or feed_on):
                apply_scan_save_raw(keep_raw or feed_on)
            status_ph = st.empty()
            status_ph.markdown("⏳ Esperando la primera captura…")
            feed_ph = st.empty()
            feed_state = {"t": 0.0}

            def _feed(line: str) -> None:
                if not feed_on:
                    return
                ln = line.strip()
                if ln.startswith("proyector"):
                    status_ph.markdown(f"🎬 **{ln}**")
                f, t = newest_raw_capture(WORK)
                if f and t > feed_state["t"]:
                    feed_state["t"] = t
                    img = cv2.imread(str(f), cv2.IMREAD_REDUCED_COLOR_2)
                    if img is not None:
                        feed_ph.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                                      caption=f"🎥 {f.parent.name}/{f.name}",
                                      use_container_width=True)

            ok = calibrate_cli(["scan", "-c", "config.json"], key_prefix="scan",
                               on_line=_feed) == 0
            reset_cache()
            if feed_on and not keep_raw:
                for d in WORK.glob("raw_proj*"):
                    shutil.rmtree(d, ignore_errors=True)
            if ok:
                st.success("Escaneo listo: revisa las capturas y continúa al paso 5.")
            else:
                st.error("Escaneo fallido: lee la salida y usa el diagnóstico.")

    # === ESCANEO POR PROYECTOR: pausas para mover la cámara ===
    _pp_cfg = load_config() or {}
    _pp_projs = _pp_cfg.get("projectors", []) or []
    if _pp_projs:
        from lw.pipeline import scan_overlaps
        st.divider()
        st.subheader("🎯 Escaneo por proyector (superficies grandes)")
        st.markdown(
            "En cada posición la cámara debe ver **ese proyector completo** (y, si puedes, un trozo "
            "de los vecinos: así se mide el solape real y la costura queda clavada). "
            "Entre un proyector y el siguiente **el escaneo queda en pausa** esperando a que tú "
            "muevas la cámara y confirmes que está colocada. No cambies la exposición entre "
            "posiciones: si cada pose sale con distinto brillo, la uniformidad lo compensa a "
            "medias y el resultado pierde calidad.")
        _pp_pause = st.session_state.pop("_scan_pause", None)
        if _pp_pause is not None:
            for _k in [k for k in list(st.session_state.keys())
                       if str(k).startswith("scan_ready_")]:
                st.session_state[_k] = False      # hay que re-confirmar la cámara
            _next_n = _pp_pause + 1
            st.success(f"⏸ **Pausa** — proyector {_pp_pause} escaneado. Mueve la cámara al "
                       f"proyector {_next_n}, encuádralo completo y confirma abajo.")
        # all_white_proj{i}.png sólo lo escribe el escaneo POR PROYECTOR (scan_proj{i}.npz
        # también existe tras un escaneo de cámara única, no sirve como marca del modo)
        _pp_done = [(WORK / f"all_white_proj{i}.png").exists() for i in range(len(_pp_projs))]
        _pp_nxt = next((i for i, d in enumerate(_pp_done) if not d), None)
        # escaneos repetidos: si el nuevo captó la misma exposición y el mismo brillo
        # que el anterior, las imágenes NO han cambiado (típico si ajustaste la
        # exposición en el diálogo ⚙️ del driver: config.json la vuelve a fijar)
        from lw.pipeline import repeat_warning, scan_history
        _hist4 = scan_history(WORK)
        for _i in range(len(_pp_projs)):
            _h4 = _hist4.get(str(_i)) or []
            if len(_h4) >= 2:
                _w4 = repeat_warning(_h4[-2], _h4[-1], _i)
                if _w4:
                    st.warning(_w4)
        _pp_cols = st.columns(len(_pp_projs))
        # Asegurar importaciones al inicio de la función
        from lw.pipeline import scan_overlaps, scan_projector

        # En la parte del bucle de proyectores
        for _i, (_c, _p) in enumerate(zip(_pp_cols, _pp_projs)):
            with _c:
                _ov = scan_overlaps(WORK, _i)
                _ovs = " · ".join(f"{k.replace('overlap_', '')} {v*100:.0f}%"
                                 for k, v in _ov.items() if v == v and v > 0)
                st.markdown(("✅" if _pp_done[_i] else "⬜") + f" **P{_i}** {_p.get('name', '')}"
                            + (f"\n\n{_ovs}" if _ovs else ""))
        st.markdown("**Siguiente:** " + (f"proyector {_pp_nxt}" if _pp_nxt is not None
                                          else "todos escaneados ✅ → paso 5"))
        for _i, _p in enumerate(_pp_projs):
            _ov = scan_overlaps(WORK, _i)
            _ovs = ", ".join(f"{k.replace('overlap_', '')}={v*100:.1f}%"
                             for k, v in _ov.items() if v == v and v > 0)
            with st.expander(("✅ " if _pp_done[_i] else "⬜ ") + f"Proyector {_i} — {_p.get('name', '')}"
                             + (f"  ·  solape {_ovs}" if _ovs else ""),
                             expanded=(_i == _pp_nxt)):
                _cap_i = scan_caption(WORK, _i)
                if _cap_i:
                    st.caption("🎞️ " + _cap_i)
                _ready = st.checkbox(
                    f"Cámara encuadrada sobre el proyector {_i} (no la moveré hasta que acabe)",
                    key=f"scan_ready_{_i}")
                if st.button(("🔁 Volver a escanear" if _pp_done[_i] else "▶ Escanear") + f" proyector {_i}",
                             key=f"scan_go_{_i}", use_container_width=True, disabled=not _ready,
                             type=("primary" if _i == _pp_nxt else "secondary")):
                    st.session_state[f"scan_prev_{_i}"] = False   # la vista previa ocupa la cámara
                    close_preview_canvas()
                    release_camera()
                    if bool((load_config() or {}).get("scan", {}).get("save_raw", False)) != (keep_raw or feed_on):
                        apply_scan_save_raw(keep_raw or feed_on)
                    _ph = st.empty()
                    _ph.markdown(f"⏳ Escaneando el proyector {_i}…")
                    _fph = st.empty()
                    _fst = {"t": 0.0}

                    def _feed_one(line: str, _ii=_i, _pph=_ph, _ffph=_fph, _sst=_fst) -> None:
                        ln = line.strip()
                        if ln.startswith("proyector") or "solape" in ln:
                            _pph.markdown(f"🎬 **{ln}**")
                        if not feed_on:
                            return
                        f, t = newest_raw_capture(WORK / f"raw_proj{_ii}")
                        if f and t > _sst["t"]:
                            _sst["t"] = t
                            img = cv2.imread(str(f), cv2.IMREAD_REDUCED_COLOR_2)
                            if img is not None:
                                _ffph.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                                            caption=f"🎥 {f.name}", use_container_width=True)

                    _ok = calibrate_cli(["scan", "-c", "config.json", "--projector", str(_i)],
                                        key_prefix=f"scan_p{_i}", on_line=_feed_one) == 0
                    reset_cache()
                    if feed_on and not keep_raw:
                        shutil.rmtree(WORK / f"raw_proj{_i}", ignore_errors=True)
                    if _ok:
                        st.session_state["_scan_pause"] = _i
                        st.rerun()
                    else:
                        st.error("Escaneo fallido: lee la salida y usa el diagnóstico.")

                _pv_on = st.toggle(
                    "👁 Vista previa: ver el área a escanear antes de escanear",
                    key=f"scan_prev_{_i}",
                    help="Cámara en vivo con la zona iluminada y el marco del 5% sobre la pared: "
                         "comprueba que ESTE proyector se ve COMPLETO dentro del encuadre y que la "
                         "exposición sigue siendo la del paso 3. Con «Patrones del escaneo» verás "
                         "los patrones reales proyectados mientras ajustas. En los modos en blanco "
                         "el proyector dibuja una rampa en los bordes de su frame: proyectada, esa "
                         "franja cae en la pared donde va el blend con el vecino (su ancho es el "
                         "solape de config.json), así que marcan el borde de solape de este "
                         "proyector sin depender de dónde esté la cámara. Al escanear se apaga sola.")
                if not _pv_on:
                    close_preview_canvas()
                else:
                    _pm = st.radio("Qué muestran los proyectores mientras miras",
                                   ["Blanco (todos)", f"Solo el proyector {_i} en blanco",
                                    "Patrones del escaneo"],
                                   key=f"scan_prev_mode_{_i}", horizontal=True)
                    _ccfg = load_config() or {}
                    if "_prev_cv" not in st.session_state:
                        _pcv, _pnorm, _pe = projector_canvas(_ccfg)
                        if _pcv is None:
                            st.warning(f"Vista de proyector no disponible: {_pe}")
                        else:
                            st.session_state["_prev_cv"] = (_pcv, _pnorm)
                    if "_prev_cv" in st.session_state:
                        _pcv, _pnorm = st.session_state["_prev_cv"]

                        def _paint4(_mk, _cv4=_pcv, _pn4=_pnorm, _only4=None):
                            for _j, _pj in enumerate(_pn4["projectors"]):
                                _w4, _h4 = int(_pj["width"]), int(_pj["height"])
                                _img4 = (np.zeros((_h4, _w4), np.uint8)
                                         if (_only4 is not None and _j != _only4)
                                         else _mk(_w4, _h4, _j))
                                try:
                                    _cv4.set(_j, _img4)
                                except Exception:
                                    pass
                            try:
                                _cv4.flush(1)
                            except Exception:
                                pass

                        # blanco con la rampa de solape dibujada en los bordes del
                        # proyector: proyectada, la franja cae en la pared justo donde
                        # el calibrador va a hacer el blend con el vecino
                        _ovl4 = float((_ccfg.get("blend") or {}).get("overlap_frac", 0.15) or 0.15)
                        _ovl4 = min(max(_ovl4, 0.02), 0.45)
                        _npj4 = len(_pnorm["projectors"])

                        def _guia4(_w, _h, _j, _f=_ovl4, _n=_npj4):
                            return blend_band_image(_w, _h,
                                                    _f if _j > 0 else 0.0,
                                                    _f if _j < _n - 1 else 0.0)

                        if "Patrones" not in _pm:
                            _paint4(_guia4, _only4=(_i if "Blanco" not in _pm else None))
                        _pat4 = pattern_cycle(_pnorm) if "Patrones" in _pm else None
                        try:
                            cam4 = get_camera(_ccfg.get("camera") or {})
                        except Exception as _ce:
                            cam4 = None
                            st.error(f"No se pudo abrir la cámara: {_ce}")
                        if cam4 is not None:
                            _ph4, _mt4 = st.empty(), st.empty()
                            _bnd4 = " · ".join(_x for _x in (
                                (f"franja izquierda: solape con P{_i - 1}" if _i > 0 else ""),
                                (f"franja derecha: solape con P{_i + 1}"
                                 if _i + 1 < _npj4 else "")) if _x)
                            _slp4 = seam_overlap_text(_i, _npj4)
                            for _k4 in range(900):        # ~45 s; se corta con cualquier rerun
                                if _pat4 is not None and _k4 % 6 == 0:
                                    _step = next(_pat4)
                                    for _j, _pj in enumerate(_pnorm["projectors"]):
                                        _kv = _step.get((int(_pj["width"]), int(_pj["height"])))
                                        if _kv is not None:
                                            try:
                                                _pcv.set(_j, _kv[1])
                                            except Exception:
                                                pass
                                    try:
                                        _pcv.flush(1)
                                    except Exception:
                                        pass
                                try:
                                    _f4 = cam4.read_bgr()
                                except Exception as _e4:
                                    st.error(f"falla de captura: {_e4}")
                                    break
                                _ph4.image(cv2.cvtColor(framing_overlay(_f4),
                                                        cv2.COLOR_BGR2RGB),
                                           caption=("🎥 Cámara en vivo — proyector "
                                                    f"{_i} · " + ("todos en blanco" if "Blanco" in _pm
                                                    else ("patrones de escaneo" if "Patrones" in _pm
                                                    else f"sólo el proyector {_i} en blanco"))
                                                    + ((f" · en la pared: {_bnd4} "
                                                        f"({100 * _ovl4:.0f}% del ancho, rampa a oscuro)")
                                                       if _bnd4 and "Patrones" not in _pm else "")
                                                    + (f" · solape medido: {_slp4}" if _slp4 else "")),
                                           use_container_width=True)
                                if _k4 % 6 == 0:
                                    _d4 = getattr(cam4, "cap", None)
                                    _drv4 = (f" · exposición **{_d4.get(cv2.CAP_PROP_EXPOSURE):.1f}**"
                                             f" · ganancia **{_d4.get(cv2.CAP_PROP_GAIN):.0f}**"
                                             f" · autoexp. **{_d4.get(cv2.CAP_PROP_AUTO_EXPOSURE):.2f}**"
                                             if _d4 is not None else "")
                                    _mt4.caption(
                                        f"{framing_verdict_text(cv2.cvtColor(_f4, cv2.COLOR_BGR2GRAY))}"
                                        f"{_drv4} · config.json: exposición "
                                        f"{(_ccfg.get('camera') or {}).get('exposure')}, ganancia "
                                        f"{(_ccfg.get('camera') or {}).get('gain')}")
                                time.sleep(1 / 20)

    # Mostrar información de escaneos individuales
    scan_proj_files = sorted(WORK.glob("scan_proj*.npz"))
    if scan_proj_files:
        st.markdown("---")
        st.subheader("Escaneos individuales:")
        for f in scan_proj_files:
            st.write(f"• {f.name}")

    from lw.clean import clean, count_files, map_files, scan_files, size_mb
    with st.expander("🧹 Borrar escaneos viejos (empezar de cero)"):
        _arts = scan_files(WORK, OUT)
        _maps = map_files(OUT)
        st.markdown(f"- `work/`: **{count_files(_arts)}** archivos de escaneo, "
                    f"**{size_mb(_arts):.0f} MB**\n"
                    f"- `output/`: **{len(_maps)}** archivos de mapas")
        st.caption("Borra scan_proj*.npz, decoded_*, preflight_*, captures_*, raw_proj*/, "
                   "all_white.png, all_white_proj*.png y overlap_proj*. No toca config.json "
                   "ni nada que no sea del escaneo.")
        _ce = st.checkbox("Borrar también los bordes marcados (`screen.json` / `screen_proj*.json`)",
                          key="clean_edges")
        _cm = st.checkbox("Borrar también los mapas de `output/`", key="clean_maps")
        _cy = st.checkbox("Sí, borrar (no se puede deshacer)", key="clean_yes")
        if st.button("🗑 Borrar ahora", type="primary", disabled=not _cy, key="clean_go"):
            _n, _mb = clean(WORK, OUT, edges=_ce, maps=_cm)
            reset_cache()
            st.session_state["_msg_clean"] = (
                f"Borrados {_n} archivos ({_mb:.0f} MB)." if _n
                else "No había nada que borrar.")
            st.session_state["_clean_done"] = True
            st.rerun()
    if st.button("🩺 Diagnosticar el último escaneo"):
        calibrate_cli(["diagnose", "-c", "config.json"], key_prefix="diag")
    diag_img = WORK / "diagnostico.png"
    if diag_img.exists():
        st.image(str(diag_img), caption="work/diagnostico.png — negro | blanco | dif. | decodificado")

    sheets = sorted(WORK.glob("captures_proj*.png"))
    if sheets:
        st.markdown("### Todas las capturas del escaneo")
        st.caption("Una hoja por proyector: se ven todos los patrones y qué zona captó mal la cámara.")
        for p in sheets:
            st.image(str(p), caption=f"{p.name} — todas las capturas de ese proyector")
        with st.expander("Ver una captura individual"):
            projs = sorted(WORK.glob("raw_proj*"))
            if projs:
                pj = st.selectbox("Proyector", [q.name for q in projs], key="raw_proj_sel")
                names = [f.stem for f in sorted((WORK / pj).glob("*.png"))]
                if names:
                    nm = st.selectbox("Patrón", names, key="raw_name_sel")
                    st.image(str(WORK / pj / f"{nm}.png"), caption=f"{pj}/{nm}.png")
    else:
        st.caption("Activa 'Guardar TODAS las capturas' y escanea para ver las imágenes aquí.")

    pre = sorted(WORK.glob("preflight_proj*.png"))
    if pre:
        st.markdown("**Preflight (negro / blanco)**")
        for p in pre:
            st.image(str(p), caption=p.name)
    aw = WORK / "all_white.png"
    if aw.exists():
        st.image(str(aw), caption="all_white.png — todos los proyectores en blanco")
    for p in sorted(WORK.glob("decoded_proj*.png")):
        st.image(str(p), caption=f"{p.name} — 'rayas mezcladas' ⇒ sube settle_ms / sync auto")

# ==========================================================================
# paso 5: marcar bordes con clics + lupa (alta precisión)
# ==========================================================================
elif step.startswith("5 "):
    st.title("5 · Marcar los bordes (alta precisión)")
    st.markdown(
        "Se marcan los cuatro bordes de la **zona iluminada** por el proyector, en orden: "
        "arriba → derecha → abajo → izquierda.\n\n"
        "1. **Vista general**: clic donde termina la zona → añade un punto y abre la lupa.\n"
        "2. **Lupa**: clic en ella para afinar; **cada cuadro de la rejilla es 1 píxel de cámara** "
        "(a 8× o más).\n"
        "3. Repite a lo largo del borde: **los puntos que quieras**, incluyendo las 2 esquinas "
        "(más puntos = curva más fiel; no hace falta que estén equiespaciados).\n"
        "4. **Enter** confirma el borde y salta al siguiente.\n"
        "5. 💾 guarda los bordes de ese proyector → cuando estén todos, paso 6.\n\n"
        "Para corregir un punto: ◀ ▶ para elegirlo, flechas para moverlo en pasos de 0.05 px, "
        "o **🧲** para pegarlo al gradiente real del borde (precisión sub-píxel).")
    if not ss["scanned"]:
        st.warning("Primero escanea (paso 4).")
    elif not HAS_CLICKS:
        st.error("Falta el paquete de clics: `.venv\\Scripts\\pip.exe install "
                 "streamlit-image-coordinates` y recarga.")
    else:
        _cfg5 = load_config() or {}
        _projs5 = _cfg5.get("projectors", []) or []
        _have5 = {i: (WORK / f"all_white_proj{i}.png").exists() for i in range(len(_projs5))}
        _per5 = bool(_projs5) and any(_have5.values())
        if _per5:
            _done5 = [(WORK / f"screen_proj{i}.json").exists() for i in range(len(_projs5))]
            _c5 = st.columns(len(_projs5))
            for _i, (_cc, _p5) in enumerate(zip(_c5, _projs5)):
                with _cc:
                    st.markdown(("✅" if _done5[_i] else "⬜") + f" **P{_i}** {_p5.get('name', '')}"
                                + ("\n\nbordes guardados" if _done5[_i] else ""))
            _pend5 = [i for i, d in enumerate(_done5) if not d and _have5[i]]
            _sel5 = st.selectbox(
                "Proyector cuyos bordes vas a marcar", list(range(len(_projs5))),
                index=(_pend5[0] if _pend5 else 0),
                format_func=lambda i: f"{i} · {_projs5[i].get('name', '')} "
                                      + ("(ya marcado)" if _done5[i] else "(pendiente)"),
                key="edge_proj_sel5")
            st.session_state["_edge_proj"] = int(_sel5)
            st.info(f"La cámara debe seguir exactamente donde estaba cuando escaneaste el proyector "
                    f"{int(_sel5)}. Si ya la moviste, vuelve al paso 4 y reescánalo.")
            _bg_path = WORK / f"all_white_proj{int(_sel5)}.png"
            img_raw = (cv2.imread(str(_bg_path), cv2.IMREAD_GRAYSCALE)
                       if _have5.get(int(_sel5)) else None)
            save_path = WORK / f"screen_proj{int(_sel5)}.json"
        else:
            st.session_state["_edge_proj"] = None
            save_path = WORK / "screen.json"
            src = st.radio("Imagen de fondo", ["work/all_white.png (recomendada)", "foto del paso 3"],
                           captions=["capturada en el escaneo", "tomada en el paso 3"], key="edge_src")
            if src.startswith("work") and (WORK / "all_white.png").exists():
                _bg_path = WORK / "all_white.png"
                img_raw = cv2.imread(str(_bg_path), cv2.IMREAD_GRAYSCALE)
            elif st.session_state.get("photo") is not None:
                _bg_path = None            # foto guardada en sesión: no hay archivo
                img_raw = cv2.cvtColor(st.session_state["photo"], cv2.COLOR_BGR2GRAY)
            else:
                _bg_path = None
                img_raw = None
        _bg_name = (_bg_path.name if _bg_path is not None else "foto del paso 3")
        _bg_stamp = file_stamp(_bg_path)
        _P5 = st.session_state.get("_edge_proj")
        _S5 = "" if _P5 is None else f"_{int(_P5)}"

        def K(name: str, _suf: str = _S5) -> str:
            """Clave de sesión única por proyector (en modo de un solo lienzo: igual)."""
            return f"{name}{_suf}"

        # --- frescura de la imagen de fondo --------------------------------
        # Al re-escanear un proyector se reescribe work/all_white_proj{i}.png.
        # Si su huella cambia, la imagen es distinta: hay que redibujar la vista
        # y descartar los puntos marcados sobre la imagen anterior.
        _stk = f"_bgstamp{_S5}"
        _prev_stamp = st.session_state.get(_stk)
        _bg_changed = _prev_stamp is not None and _bg_stamp != _prev_stamp
        st.session_state[_stk] = _bg_stamp
        _stag = "none" if not _bg_stamp else f"{_bg_stamp[0]}_{_bg_stamp[1]}"

        def CK(name: str) -> str:
            """Clave del componente de clics, con la huella de la imagen dentro.

            Un widget de componente con la misma key puede seguir mostrando su
            imagen anterior (Streamlit reutiliza la instancia); cambiando la key
            cuando cambia el archivo, la vista se redibuja con la imagen nueva.
            """
            return f"{K(name)}_{_stag}"

        _ek = K("edges")
        if _ek not in st.session_state:
            _e5 = {k: [] for k in EDGES_ORDER}
            # sólo se reutilizan los puntos si se marcaron sobre ESTA imagen
            if save_path.exists() and points_match_scan(save_path, _bg_path):
                try:
                    from lw.screen import load_edges
                    _ee, _ = load_edges(save_path)
                    for _k5 in EDGES_ORDER:
                        _e5[_k5] = [[float(x), float(y)] for x, y in _ee.get(_k5, [])]
                except Exception:
                    pass
            st.session_state[_ek] = _e5
        if _bg_changed:
            # escaneo nuevo con el wizard abierto: los puntos viejos no valen
            st.session_state[_ek] = {k: [] for k in EDGES_ORDER}
            st.session_state[K("_fine")] = {}
            st.session_state[K("_edge_i")] = 0
            st.session_state[K("_msg")] = (
                True, f"Imagen nueva de `{_bg_name}` ({stamp_label(_bg_stamp)}): "
                      "los puntos del escaneo anterior se descartaron. Marca los bordes otra vez.")
        edges = st.session_state[_ek]
        for k in EDGES_ORDER:
            edges.setdefault(k, [])
        st.session_state.setdefault(K("_edge_i"), 0)
        st.session_state.setdefault(K("_fine"), {})
        st.session_state.setdefault(K("_zoom_sel"), 8)
        st.session_state.setdefault(K("edge_mode_sel"), "arclength")

        msg = st.session_state.pop(K("_msg"), None)
        if msg:
            (st.success if msg[0] else st.error)(msg[1])
        if (img_raw is not None and save_path.exists() and _bg_path is not None
                and not points_match_scan(save_path, _bg_path)):
            st.warning(f"`{save_path.name}` se guardó **antes** de este escaneo "
                       f"(`{_bg_name}`, {stamp_label(_bg_stamp)}), así que sus puntos no "
                       "corresponden a esta imagen y se ignoran. Vuelve a marcar los bordes; "
                       "si querías conservarlos, re-escanea con la cámara en la pose en que los marcaste.")

        if img_raw is None:
            st.warning("No hay imagen: escanea primero ese proyector (paso 4).")
        else:
            H, W = img_raw.shape[:2]
            img_bgr = _bgr(img_raw)
            DW = 1100

            # --- fantasma del borde del vecino que se solapa ---------------
            # Se usa la captura 'overlap_projI_vecinoJ.png': el vecino encendido
            # en ESTA misma pose, así su filo cae donde está el solape real.
            img_over, img_zoom = img_bgr, img_bgr
            _ghosts5 = []
            if _P5 is not None:
                from lw.screen import draw_ghosts, ghost_legend, neighbor_ghosts
                _ghosts5 = neighbor_ghosts(
                    WORK, int(_P5), len(_projs5),
                    float((_cfg5.get("scan") or {}).get("white_threshold") or 25))
                _g_ok = [g for g in _ghosts5 if g["x"] is not None]
                if _ghosts5:
                    st.caption(f"👻 Solape con los vecinos: {ghost_legend(_ghosts5)}")
                    if any(g["stale"] for g in _ghosts5):
                        st.warning("Algún vecino no tiene un borde fiable (captura repetida o "
                                   "borde fuera del encuadre): repite el escaneo de este "
                                   "proyector (paso 4) para verlo bien.")
                    if _g_ok:
                        _g_on = st.checkbox(
                            "👻 Ver el borde del vecino (solape)", value=True,
                            key=K("ghost_on"),
                            help="Fantasma del borde del proyector vecino que se solapa: el que "
                                 "queda más cerca de éste, capturado en la misma pose de cámara.")
                        if _g_on:
                            img_over = draw_ghosts(img_bgr, _ghosts5, labels=True)
                            img_zoom = draw_ghosts(img_bgr, _ghosts5, labels=False)
            if "_zc" not in st.session_state:
                st.session_state["_zc"] = [W / 2.0, H / 2.0]

            def _confirm_edge():
                i = int(st.session_state.get(K("_edge_i"), 0))
                k = EDGES_ORDER[i]
                pts_k = st.session_state[K("edges")][k]
                if len(pts_k) < 2:
                    st.session_state[K("_msg")] = (
                        False, f"Borde {k.upper()}: necesito ≥2 puntos (hay {len(pts_k)}). "
                               "Marca al menos las 2 esquinas de ese lado.")
                    return
                if i < len(EDGES_ORDER) - 1:
                    nxt = EDGES_ORDER[i + 1]
                    st.session_state[K("_edge_i")] = i + 1
                    st.session_state[K("_msg")] = (
                        True, f"Borde {k.upper()} confirmado con {len(pts_k)} puntos → «{nxt}».")
                    return
                missing = [kk for kk in EDGES_ORDER if len(st.session_state[K("edges")][kk]) < 2]
                if missing:
                    st.session_state[K("_edge_i")] = EDGES_ORDER.index(missing[0])
                    st.session_state[K("_msg")] = (
                        False, f"Borde {k.upper()} confirmado, pero faltan ≥2 puntos en: "
                               + ", ".join(missing))
                    return
                st.session_state["_do_save"] = (edges, st.session_state.get(K("edge_mode_sel"), "arclength"),
                                                 str(save_path), K("_msg"))

            cur = EDGES_ORDER[int(st.session_state[K("_edge_i")])]
            for bi, (bcol, bk) in enumerate(zip(st.columns(4), EDGES_ORDER)):
                with bcol:
                    if st.button(f"{bk.capitalize()} · {len(edges[bk])} pts",
                                 key=K(f"edgebtn_{bk}"), use_container_width=True,
                                 type="primary" if bk == cur else "secondary",
                                 help=EDGE_HELP[bk]):
                        st.session_state[K("_edge_i")] = bi
                        st.session_state[K("_fine")][bk] = (len(edges[bk]) - 1) if edges[bk] else None
                        st.rerun()
            cur = EDGES_ORDER[int(st.session_state[K("_edge_i")])]
            fine_map = st.session_state[K("_fine")]
            pts = edges[cur]
            fine = fine_map.get(cur)
            if fine is None and pts:
                fine = len(pts) - 1
            if fine is not None and pts:
                fine = max(0, min(int(fine), len(pts) - 1))
            fine_map[cur] = fine

            st.caption("Puntos: " + " · ".join(f"**{k}**: {len(edges[k])}" for k in EDGES_ORDER)
                       + "  ·  los puntos deben ir **en orden** a lo largo del borde")

            _f1, _f2 = st.columns([3, 1])
            with _f1:
                st.caption(f"🖼️ Fondo: `{_bg_name}` · capturado **{stamp_label(_bg_stamp)}**"
                           + ("" if _bg_path is not None else " (foto guardada en la sesión)"))
            with _f2:
                if st.button("🔄 Recargar imagen", use_container_width=True,
                             help="Vuelve a leer el archivo del disco y reinicia el zoom. Úsalo "
                                  "si reescaneaste este proyector y la vista no se actualizó."):
                    st.session_state.pop(_stk, None)
                    for _kk in [k for k in list(st.session_state.get("_click_seen", {}))
                                if str(k).startswith(K("click_"))]:
                        st.session_state["_click_seen"].pop(_kk, None)
                    reset_cache()
                    st.rerun()

            ov, sx, sy = overview_view(img_over, edges, cur, DW, fine)
            ov_click = click_component(ov, DW, CK("click_overview"))
            if ov_click:
                p = [float(ov_click[0] / sx), float(ov_click[1] / sy)]
                pts.append(p)
                fine_map[cur] = len(pts) - 1
                st.session_state["_zc"] = p
                st.rerun()
            st.caption(f"☝️ Vista general ({W}×{H} px de cámara): clic en el borde de la pantalla "
                       f"para añadir el punto {len(pts) + 1} del borde «{cur}».")

            st.divider()
            cc = st.columns([1.5, 0.55, 0.55, 2.4])
            with cc[0]:
                zoom = st.select_slider("Zoom de la lupa (×)", options=[2, 4, 6, 8, 12, 16, 24],
                                        key=K("_zoom_sel"),
                                        help="A 8× cada cuadro de la rejilla es 1 píxel de cámara")
            if pts:
                with cc[1]:
                    if st.button("◀", use_container_width=True, help="Punto anterior"):
                        fine = (fine - 1) % len(pts)
                        fine_map[cur] = fine
                        st.session_state["_zc"] = list(pts[fine])
                        st.rerun()
                with cc[2]:
                    if st.button("▶", use_container_width=True, help="Punto siguiente"):
                        fine = (fine + 1) % len(pts)
                        fine_map[cur] = fine
                        st.session_state["_zc"] = list(pts[fine])
                        st.rerun()
                cc[3].caption(f"Lupa centrada en el punto **{fine + 1}/{len(pts)}** (×{zoom}). "
                              "Clic = poner ese punto exactamente ahí; la vista no salta, así que "
                              "puedes pulsar varias veces hasta clavarlo.")
            else:
                cc[3].caption("Marca primero un punto en la vista general.")

            zv, ix, iy, cw, ch, dh = zoom_view(img_zoom, edges, cur, DW,
                                               st.session_state["_zc"], zoom, fine)
            zv_click = click_component(zv, DW, CK("click_zoom"))
            if zv_click:
                p = [float(ix + zv_click[0] * cw / DW), float(iy + zv_click[1] * ch / dh)]
                if pts and fine is not None:
                    pts[fine] = p
                else:
                    pts.append(p)
                    fine_map[cur] = len(pts) - 1
                st.rerun()

            if pts and fine is not None:
                with st.expander("Ajuste fino y pegado al borde real", expanded=True):
                    st.caption(f"Punto **{fine + 1}**: x = **{pts[fine][0]:.2f}**, "
                               f"y = **{pts[fine][1]:.2f}** px de cámara")
                    step_val = st.radio("Paso de las flechas (px)", [0.05, 0.25, 0.5, 1.0], index=1,
                                        horizontal=True, key=K("nudge_step"))
                    d = float(step_val)
                    q1, q2, q3, q4, q5 = st.columns(5)
                    if q1.button("⬅", use_container_width=True, help=f"−{d} px en x"):
                        pts[fine][0] -= d
                        st.rerun()
                    if q2.button("⬆", use_container_width=True, help=f"−{d} px en y"):
                        pts[fine][1] -= d
                        st.rerun()
                    if q3.button("⬇", use_container_width=True, help=f"+{d} px en y"):
                        pts[fine][1] += d
                        st.rerun()
                    if q4.button("➡", use_container_width=True, help=f"+{d} px en x"):
                        pts[fine][0] += d
                        st.rerun()
                    if q5.button("🧲 Ajustar borde al filo real", use_container_width=True,
                                 help=f"Pega los puntos interiores de «{cur}» al máximo gradiente "
                                      "(el filo real del blanco). Las 2 esquinas no se tocan. "
                                      "Sólo este borde: los otros no se modifican."):
                        from lw.screen import refine_edges
                        new, skipped = refine_edges(img_raw, {cur: pts})
                        edges[cur] = new[cur]
                        st.session_state[K("_fine")][cur] = (len(pts) - 1) if pts else None
                        st.session_state[K("_msg")] = (
                            True, f"Borde {cur.upper()}: {max(0, len(pts) - 2)} puntos interiores "
                                  f"pegados al filo real (sub-píxel); {skipped} sin borde claro.")
                        st.rerun()

            st.divider()
            with st.form(K("adv_form"), clear_on_submit=True):
                t1, t2 = st.columns([3, 1])
                t1.text_input("⏎ Pulsa **Enter** para confirmar este borde y pasar al siguiente",
                              key=K("adv_txt"),
                              placeholder=f"borde «{cur}»: {len(pts)} puntos")
                t2.form_submit_button("⏎ Confirmar", on_click=_confirm_edge,
                                      use_container_width=True)

            st.radio("Cómo repartir los puntos a lo largo de cada borde",
                     ["arclength", "uniform"], index=0, horizontal=True,
                     key=K("edge_mode_sel"),
                     captions=["puntos libres (recomendado con la lupa)",
                               "solo si son marcas de cinta equiespaciadas FÍSICAMENTE"])
            missing_all = [k for k in EDGES_ORDER if len(edges[k]) < 2]
            s1, s2, s3 = st.columns(3)
            if s1.button("↩ Borrar último punto", use_container_width=True):
                if pts:
                    pts.pop()
                    fine_map[cur] = (len(pts) - 1) if pts else None
                    st.rerun()
            if s2.button("🗑 Reiniciar todo", use_container_width=True):
                st.session_state[K("edges")] = {k: [] for k in EDGES_ORDER}
                st.session_state[K("_fine")] = {}
                st.session_state[K("_edge_i")] = 0
                st.rerun()
            _save_lbl = (f"💾 Guardar screen_proj{int(_P5)}.json" if _P5 is not None
                         else "💾 Guardar screen.json")
            if s3.button(_save_lbl, type="primary", use_container_width=True,
                         disabled=bool(missing_all),
                         help=(f"Faltan 2+ puntos en: {', '.join(missing_all)}" if missing_all
                               else "Listo para el paso 6")):
                st.session_state["_do_save"] = (edges, st.session_state.get(K("edge_mode_sel"), "arclength"),
                                                 str(save_path), K("_msg"))
                st.rerun()
            if missing_all:
                st.caption("Para guardar hacen falta ≥2 puntos (las 2 esquinas) en: "
                           + ", ".join(missing_all))
            elif save_path.exists():
                st.caption(f"Ya existe `{save_path}` — guardar lo sobrescribe.")

# ==========================================================================
# paso 6: calcular
# ==========================================================================
elif step.startswith("6 "):
    st.title("6 · Calcular los mapas")
    st.markdown("Genera `output/projN_map.exr`, `projN_black.exr` y `calibration.json`.")
    from lw.diagnose import scan_ok
    from lw.pipeline import per_projector_ready, scan_overlaps
    _c6 = load_config() or {}
    _p6 = _c6.get("projectors", []) or []
    _per6 = bool(_p6) and any((WORK / f"all_white_proj{i}.png").exists()
                              for i in range(len(_p6)))
    _falt6 = [i for i in range(len(_p6)) if not (WORK / f"scan_proj{i}.npz").exists()] if _per6 else []
    _falt6e = [i for i in range(len(_p6)) if not (WORK / f"screen_proj{i}.json").exists()] if _per6 else []
    if _per6:
        st.info("Modo **por proyector**: cada escaneo se hizo con la cámara en su posición, así que "
                "el blend se calcula en el espacio de CONTENIDO. El resultado es un único lienzo: los "
                "mapas de todos los proyectores comparten las mismas coordenadas de imagen.")
        if _falt6:
            st.warning("Faltan escaneos: " + ", ".join(f"proyector {i}" for i in _falt6)
                       + " — vuelve al paso 4.")
        if _falt6e:
            st.warning("Faltan bordes: " + ", ".join(f"proyector {i}" for i in _falt6e)
                       + " — vuelve al paso 5.")
        if _falt6e and not _falt6:
            _ov6 = {i: scan_overlaps(WORK, i) for i in range(len(_p6))}
            _w = []
            for i in range(len(_p6) - 1):
                _v = [d.get("overlap_right") for d in (_ov6.get(i),)] + \
                     [d.get("overlap_left") for d in (_ov6.get(i + 1),)]
                _v = [x for x in _v if x is not None and x == x]
                _w.append(f"P{i}-P{i+1}: " + (f"{np.mean(_v)*100:.0f}%" if _v else "sin medir"))
            st.caption("Solape medido en las costuras → " + " · ".join(_w))
    scans = scan_ok(WORK) if WORK.exists() else []
    bad = [s for s in scans if not s["ok"]]
    if bad:
        det = " · ".join(f"P{s['proj']}: iluminado {s['lit']:.2f}%, decodificado {s['valid']:.2f}%"
                         for s in bad)
        st.error(
            f"**El escaneo no sirve** ({det}).\n\n"
            "El cálculo fallará con «Muy pocas celdas válidas» porque en `work/scan_proj*.npz` no hay "
            "patrones decodificados. Causas típicas, en orden:\n\n"
            "1. **La cámara no es una cámara**: si en el paso 3 elegiste una fuente **NDI de pantalla** "
            "(se llama como el PC, p. ej. `DESKTOP-… (celarg)`), captura el escritorio y no ve la pared. "
            "Elige la webcam real (`Camo`, `NDI Webcam Video`…) en el paso 3.\n"
            "2. **Encuadre**: la pared/ciclorama tiene que llenar el encuadre.\n"
            "3. **Exposición**: ni saturada ni a oscuras (mira `work/all_white.png`).\n"
            "4. **Salida**: los patrones deben salir en el splitter (`displays`), no tapados por "
            "TouchDesigner.\n\n"
            "Vuelve al **paso 4** y reescanea; `diagnose` (en el log del escaneo) da el detalle.")
        if (WORK / "diagnostico.png").exists():
            st.image(str(WORK / "diagnostico.png"), caption="diagnostico.png — qué ve la cámara")
    # --- grosor de la costura: vista previa antes de recalcular -----------------
    # La rampa de mezcla ocupa TODO el solape (es distancia al borde normalizada),
    # por eso sale tan ancha. Aquí se elige cuánto adelgazarla y se ve el resultado
    # sin gastar los minutos del cálculo.
    _bl6 = (_c6.get("blend") or {})
    _nw_def = float(_bl6.get("narrow", 1.0) or 1.0)
    st.subheader("Grosor de la costura (blend)")
    st.caption(
        "El canal B de los mapas ya lleva la rampa y ocupa **todo** el solape: por eso se ve "
        "gruesa. Adelgazarla la comprime alrededor del centro de la costura y **renormaliza**, "
        "así que la suma de luz sigue siendo 1 (no aparece banda brillante). La vista previa "
        "usa las **huellas reales** de tus escaneos (mismo cálculo que los mapas), así que la "
        "forma en cono del solape sale tal cual.")
    _nw6 = st.slider("Adelgazar la costura (×)", 1.0, 6.0,
                     value=min(6.0, max(1.0, _nw_def)), step=0.5, key="blend_narrow6",
                     help="1 = como está ahora · 2 = la mitad de ancho · 3 = un tercio")
    _have6 = bool(_p6) and not (_falt6e or _falt6)
    if not _have6:
        st.caption("Escanea y marca los bordes para ver la vista previa.")
    else:
        try:
            from lw.config import normalize
            from lw.pipeline import geometry_cached, seam_preview
            _cfg6 = json.loads(json.dumps(_c6))      # copia: normalize() añade rect/width
            normalize(_cfg6)
            if not geometry_cached(_cfg6, WORK):
                with st.spinner("Calculando las huellas de los escaneos (sólo la primera vez, "
                                "1–2 min; después queda guardado en `work/seam_geometry.npz`)…"):
                    _pv6 = seam_preview(_cfg6, WORK, narrow=float(_nw6))
            else:
                _pv6 = seam_preview(_cfg6, WORK, narrow=float(_nw6))
            st.image(_pv6["img"],
                     caption=f"Costuras con grosor {_nw6:g} — {_pv6['source']}",
                     use_container_width=True)
            for _s in _pv6["seams"]:
                st.caption(f"Costura {_s['k']}–{_s['k']+1}: banda de "
                           f"**{_s['band_px_canvas']:.0f} px de lienzo** "
                           f"(≈ {_s['band_px_projector']:.0f} px de la imagen del proyector) · "
                           f"solape {_s['overlap_canvas_px']:.0f} px · "
                           f"suma de luz {_s['sum_median']:.3f}")
            if _pv6.get("stale_maps"):
                st.warning("Los mapas de `output/` son anteriores a estos escaneos o bordes: "
                           "la vista previa es la correcta, pero lo que proyectas aún es el "
                           "mapa viejo. Recalcula para que coincidan.")
        except Exception as _e6:
            st.info(f"Vista previa no disponible: {_e6}")
        if abs(_nw6 - _nw_def) > 1e-9:
            st.caption(f"config.json tiene `blend.narrow = {_nw_def:g}`; al calcular se guarda {_nw6:g}.")

    if not ss["edges"] or _falt6e or _falt6:
        st.warning("Faltan bordes (`screen_proj*.json`) o escaneos — termina los pasos 4 y 5.")
    elif bad:
        st.button("▶ Calcular (≈1–3 min)", use_container_width=True, disabled=True,
                  help="Reescanea primero: el escaneo actual no tiene patrones válidos")
    elif st.button(f"▶ Calcular con grosor {_nw6:g} (≈1–3 min)", use_container_width=True):
        def _set_narrow6(c, _v=float(_nw6)):
            c.setdefault("blend", {})["narrow"] = _v
        edit_config(_set_narrow6)
        calibrate_cli(["compute", "-c", "config.json"], key_prefix="comp")
        reset_cache()
    meta_path = OUT / "calibration.json"
    if meta_path.exists():
        m = json.loads(meta_path.read_text(encoding="utf-8"))
        st.success("Mapas generados")
        st.json({k: m[k] for k in ("aspect_hint_from_camera", "max_overlap", "uniformity")
                 if k in m}, expanded=False)
        for p in m.get("projectors", []):
            pv = OUT / p.get("file", "").replace("map.exr", "preview.png")
            if pv.exists():
                st.image(str(pv), caption=f"{p['name']} — vista previa")
        if (OUT / "content_overview.png").exists():
            st.image(str(OUT / "content_overview.png"),
                     caption="content_overview.png — huella de cada proyector en el lienzo común")
            _cp = [(p.get("name", f"P{p.get('index')}"), p.get("content_window"))
                   for p in m.get("projectors", []) if p.get("content_window")]
            if _cp:
                st.caption("Ventana de lienzo de cada proyector (u): " + " · ".join(
                    f"{n} [{w[0]:.3f}, {w[1]:.3f}]" for n, w in _cp))
        elif (OUT / "camera_overview.png").exists():
            st.image(str(OUT / "camera_overview.png"),
                     caption="camera_overview.png — huellas y rejilla")
        if st.button("📝 Registrar esta sesión en la bóveda Obsidian"):
            st.info(obsidian_log_session(m))

# ==========================================================================
# paso 7: verificar
# ==========================================================================
elif step.startswith("7 "):
    st.title("7 · Verificar en los proyectores")
    st.markdown(
        "Se abre una ventana sobre los proyectores con rejilla/blanco. Controles en vivo:\n"
        "**g/G** gamma · **u/U** uniformidad (0.5–0.8 en convexo) · **b/B** nivel de negro "
        "(en modo Negro) · **m** modo · **q** salir (desde esa ventana).")
    if not ss["computed"]:
        st.warning("Primero calcula (paso 6).")
    elif st.button("▶ Verificar en los proyectores", use_container_width=True):
        calibrate_cli(["verify", "-c", "config.json"], key_prefix="verify")

# ==========================================================================
# paso 8: touchdesigner
# ==========================================================================
elif step.startswith("8 "):
    st.title("8 · TouchDesigner")
    if not ss["computed"]:
        st.warning("Primero calcula (paso 6).")
    else:
        st.markdown(
            f"1. En TD crea un **Text DAT** y pega `touchdesigner/build_librewarp.py`.\n"
            f"2. Arriba del script cambia `CALIB_DIR`:\n\n"
            f"   ```python\n   CALIB_DIR = r'{OUT}'\n   ```\n\n"
            f"3. Clic derecho → **Run Script** → aparece el COMP `librewarp` "
            f"(canvas del tamaño del splitter).\n"
            f"4. Conecta tu contenido y pulsa **Abrir ventanas**.")
        if os.name == "nt" and st.button("📂 Abrir la carpeta output"):
            os.startfile(str(OUT))


# ==========================================================================
# paso 9: agente + obsidian
# ==========================================================================
elif step.startswith("9 "):
    st.title("9 · Agente copiloto (fcc-claude) + aprendizaje en Obsidian")
    st.markdown(
        "El proyecto viene preparado para que un agente te guíe y **aprenda**:\n\n"
        "1. Abre una terminal en `librewarp\\` y lanza tu `fcc-claude` ahí.\n"
        "2. Lee automáticamente `CLAUDE.md`: pasos, tabla de errores, reglas genéricas de "
        "pantallas/splitters y cámaras.\n"
        "3. **Bóveda Obsidian** en `ObsidianVault\\LibreWarp\\`:\n"
        "   - `Fallos y soluciones.md` — el agente la CONSULTA antes de diagnosticar y AGREGA "
        "cada fallo resuelto (aprende de tu setup).\n"
        "   - `Registro de calibraciones.md` — una fila por sesión.\n"
        "   - `Setup actual.md` — hardware vigente.\n"
        "   - El paso 6 tiene un botón para registrar la sesión automáticamente.\n"
        "4. Prompts útiles (puedes pedir al agente que registre un fallo en la bóveda):\n"
        "   - `guiame con la calibración`\n"
        "   - `mi escaneo sale con rayas mezcladas — qué es`\n"
        "   - `registra este fallo y el arreglo en la bóveda Obsidian`\n"
    )
    st.code(
        f"cd {HERE.parent}\n"
        f"fcc-claude\n"
        f"> guiame con la calibración",
        language="bash",
    )
    st.markdown(
        "5. **Sub-agente listo**: abre `fcc-claude` en la carpeta del proyecto y pídele lo que "
        "necesites. Lee `CLAUDE.md` y la bóveda Obsidian para contextualizarse."
    )