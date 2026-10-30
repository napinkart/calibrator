import json
import numpy as np
import cv2
from pathlib import Path
from scipy import ndimage
from scipy.interpolate import CubicSpline, griddata
from scipy.signal import savgol_filter

# ... (keep existing imports and constants) ...

# Nueva función de guía visual para marcar bordes con sugerencias
def click_edges_with_guidance(image: np.ndarray, existing: dict | None = None,
                               max_display: int = 1600, ghosts: list[dict] | None = None,
                               suggested_edges: dict | None = None,
                               overlap_source: str = "measured") -> dict | None:
    """Ventana OpenCV para marcar los bordes CON guía visual para evitar romper solapes.

    Nueva funcionalidad:
    - Superposición de bordes sugeridos (semi-transparentes) para alinear visualmente.
    - Grilla de referencia que muestra dónde deberían caer las ventanas de lienzo.
    - Validación en tiempo real: si un borde se aleja demasiado de la posición esperada, se muestra un aviso.
    - Guardado automático de la imagen con guía para referencia posterior.
    """
    # ... (copy existing click_edges logic up to the point where edges are defined) ...

    # Nueva sección de guía visual
    overlay = np.zeros_like(image, dtype=np.uint8)
    guide_img = image.copy()

    # 1. Mostrar bordes sugeridos si se pasan como argumento
    if suggested_edges:
        for k, pts in suggested_edges.items():
            if k in edges and len(edges[k]) >= 2:
                col = (200, 200, 200) if k == "top" else (200, 150, 150) if k == "bottom" else (150, 200, 150) if k == "left" else (150, 150, 200)
                cv2.polylines(overlay, [np.array(pts, np.int32)], True, col, 2, cv2.LINE_AA)
                # Mostrar número de punto en los extremos
                for idx in (0, len(pts)-1):
                    p = pts[idx]
                    cv2.putText(overlay, f"{k[0].upper()}-{idx+1}", (int(p[0])-10, int(p[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        # Mezclar sugerencias suavemente
        guide_img = cv2.addWeighted(image, 0.85, overlay, 0.35, 0)

    # 2. Mostrar ventanas de lienzo esperadas (para alinear los bordes)
    if overlap_source == "measured" and existing:
        try:
            from ..pipeline import load_geometry
            geo = load_geometry(Path('work'))
            if geo:
                wins = geo['wins']
                for i, (u0, u1) in enumerate(wins):
                    # Dibujar un rectángulo sutil en el espacio de contenido
                    w_px = int((u1 - u0) * image.shape[1])
                    h_px = max(30, int(0.4 * image.shape[0]))
                    if i == 0:
                        pos = (int(u0 * image.shape[1]), int(0.1 * image.shape[0]))
                    elif i == len(wins)-1:
                        pos = (int(u1 * image.shape[1]) - w_px, int(0.6 * image.shape[0]))
                    else:
                        pos = (int(u0 * image.shape[1]), int(0.3 * image.shape[0]))
                    cv2.rectangle(guide_img, pos, (pos[0]+w_px, pos[1]+h_px), (100, 100, 255), 1)
                    cv2.putText(guide_img, f"P{i+1}", (pos[0]+2, pos[1]+12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        except Exception:
            pass

    # 3. Validación en tiempo real: si un borde se aleja demasiado de la posición esperada
    if suggested_edges:
        for k, pts in suggested_edges.items():
            if k in edges and len(edges[k]) >= 2:
                suggested = np.array(suggested_edges[k])
                current = np.array(edges[k])
                dists = np.linalg.norm(current - suggested, axis=1)
                avg_dist = np.mean(dists)
                if avg_dist > 30:
                    cv2.putText(guide_img, f"⚠️ BORDE {k} DESVIADO {avg_dist:.1f}px", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Mostrar la imagen con guía
    cv2.imshow("LibreWarp - Marcar Bordes (con Guía)", guide_img)
    cv2.waitKey(1)

    # Guardar la imagen con guía para referencia posterior
    guide_path = Path('work') / 'edge_guide_preview.png'
    cv2.imwrite(str(guide_path), guide_img)

    # Mostrar mensaje final
    msg = f"[{state['i']+1}/4] {EDGE_HELP[k]}  ({len(edges.get(k, []))} pts)  Enter=siguiente  clic der=borrar  Esc=cancelar"
    msg += f"  |  Guía: bordes sugeridos (transparentes) y ventanas de lienzo (amarillas)"
    if ghosts:
        msg += f"  |  g=vecinos ({'visibles' if ghost_vis else 'ocultos'})"
    cv2.rectangle(guide_img, (0, 0), (guide_img.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(guide_img, msg, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colors[k], 1, cv2.LINE_AA)
    cv2.imshow(win, guide_img)
    cv2.waitKey(30)

    # Antes de cerrar, validar que los bordes no estén demasiado corridos
    if suggested_edges:
        for k, pts in suggested_edges.items():
            if k in edges and len(edges[k]) >= 2:
                suggested = np.array(suggested_edges[k])
                current = np.array(edges[k])
                dists = np.linalg.norm(current - suggested, axis=1)
                if np.mean(dists) > 20:
                    if not st.confirm("⚠️ Los bordes se desviaron mucho del esperado. ¿Continuar de todos modos?"):
                        return None

    return {k: [list(map(float, p)) for p in edges[k]] for k in EDGES}
