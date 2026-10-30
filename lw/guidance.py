"""
Módulo de guía para calibración asistida por Obsidian.
Proporciona instrucciones específicas de colocación de cámara y genera notas de Obsidian.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import cv2

from . import obsidian


def generate_camera_placement_guidance(cfg: dict, work: Path) -> str:
    """
    Analiza la configuración actual y proporciona instrucciones específicas
    para la colocación de la cámara.

    Args:
        cfg: Configuración normalizada de LibreWarp
        work: Directorio de trabajo

    Returns:
        Cadena con instrucciones de colocación de cámara
    """
    displays = cfg.get("displays", [])
    projectors = cfg.get("projectors", [])

    if not displays or not projectors:
        return "No se pudo determinar la configuración de displays/proyectores."

    # Asumimos que usamos el primer display (común con QuadHead2Go)
    display = displays[0]
    disp_x, disp_y = display.get("x", 0), display.get("y", 0)
    disp_width, disp_height = display.get("width", 1920), display.get("height", 1080)
    layout = display.get("layout", "3x1")

    # Calcular el centro aproximado del área de proyección
    center_x = disp_x + disp_width // 2
    center_y = disp_y + disp_height // 2

    # Determinar número de columnas y filas del layout
    if "x" in layout:
        cols, rows = map(int, layout.split("x"))
    else:
        # Layout predeterminado basado en resolución común
        if disp_width == 5760 and disp_height == 1080:  # 3x1
            cols, rows = 3, 1
        elif disp_width == 3840 and disp_height == 2160:  # 2x2
            cols, rows = 2, 2
        else:
            cols, rows = 3, 1  # predeterminado

    # Calcular ancho y height de cada proyector individual
    proj_width = disp_width // cols
    proj_height = disp_height // rows

    # Instrucciones básicas de colocación
    guidance = f"""# Guía de Colocación de Cámara

## Configuración Detectada
- Display: {display.get('name', 'Desconocido')} ({disp_width}x{disp_height})
- Layout: {layout} ({cols}x{rows})
- Proyectores configurados: {len(projectors)}
- Centro del display: ({center_x}, {center_y})

## Recomendaciones de Colocación

### Distancia y Ángulo
1. **Distancia desde la pared**: Para una lente estándar, coloque la cámara a una distancia aproximadamente igual al ancho del área de proyección.
   - Ancho total de proyección: {disp_width}px
   - Distancia recomendada: {disp_width // 2} - {disp_width} px (ajustar según lente de cámara)

2. **Altura**: Centrada verticalmente con el display
   - Altura del display: {disp_height}px
   - Posición Y recomendada: {center_y} ± {disp_height // 4} px

3. **Ángulo**: Apuntando perpendicularmente al centro del display
   - Ángulo horizontal: 0° (centrado en {center_x})
   - Ángulo vertical: 0° (centrado en {center_y})

### Configuración de Cámara (para ajuste fino)
- **Enfoque**: Enfocar en el centro del área de proyección
- **Exposición**: Ajustar hasta que no haya saturación roja en modo 'ajustar cámara'
- **Balance de blancos**: Fijo (ej. Daylight), nunca automático
- **Estabilización**: APAGADA (especialmente importante para NDI)

### Verificación de Cobertura
Después de colocar la cámara, ejecute:
```
python calibrate.py adjust -c config.json
```
y verifique que:
1. No aparezcan píxeles rojos (saturación) al mostrar blanco
2. Las rayas finas se vean nítidas en todo el área de proyección
3. Cada proyector muestra su número completo al ejecutar 'identify'

## Próximos Pasos
1. Coloque la cámara siguiendo estas recomendaciones
2. Ajuste la exposición/foco hasta obtener una imagen óptima
3. Proceda con el escaneo: `python calibrate.py scan -c config.json`
"""

    # Si hay work/all_white.png, podemos dar recomendaciones más específicas
    all_white_path = work / "all_white.png"
    if all_white_path.exists():
        try:
            img = cv2.imread(str(all_white_path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                mean_val = img.mean()
                std_val = img.std()
                guidance += f"\n\n## Verificación de Imagen de Referencia\n"
                guidance += f"- Imagen all_white.png: media={mean_val:.1f}, desviación={std_val:.1f}\n"
                if mean_val < 50:
                    guidance += "- ⚠️ Imagen muy oscura: aumentar exposición o verificar que los proyectores estén encendidos\n"
                elif mean_val > 200:
                    guidance += "- ✅ Nivel de brillo adecuado para ajuste de cámara\n"
                else:
                    guidance += "- ⚠️ Brillo bajo: considere aumentar ligeramente la exposición\n"
        except Exception:
            pass  # Ignorar errores de lectura de imagen

    return guidance


def generate_edge_marking_note(cfg: dict, work: Path) -> Path:
    """
    Crea una nota de Obsidian con la imagen de referencia para marcar bordes.

    Args:
        cfg: Configuración normalizada
        work: Directorio de trabajo

    Returns:
        Ruta a la nota de Obsidian creada, o None si la bóveda no está disponible
    """
    # Usar la utilidad de Obsidian
    all_white_path = work / "all_white.png"
    if not all_white_path.exists():
        raise FileNotFoundError(f"No se encontró {all_white_path}. Ejecute 'scan' primero.")

    return obsidian.create_edge_marking_note(all_white_path, work)


def extract_edge_points_from_obsidian(note_path: Path) -> Dict[str, Any]:
    """
    Extrae los puntos de borde marcados de una nota de Obsidian.
    Esta implementación asume un formato simple de anotación.

    Args:
        note_path: Ruta a la nota de Obsidian

    Returns:
        Diccionario con los puntos extraídos en formato screen.json
    """
    # Delegar a la utilidad de Obsidian
    result = obsidian.extract_edge_points_from_obsidian_note(note_path)
    if result is not None:
        return result

    # Si no se pueden extraer automáticamente, retornar None para indicar
    # que se necesita intervención manual o método tradicional
    return None


def generate_status_note(cfg: dict, work: Path, out: Path, step_name: str,
                        metrics: Dict[str, Any] = None) -> Optional[Path]:
    """
    Genera una nota de estado en el vault de Obsidian después de completar un paso.

    Args:
        cfg: Configuración normalizada
        work: Directorio de trabajo
        out: Directorio de salida
        step_name: Nombre del paso completado
        metrics: Métricas opcionales para incluir en la nota

    Returns:
        Ruta a la nota de estado creada, o None si la bóveda no está disponible
    """
    # Preparar metadatos
    metadata = metrics or {}

    # Añadir información básica de la configuración
    displays = cfg.get("displays", [])
    projectors = cfg.get("projectors", [])

    if displays:
        metadata["display_count"] = len(displays)
        if len(displays) > 0:
            d = displays[0]
            metadata["primary_display"] = f"{d.get('name', 'Desconocido')} ({d.get('width', 0)}x{d.get('height', 0)})"

    metadata["projector_count"] = len(projectors)

    # Crear la nota usando la utilidad de Obsidian
    return obsidian.create_status_note(step_name, metadata, work, out)