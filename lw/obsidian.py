"""
Utilidades para integración con el vault de Obsidian.
Funciones helper para crear, leer y gestionar notas en el vault de Obsidian de LibreWarp.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Ruta al vault de Obsidian (definida en CLAUDE.md)
OBSIDIAN_VAULT = Path(__file__).parent.parent.parent / "ObsidianVault" / "LibreWarp"

# La bóveda es OPCIONAL: en PCs nuevas puede no existir. Si falta o no es
# escribible, el sistema sigue funcionando y simplemente no registra notas.
_warned_unavailable = False


def get_vault_path() -> Path:
    """Obtiene la ruta al vault de Obsidian de LibreWarp."""
    return OBSIDIAN_VAULT


def _vault_available() -> bool:
    """True si la bóveda existe (o se puede crear) y es escribible.

    Avisa una sola vez por proceso para no ensuciar la consola del scan/compute.
    """
    global _warned_unavailable
    vault_path = get_vault_path()
    try:
        vault_path.mkdir(parents=True, exist_ok=True)
        probe = vault_path / ".write_test"
        probe.write_text("", encoding="utf-8")
        probe.unlink()
        return True
    except OSError as e:
        if not _warned_unavailable:
            print(f"[obsidian] Bóveda no disponible ({e}); no se registrarán notas.")
            _warned_unavailable = True
        return False


def initialize_vault_if_needed():
    """
    Inicializa el vault de Obsidian con los archivos necesarios si no existen.
    Crea los archivos mencionados en CLAUDE.md:
    - 00 Índice.md
    - Fallos y soluciones.md
    - Registro de calibraciones.md
    - Setup actual.md
    """
    if not _vault_available():
        return
    vault_path = get_vault_path()

    # Asegurarse de que el directorio del vault existe
    vault_path.mkdir(parents=True, exist_ok=True)

    # Archivos que deben existir con su contenido inicial
    required_files = {
        "00 Índice.md": """# Índice de LibreWarp

Este es el mapa de contenidos (MOC) del vault de Obsidian de LibreWarp.

## Notas principales
- [[Fallos y soluciones.md]] - Registro de problemas encontrados y sus soluciones
- [[Registro de calibraciones.md]] - Historial de calibraciones realizadas
- [[Setup actual.md]] - Configuración actual del hardware y software

## Cómo usar este vault
Este vault se actualiza automáticamente durante el proceso de calibración de LibreWarp.
Las notas de estado se generan en la carpeta `status/` y las imágenes de referencia en `references/`.

Última actualización: {timestamp}
""",
        "Fallos y soluciones.md": """# Fallos y soluciones

Registro de problemas encontrados durante la calibración de LibreWarp y sus soluciones correspondientes.

## Formato de entrada
| Fecha | Síntoma | Causa | Corrección | Enlace a nota |
|-------|---------|-------|------------|---------------|
| 2026-10-17 | Ejemplo de síntoma | Ejemplo de causa | Ejemplo de corrección | [[20261017_120000]] |

*Nota: Los enlaces apuntan a notas de estado generadas automáticamente.*

## Entradas
""",
        "Registro de calibraciones.md": """# Registro de calibraciones

Historial de calibraciones realizadas con LibreWarp.

## Formato de entrada
| Fecha | Uniformidad (min/máx) | Luz conservada (%) | Máximo solapamiento | Notas |
|-------|----------------------|-------------------|---------------------|-------|
| 2026-10-17 | 0.85/0.95 | 78% | 0.15 | Ejemplo de calibración exitosa |

## Entradas
""",
        "Setup actual.md": """# Setup actual

Configuración actual del hardware y software para LibreWarp.

## Hardware
- **Proyectores**: [Detallar cantidad y modelo]
- **Cámaras**: [Detallar tipo y modelo]
- **Splitter**: [Ej: Matrox QuadHead2Go, especificando layout]
- **Superficie de proyección**: [Describir forma y material]

## Software
- **Versión de LibreWarp**: [Especificar]
- **Configuración actual**: Ver `config.json` en el directorio calibrator/
- **Última calibración**: [Fecha]

## Notas
Este archivo debe actualizarse manualmente cuando cambie el hardware o la configuración significativa.
Última actualización: {timestamp}
"""
    }

    for filename, template in required_files.items():
        file_path = vault_path / filename
        if not file_path.exists():
            # Generar contenido con timestamp actual
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            content = template.format(timestamp=timestamp)
            file_path.write_text(content, encoding="utf-8")


def ensure_vault_directory(subdir: str = "") -> Path:
    """
    Asegura que existe un directorio en el vault de Obsidian.
    También inicializa el vault con los archivos necesarios si es la primera vez.

    Args:
        subdir: Subdirectorio dentro del vault (ej. "status", "references")

    Returns:
        Path al directorio asegurado, o None si la bóveda no está disponible
    """
    if not _vault_available():
        return None
    vault_path = get_vault_path()
    if subdir:
        target_dir = vault_path / subdir
    else:
        target_dir = vault_path

    target_dir.mkdir(parents=True, exist_ok=True)

    # Inicializar el vault con archivos necesarios si es la primera vez
    # Solo hacer esto cuando se accede al directorio raíz del vault
    if not subdir:  # Cuando se pide el directorio raíz del vault
        initialize_vault_if_needed()

    return target_dir


def create_status_note(step_name: str, metadata: Optional[Dict[str, Any]] = None,
                      work_path: Optional[Path] = None,
                      output_path: Optional[Path] = None) -> Path:
    """
    Crea una nota de estado en el vault de Obsidian.

    Args:
        step_name: Nombre del paso completado
        metadata: Metadatos adicionales para incluir en la nota
        work_path: Ruta al directorio de trabajo (opcional)
        output_path: Ruta al directorio de salida (opcional)

    Returns:
        Path a la nota creada, o None si la bóveda no está disponible
    """
    status_dir = ensure_vault_directory("status")
    if status_dir is None:
        return None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    note_filename = f"{step_name.title()} - {timestamp}.md"
    note_path = status_dir / note_filename

    # Construir contenido de la nota
    note_content = f"""---
type: status
step: {step_name}
timestamp: {timestamp}
"""

    if metadata:
        for key, value in metadata.items():
            note_content += f"{key}: {value}\n"

    note_content += "---\n\n"
    note_content += f"# {step_name.title()} Completado\n\n"

    if metadata:
        note_content += "## Métricas\n"
        for key, value in metadata.items():
            note_content += f"- {key}: {value}\n"
        note_content += "\n"

    if work_path and work_path.exists():
        note_content += f"## Directorio de trabajo\n"
        note_content += f"- `{work_path}`\n\n"

    if output_path and output_path.exists():
        note_content += f"## Directorio de salida\n"
        note_content += f"- `{output_path}`\n\n"

        # Listar archivos importantes en output
        important_files = []
        for ext in ["*.exr", "*.json", "*.npy"]:
            important_files.extend(output_path.glob(ext))

        if important_files:
            note_content += f"## Archivos generados\n"
            for f in important_files[:10]:  # Limitar a 10 archivos
                note_content += f"- `{f.name}`\n"
            note_content += "\n"

    note_content += "---\n*Generado automáticamente por LibreWarp Agent*\n"

    note_path.write_text(note_content, encoding="utf-8")
    return note_path


def create_guidance_note(guidance_type: str, content: str,
                        work_path: Optional[Path] = None) -> Path:
    """
    Crea una nota de guía en el vault de Obsidian.

    Args:
        guidance_type: Tipo de guía (ej. "camera_placement", "troubleshooting")
        content: Contenido de la guía en formato markdown
        work_path: Ruta al directorio de trabajo (opcional)

    Returns:
        Path a la nota creada, o None si la bóveda no está disponible
    """
    status_dir = ensure_vault_directory("status")
    if status_dir is None:
        return None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    note_filename = f"{guidance_type.title()} Guidance - {timestamp}.md"
    note_path = status_dir / note_filename

    note_content = f"""---
type: guidance
guidance_type: {guidance_type}
timestamp: {timestamp}
"""

    if work_path:
        note_content += f"work_path: {work_path}\n"

    note_content += "---\n\n"
    note_content += content
    note_content += "\n\n---\n*Generado automáticamente por LibreWarp Agent*\n"

    try:
        note_path.write_text(note_content, encoding="utf-8")
    except OSError:
        return None
    return note_path


def create_edge_marking_note(reference_image_path: Path,
                           work_path: Optional[Path] = None) -> Path:
    """
    Crea una nota de Obsidian para marcado de bordes con imagen de referencia.

    Args:
        reference_image_path: Path a la imagen de referencia (usualmente all_white.png)
        work_path: Ruta al directorio de trabajo (opcional)

    Returns:
        Path a la nota creada en el vault, o None si la bóveda no está disponible
    """
    # Asegurar que existe el directorio de referencias
    refs_dir = ensure_vault_directory("references")
    if refs_dir is None:
        return None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Copiar imagen de referencia al vault de referencias
    ref_filename = f"all_white_{timestamp}.png"
    ref_dst = refs_dir / ref_filename

    # Copiar el archivo
    import shutil
    try:
        shutil.copy2(reference_image_path, ref_dst)
    except (OSError, FileNotFoundError):
        # Si la imagen no existe o no se puede copiar, no creamos la nota
        return None

    # Crear la nota de marcado de bordes
    note_filename = f"Edge Marking - {timestamp}.md"
    note_path = ensure_vault_directory("") / note_filename  # Nota en raíz del vault

    note_content = f"""---
type: edge-marking
reference: references/{ref_filename}
status: pending
timestamp: {timestamp}
"""

    if work_path:
        note_content += f"work_path: {work_path}\n"

    note_content += """---
# Marcado de Bordes de Pantalla

## Instrucciones

Marque los cuatro bordes de la superficie de proyección en esta imagen:
- **SUPERIOR**: De izquierda a derecha
- **DERECHO**: De arriba a abajo
- **INFERIOR**: De izquierda a derecha
- **IZQUIERDO**: De arriba a abajo

## Cómo marcar puntos
1. Use las herramientas de anotación de Obsidian para colocar puntos a lo largo de cada borde
2. Coloque al menos 4 puntos por borde (más puntos = mejor precisión)
3. Siga exactamente el orden indicado para cada borde
4. Para superficies curvas, siga la curva natural de la pared
5. Para superficies planas, marque puntos equiespaciados

## Formato de anotación esperado
El sistema busca puntos en estos formatos:
- `[[point::x,y]]` o `[[point x,y]]` (recomendado)
- O simplemente coordenadas como `x,y` en una línea

Ejemplo de anotación válida:
```
[[point::100,200]]
[[point::150,250]]
[[point::200,300]]
```

## Nota importante
- Los puntos deben seguir el orden especificado para cada borde
- En esquinas, marque el punto en ambas líneas que se unen (será el mismo punto físico)
- Guarde la nota cuando termine de marcar todos los bordes

## Próximos pasos
Después de guardar esta nota, el sistema detectará automáticamente los cambios y
procederá con el cálculo.

![Imagen de referencia](references/{ref_filename})
"""

    note_path.write_text(note_content, encoding="utf-8")
    return note_path


def get_latest_note_by_type(note_type: str, directory: str = "status") -> Optional[Path]:
    """
    Obtiene la nota más reciente de un tipo específico.

    Args:
        note_type: Tipo de nota (ej. "status", "guidance", "edge-marking")
        directory: Directorio dentro del vault donde buscar

    Returns:
        Path a la nota más reciente o None si no se encuentra
    """
    target_dir = ensure_vault_directory(directory)
    if target_dir is None or not target_dir.exists():
        return None

    # Buscar notas que coincidan con el tipo en su frontmatter
    matching_notes = []
    for note_file in target_dir.glob("*.md"):
        try:
            content = note_file.read_text(encoding="utf-8")
            if content.startswith("---"):
                # Extraer frontmatter
                end_marker = content.find("---", 3)
                if end_marker != -1:
                    frontmatter = content[3:end_marker]
                    if f"type: {note_type}" in frontmatter:
                        matching_notes.append(note_file)
        except Exception:
            continue  # Ignorar archivos que no se pueden leer

    if not matching_notes:
        return None

    # Ordenar por fecha de modificación (más reciente primero)
    matching_notes.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matching_notes[0]


def extract_edge_points_from_obsidian_note(note_path: Path) -> Optional[Dict[str, Any]]:
    """
    Extrae puntos de borde de una nota de Obsidian.
    Busca anotaciones en formato [[point::x,y]] o similare.
    Asume que el usuario marca puntos en el orden correcto para cada borde:
    - TOP: de izquierda a derecha
    - RIGHT: de arriba a abajo
    - BOTTOM: de izquierda a derecha
    - LEFT: de arriba a abajo

    Args:
        note_path: Path a la nota de Obsidian

    Returns:
        Diccionario con los puntos extraídos en formato compatible con screen.json
        o None si no se pueden extraer puntos válidos
    """
    try:
        content = note_path.read_text(encoding="utf-8")

        # Buscar patrones de puntos anotados
        # Formato esperado: [[point::x,y]] o [[point:x,y]]
        import re

        # Patrón para capturar coordenadas
        point_pattern = r'\[\[point[:::](\d+\.?\d*),\s*(\d+\.?\d*)\]\]'
        matches = re.findall(point_pattern, content)

        if not matches:
            # Intentar formato alternativo sin doble dos puntos
            point_pattern = r'\[\[point\s+(\d+\.?\d*)[,]\s*(\d+\.?\d*)\]\]'
            matches = re.findall(point_pattern, content)

        points = []
        if matches:
            for x_str, y_str in matches:
                try:
                    points.append((float(x_str), float(y_str)))
                except ValueError:
                    continue
        else:
            # Buscar patrones como "x,y" o "x y" en líneas
            for line in content.split('\n'):
                coord_matches = re.findall(r'(\d+\.?\d*)[,;\s](\d+\.?\d*)', line)
                for x_str, y_str in coord_matches:
                    try:
                        points.append((float(x_str), float(y_str)))
                    except ValueError:
                        continue

        if len(points) < 8:  # Necesitamos al menos 2 puntos por borde (8 total para 4 bordes)
            return None

        # Distribuir los puntos entre los 4 bordes
        # Asumimos que el usuario marcó en el orden: top (izq→der), right (arr→aba), bottom (izq→der), left (arr→aba)
        points_per_edge = len(points) // 4
        remainder = len(points) % 4

        # Ajustar para distribuir puntos sobrantes
        edges = {
            "top": [],
            "right": [],
            "bottom": [],
            "left": []
        }

        point_index = 0
        for edge_name in ["top", "right", "bottom", "left"]:
            # Calcular cuántos puntos asignar a este borde
            num_points = points_per_edge + (1 if point_index < remainder else 0)

            # Asignar puntos a este borde
            for i in range(num_points):
                if point_index < len(points):
                    x, y = points[point_index]
                    edges[edge_name].append([x, y])
                    point_index += 1

        # Verificar que cada borde tenga al menos 2 puntos
        for edge_name, points_list in edges.items():
            if len(points_list) < 2:
                return None

        # Crear estructura de screen.json
        screen_data = {
            "edge_param": "arclength",  # valor por defecto
            "edges": edges
        }

        return screen_data

    except Exception as e:
        # En caso de error, retornamos None para que el usuario use el método tradicional
        return None


def watch_for_edge_marking_notes(cfg: dict, work_path: Path, out_path: Path, stop_event=None):
    """
    Fondo que observa el vault de Obsidian por notas de marcado de bordes y las procesa automáticamente.

    Args:
        cfg: Configuración de LibreWarp
        work_path: Directorio de trabajo
        out_path: Directorio de salida
        stop_event: Evento para señalar cuando detener el watcher (opcional)
    """
    import time
    import threading

    # Si la bóveda no está disponible, no podemos observar notas: regresamos sin error
    if not _vault_available():
        return

    vault_path = get_vault_path()
    status_dir = vault_path / "status"

    # Asegurarse de que el directorio de estado existe
    status_dir.mkdir(parents=True, exist_ok=True)

    # Rastreamos las notas que ya hemos procesado para evitar duplicados
    processed_notes = set()

    print(f"[WATCHER] Iniciando observación de {status_dir} para notas de marcado de bordes...")
    print(f"[WATCHER] Presione Ctrl+C para detener el observador")

    try:
        while not (stop_event and stop_event.is_set()):
            # Buscar notas de marcado de bordes en el directorio de estado
            for note_file in status_dir.glob("Edge Marking - *.md"):
                try:
                    # Verificar si esta nota ya ha sido procesada
                    if note_file in processed_notes:
                        continue

                    # Leer la nota para verificar su estado
                    content = note_file.read_text(encoding="utf-8")
                    if "status: complete" in content:
                        # Ya fue procesada, marcarla como vista
                        processed_notes.add(note_file)
                        continue

                    # Verificar si tiene el frontmatter de edge-marking
                    if content.startswith("---") and "type: edge-marking" in content:
                        print(f"[WATCHER] Detectada nueva nota de marcado de bordes: {note_file.name}")

                        # Intentar extraer puntos de la nota
                        edge_points = extract_edge_points_from_obsidian_note(note_file)

                        if edge_points is not None:
                            # Guardar los puntos extraídos en screen.json
                            screen_path = work_path / "screen.json"
                            with open(screen_path, "w") as f:
                                json.dump(edge_points, f, indent=2)

                            print(f"[WATCHER] Puntos de borde extraídos y guardados en {screen_path}")
                            print(f"[WATCHER] Procediendo con el cálculo automático...")

                            # Ejecutar compute automáticamente
                            from lw import pipeline
                            try:
                                pipeline.compute(cfg, work_path, out_path)
                                print(f"[WATCHER] ✅ Cálculo completado exitosamente")

                                # Actualizar la nota de Obsidian para marcarla como completada
                                try:
                                    note_content = note_file.read_text(encoding="utf-8")
                                    if "status: pending" in note_content:
                                        note_content = note_content.replace("status: pending", "status: complete")
                                        note_file.write_text(note_content, encoding="utf-8")
                                        print(f"[WATCHER] Nota de marcado de bordes actualizada a 'complete'")
                                except Exception:
                                    pass  # No es crítico si no podemos actualizar la nota

                                # Generar nota de estado para el cálculo completo
                                from datetime import datetime
                                status_dir = vault_path / "status"
                                status_dir.mkdir(parents=True, exist_ok=True)

                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                note_filename = f"Compute Complete - {timestamp}.md"
                                note_path_compute = status_dir / note_filename

                                # Leer métricas del cálculo
                                metrics = {}
                                if (out_path / "calibration.json").exists():
                                    try:
                                        cal_data = json.loads((out_path / "calibration.json").read_text())
                                        uniformity = cal_data.get("uniformity", {})
                                        if uniformity:
                                            metrics["uniformity_min_max"] = uniformity.get("brightness_min_over_max", "N/A")
                                            metrics["light_conserved_percent"] = uniformity.get("light_kept_avg", 0) * 100
                                        metrics["max_overlap"] = cal_data.get("max_overlap", 0)
                                        metrics["projectors_processed"] = len(cal_data.get("projectors", []))
                                    except Exception:
                                        pass

                                note_content = f"""---
type: status
step: compute_complete
timestamp: {timestamp}
---

# Cálculo Completado (Flujo Obsidian Automático)

## Resultados
"""
                                if metrics:
                                    for key, value in metrics.items():
                                        note_content += f"- {key}: {value}\n"
                                else:
                                    note_content += "- Métricas no disponibles\n"

                                note_content += f"\n## Archivos generados\n"
                                note_content += f"- Mapas de warp: `output/projN_map.exr`\n"
                                note_content += f"- Mapas de negro: `output/projN_black.exr`\n"
                                note_content += f"- Archivo de calibración: `output/calibration.json`\n"

                                note_content += f"\n## Próximos pasos\n"
                                note_content += f"1. **Verificar resultados**: `python calibrate.py verify -c config.json`\n"
                                note_content += f"2. **Usar en TouchDesigner**: Ejecutar el Text DAT con CALIB_DIR apuntando a `output/`\n"

                                note_path_compute.write_text(note_content, encoding="utf-8")
                                print(f"[WATCHER] Nota de estado de cálculo generada en: {note_path_compute}")

                            except Exception as e:
                                print(f"[WATCHER] ❌ Error durante el cálculo: {e}")

                            # Marcar esta nota como procesada
                            processed_notes.add(note_file)
                        else:
                            print(f"[WATCHER] ⚠️ No se pudieron extraer puntos válidos de la nota {note_file.name}")
                            print(f"[WATCHER]     Asegúrese de haber marcado puntos en el formato correcto:")
                            print(f"[WATCHER]     - Use anotaciones como [[point::x,y]] o [[point x,y]]")
                            print(f"[WATCHER]     - Marque al menos 2 puntos por borde en el orden correcto")
                            # No la marcamos como procesada para que pueda intentarlo de nuevo después de más anotaciones

                except Exception as e:
                    print(f"[WATCHER] Error procesando nota {note_file}: {e}")

            # Esperar antes de volver a verificar
            time.sleep(3)

    except KeyboardInterrupt:
        print(f"\n[WATCHER] Observador detenido por el usuario")
    except Exception as e:
        print(f"\n[WATCHER] Error en el observador: {e}")
    finally:
        print(f"[WATCHER] Observador finalizado")