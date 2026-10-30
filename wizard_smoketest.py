"""Smoke test del wizard: AppTest recorre los 10 pasos y prueba el log a Obsidian."""
import json
import sys
from pathlib import Path

import streamlit.testing.v1 as stt

HERE = Path(__file__).parent
STEPS = None

at = stt.AppTest.from_file(str(HERE / "wizard.py"), default_timeout=120)
at.run()
assert not at.exception, f"excepción en la carga inicial: {at.exception}"
print("paso inicial OK")

# recorre todos los pasos del radio de la BARRA LATERAL (at.radio incluye también
# los radios del cuerpo: el paso 5 tiene el de imagen de fondo y el modo de bordes)
radio = at.sidebar.radio[0]
ops = list(radio.options)
for i, opname in enumerate(ops):
    at.sidebar.radio[0].set_value(opname).run()
    assert not at.exception, f"excepción en paso '{opname}': {at.exception}"
    print(f"paso {i} '{opname}' OK")

# navegación del Home: pulsar un tile debe llevar directo a esa sección
at = stt.AppTest.from_file(str(HERE / "wizard.py"), default_timeout=120)
at.run()
assert not at.exception, at.exception
_t = [b for b in at.button if b.key and b.key.startswith("tile_") and "4 · Escanear" in b.key]
assert _t, "no existe el tile de navegación «Escaneo» en el Home"
_t[0].click().run()
assert not at.exception, at.exception
assert at.sidebar.radio[0].value == "4 · Escanear", at.sidebar.radio[0].value
print("Home: tile «Escanear» navega al paso 4 OK")

# obsidian_log_session con metadatos reales de la simulación
meta_path = HERE / "sim_output" / "convex3" / "calibration.json"
if meta_path.exists():
    sys.path.insert(0, str(HERE))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    import wizard
    msg = wizard.obsidian_log_session(meta)
    print("obsidian:", msg)
    assert "No se pudo" not in msg, msg
    reg = HERE.parent / "ObsidianVault" / "LibreWarp" / "Registro de calibraciones.md"
    assert any("proyectores | mapas OK" in l for l in reg.read_text(encoding="utf-8").splitlines()), \
        "la fila no se insertó en el registro"
    print("registro Obsidian: fila insertada OK")
else:
    print("(sin sim_output/convex3: obsidian no probado)")

print("SMOKETEST: TODOS LOS PASOS OK")
