"""Crea el ZIP de entrega para Google Classroom: entrega/practica2-grupo-01-equipo-07.zip

Contenido: documento de diseño (PDF), notebook ejecutado (.ipynb), fuente .py del notebook y README.
Antes de empaquetar verifica que el documento no tenga datos pendientes (integrantes, cédula, video).

Uso:  python informe/empaquetar_entrega.py
"""
import sys
import zipfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
NOMBRE_ZIP = "practica2-grupo-01-equipo-07.zip"
ARCHIVOS = {
    "informe/Documento_Diseno_Grupo01_Equipo07.pdf": "Documento_Diseno_Grupo01_Equipo07.pdf",
    "Practica2_Grupo01_Equipo07.ipynb": "codigo/Practica2_Grupo01_Equipo07.ipynb",
    "src/practica2.py": "codigo/practica2.py",
    "README.md": "README.md",
}

html = (RAIZ / "informe" / "Documento_Diseno_Grupo01_Equipo07.html").read_text(encoding="utf-8")
pendientes = html.count('class="pendiente"')
if pendientes:
    print(f"AVISO: el documento tiene {pendientes} dato(s) pendiente(s) resaltados (integrantes, cédula o video).")
    print("Complete INTEGRANTES, CEDULA y URL_VIDEO en informe/generar_informe.py, regenere el PDF y vuelva a empaquetar.")
    if "--forzar" not in sys.argv:
        sys.exit("ZIP no creado. Use --forzar para crear un borrador de todos modos.")

destino = RAIZ / "entrega" / NOMBRE_ZIP
destino.parent.mkdir(exist_ok=True)
with zipfile.ZipFile(destino, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for origen, nombre in ARCHIVOS.items():
        z.write(RAIZ / origen, nombre)
print(f"ZIP creado: {destino} ({destino.stat().st_size / 1e6:.1f} MB)")
for nombre in ARCHIVOS.values():
    print("  -", nombre)
