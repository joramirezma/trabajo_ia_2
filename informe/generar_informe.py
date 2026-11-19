"""Genera el documento de diseño (HTML y PDF) del Trabajo Práctico 2 a partir de resultados/.

Uso:  python informe/generar_informe.py
El PDF se imprime con Microsoft Edge (o Chrome) en modo headless.
"""
import ast
import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
TAB = RAIZ / "resultados" / "tablas"
FIG = "../resultados/figuras"            # ruta relativa desde informe/
SALIDA_HTML = RAIZ / "informe" / "Documento_Diseno_Grupo01_Equipo07.html"
SALIDA_PDF = RAIZ / "informe" / "Documento_Diseno_Grupo01_Equipo07.pdf"

ALGS = ["Árbol de decisión", "KNN", "SVM", "Red neuronal"]
CLASES = ["Low", "Moderate", "High", "Severe"]
PENDIENTE = '<span class="pendiente">{}</span>'

# ============================================================================
# DATOS DEL EQUIPO — completar antes de generar la versión final del documento.
# Si un campo queda vacío, aparece resaltado en amarillo como pendiente.
# ============================================================================
INTEGRANTES = []        # p. ej. ["Nombre Apellido 1", "Nombre Apellido 2", "Nombre Apellido 3", "Nombre Apellido 4"]
CEDULA = ""             # cédula completa del integrante cuyos dígitos (1 y 9) definen registros y semilla
URL_VIDEO = ""          # enlace de YouTube del video pitch


def dato(valor, texto_pendiente):
    """Devuelve el valor si está completo o el marcador amarillo de pendiente."""
    return valor if valor else PENDIENTE.format(texto_pendiente)


def integrantes_html():
    if INTEGRANTES:
        return "<br>".join(INTEGRANTES)
    return "<br>".join(PENDIENTE.format(f"[Nombre integrante {i}]") for i in range(1, 5))


def video_html():
    return f'<a href="{URL_VIDEO}">{URL_VIDEO}</a>' if URL_VIDEO else PENDIENTE.format("[Enlace del video]")


def leer(nombre, **kw):
    return pd.read_csv(TAB / f"{nombre}.csv", **kw)


def f4(x):
    return f"{x:.4f}"


def figura(nombre, pie, ancho=100):
    return (f'<figure style="width:{ancho}%"><img src="{FIG}/{nombre}.png" alt="{pie}">'
            f"<figcaption>{pie}</figcaption></figure>")


def tabla_html(df, clase="", indice=True, fmt=None):
    fmt = fmt or {}
    df = df.copy()
    for c, f in fmt.items():
        df[c] = df[c].map(f)
    if indice:
        df = df.reset_index().rename(columns={"index": ""})
    return df.to_html(index=False, classes=f"tabla {clase}", border=0, escape=False, na_rep="N/D")


# ----------------------------------------------------------------------------- datos
res = leer("resultados_supervisado_largo", index_col=0)
fig3 = leer("figura3_no_supervisado", index_col=0)
det_ns = leer("detalle_no_supervisado", index_col=0)
roc = leer("roc_pr_caso8", index_col=0)
por_clase = leer("metricas_por_clase", index_col=0)
resumen_casos = leer("resumen_casos", index_col=0)
particion = leer("particion", index_col=0)
red_cfg = leer("configuracion_red", index_col=0)
perfil5 = leer("perfil_clusters_caso5", index_col=0)
perfil_db8 = leer("perfil_dbscan_caso8", index_col=0)
sens = leer("sensibilidad_dbscan", index_col=0)
pred3 = leer("predicciones_3_registros_supervisado")
asig3 = leer("asignaciones_3_registros_no_supervisado", index_col=0)
iny = leer("outliers_inyectados_caso8", index_col=0)
f1_ed, f1_out, f1_bal = (leer(f"f1_medio_{n}", index_col=0) for n in ("ed", "outliers", "balanceo"))

# Hiperparámetros elegidos por caso (salida del notebook, sección 11)
HIPER = pd.DataFrame(
    [(1, 4, 9, 100), (2, 11, 1, 100), (3, 10, 3, 10), (4, 7, 1, 10),
     (5, 4, 5, 100), (6, 11, 3, 100), (7, 10, 3, 1), (8, 7, 3, 100)],
    columns=["caso", "Árbol: max_depth (Gini)", "KNN: k", "SVM: C"]).set_index("caso")


# ----------------------------------------------------------------------------- tabla figura 1
def tabla_figura1():
    piv = res.pivot_table(index=["caso", "ED", "OUTLIERS", "BALANCEO"], columns="algoritmo",
                          values=["accuracy", "precision", "recall", "f1"])
    maximos = {(a, m): piv[(m, a)].max() for a in ALGS for m in ("accuracy", "precision", "recall", "f1")}
    h = ['<table class="tabla fig1"><thead><tr><th rowspan="2">#</th><th rowspan="2">Normalización</th>'
         '<th rowspan="2">Outliers</th><th rowspan="2">Balanceo</th>']
    h += [f'<th colspan="4" class="g{i}">{a}</th>' for i, a in enumerate(ALGS)]
    h.append("</tr><tr>")
    h += ["<th>Acc</th><th>Prec</th><th>Rec</th><th>F1</th>" for _ in ALGS]
    h.append("</tr></thead><tbody>")
    for (caso, ed, out, bal), fila in piv.iterrows():
        h.append(f"<tr><td>{caso}</td><td>CC(SI) y ED({ed})</td><td>{out}</td><td>{bal}</td>")
        for a in ALGS:
            for m in ("accuracy", "precision", "recall", "f1"):
                v = fila[(m, a)]
                cls = ' class="max"' if abs(v - maximos[(a, m)]) < 1e-9 else ""
                h.append(f"<td{cls}>{v:.4f}</td>")
        h.append("</tr>")
    h.append("</tbody></table>")
    return "".join(h)


def tabla_figura3():
    h = ['<table class="tabla fig3"><thead><tr><th rowspan="2">#</th><th rowspan="2">Normalización</th>'
         '<th rowspan="2">Outliers</th><th rowspan="2">Balanceo</th><th colspan="2">K-MEANS</th>'
         '<th>DBSCAN</th><th colspan="3">Configuración y validación externa</th></tr>'
         "<tr><th>Inertia</th><th>Silhouette</th><th>Silhouette</th><th>K óptimo (K-Means)</th>"
         "<th>ε usado · clusters · % ruido (DBSCAN)</th><th>ARI K-Means / DBSCAN</th></tr></thead><tbody>"]
    for caso, r in fig3.iterrows():
        d = det_ns.loc[caso]
        h.append(f"<tr><td>{caso}</td><td>CC(SI) y ED({r.ED})</td><td>{r.OUTLIERS}</td><td>{r.BALANCEO}</td>"
                 f"<td>{r['K-MEANS INERTIA']:,.1f}</td><td>{r['K-MEANS SILHOUETTE']:.4f}</td>"
                 f"<td>{r['DBSCAN SILHOUETTE']:.4f}</td><td>{int(d.K_optimo)}</td>"
                 f"<td>{d.db_eps:.3g} · {int(d.db_clusters)} · {d['db_%ruido']:.1f} %</td>"
                 f"<td>{d.km_ARI:.3f} / {d.db_ARI:.3f}</td></tr>")
    h.append("</tbody></table>")
    return "".join(h)


def tabla_por_clase_caso8():
    t = por_clase[por_clase.caso == 8].pivot_table(index="algoritmo", columns="clase",
                                                   values=["precision", "recall", "f1"]).reindex(ALGS)
    h = ['<table class="tabla"><thead><tr><th rowspan="2">Algoritmo</th>']
    h += [f'<th colspan="3">{c}</th>' for c in CLASES]
    h.append("</tr><tr>" + "".join("<th>Prec</th><th>Rec</th><th>F1</th>" for _ in CLASES) + "</tr></thead><tbody>")
    for a, fila in t.iterrows():
        h.append(f"<tr><td>{a}</td>" + "".join(
            f"<td>{fila[('precision', c)]:.3f}</td><td>{fila[('recall', c)]:.3f}</td><td>{fila[('f1', c)]:.3f}</td>"
            for c in CLASES) + "</tr>")
    h.append("</tbody></table>")
    return "".join(h)


# ----------------------------------------------------------------------------- valores para el texto
def val(caso, alg, m="f1"):
    return res[(res.caso == caso) & (res.algoritmo == alg)][m].iloc[0]


def rec(caso, alg, clase):
    return por_clase[(por_clase.caso == caso) & (por_clase.algoritmo == alg) & (por_clase.clase == clase)].recall.iloc[0]


def prec(caso, alg, clase):
    return por_clase[(por_clase.caso == caso) & (por_clase.algoritmo == alg) & (por_clase.clase == clase)].precision.iloc[0]


f1max = res.loc[res.groupby("algoritmo").f1.idxmax()].set_index("algoritmo").reindex(ALGS)
mejor_global = res.loc[res.f1.idxmax()]
svm_385_low = int(((pred3.algoritmo == "SVM") & (pred3["reg 385 (real Severe)"] == "Low")).sum())
aciertos_3 = int(sum((pred3[c] == c.split("real ")[1].rstrip(")")).sum()
                     for c in pred3.columns if c.startswith("reg ")))
n_out_filas = iny.fila.nunique()
s8 = sens[sens.caso == 8].set_index("factor")
p5 = perfil5

TRIVIAL_F1 = (2 * 0.7 / 1.7) / 4      # clasificador que siempre predice Low en un test 70/10/10/10

# ----------------------------------------------------------------------------- variables del dataset
VARIABLES = [
    ("city_zone", "Categórica nominal", "Zona de la ciudad (6): Suburban North, Financial District, Downtown Core, Tech Park, Residential West, Industrial East"),
    ("road_type", "Categórica nominal", "Tipo de vía (4): Arterial, Collector, Highway, Local Street"),
    ("peak_period", "Categórica nominal", "Franja horaria (5): Night, Off-Peak, Morning Peak, Midday, Evening Peak"),
    ("weather_condition", "Categórica nominal", "Clima (4): Clear, Rain, Heavy Rain, Fog"),
    ("iot_sensor_health", "Categórica nominal (binaria)", "Estado del sensor IoT: Optimal / Degraded"),
    ("road_condition", "Categórica ordinal", "Estado del pavimento: Good < Wet < Poor"),
    ("latitude, longitude", "Numérica continua", "Ubicación de la intersección"),
    ("lanes, speed_limit", "Numérica discreta", "Número de carriles (2–4) y límite de velocidad (40/60/90 km/h)"),
    ("vehicle_count", "Numérica continua", "Vehículos contados en la hora"),
    ("average_speed", "Numérica continua", "Velocidad media (km/h)"),
    ("heavy_vehicle_count, motorcycle_count, public_transport_count", "Numérica continua", "Conteos por tipo de vehículo"),
    ("traffic_density", "Numérica continua", "Densidad de tráfico (vehículos por km de carril)"),
    ("queue_length, average_wait_time", "Numérica continua", "Longitud de la cola (m) y tiempo medio de espera (min)"),
    ("hour, day_of_week", "Numérica discreta", "Hora del día (0–23) y día de la semana (0–6)"),
    ("is_weekend, is_holiday, rush_hour", "Binaria", "Indicadores de fin de semana, festivo y hora punta"),
    ("temperature, rainfall_mm, visibility_km, air_quality_index", "Numérica continua", "Condiciones ambientales"),
    ("nearby_school, nearby_hospital, nearby_market", "Binaria", "Equipamientos cercanos a la intersección"),
    ("construction_activity, public_event, accident_reported, emergency_vehicle_detected", "Binaria", "Eventos que afectan el tráfico"),
    ("parking_occupancy", "Numérica continua", "Ocupación de parqueaderos cercanos (%)"),
    ("traffic_camera_count", "Numérica discreta", "Cámaras de tráfico en la intersección (1–4)"),
    ("signal_cycle_seconds, green_light_duration", "Numérica discreta", "Ciclo semafórico y duración de la luz verde (s)"),
    ("emission_estimate, fuel_waste_estimate", "Numérica continua", "Emisiones estimadas y combustible desperdiciado"),
    ("<b>congestion_level</b>", "<b>Objetivo (ordinal, 4 clases)</b>", "<b>Low, Moderate, High, Severe</b>"),
]
tabla_vars = "".join(f"<tr><td><code>{a}</code></td><td>{b}</td><td>{c}</td></tr>" for a, b, c in VARIABLES)

CSS = """
@page { size: A4; margin: 16mm 15mm 18mm 15mm; }
:root { --tinta:#0b0b0b; --tinta2:#52514e; --muted:#898781; --linea:#e1e0d9; --azul:#2a78d6; --azul-osc:#184f95;
        --fondo:#fcfcfb; --suave:#f3f2ee; }
* { box-sizing: border-box; }
body { font-family: "Segoe UI", system-ui, -apple-system, sans-serif; color: var(--tinta); background: #fff;
       font-size: 10.2pt; line-height: 1.45; margin: 0; }
h1 { font-size: 19pt; color: var(--azul-osc); margin: 8mm 0 4mm; padding-bottom: 2mm;
     border-bottom: 2px solid var(--azul); }
h2 { font-size: 13.5pt; color: var(--azul-osc); margin: 6mm 0 2mm; page-break-after: avoid; }
h3 { font-size: 11.2pt; margin: 4mm 0 1.5mm; page-break-after: avoid; }
p, li { text-align: justify; }
ul, ol { padding-left: 6mm; margin: 1mm 0 2mm; }
code { font-family: Consolas, monospace; font-size: 8.8pt; background: var(--suave); padding: 0 1mm; border-radius: 2px; }
figure { margin: 3mm auto; page-break-inside: avoid; text-align: center; }
figure img { width: 100%; border: 1px solid var(--linea); border-radius: 3px; }
figcaption { font-size: 8.6pt; color: var(--tinta2); margin-top: 1mm; text-align: center; }
.tabla { border-collapse: collapse; width: 100%; font-size: 8.4pt; margin: 2mm 0 3mm; page-break-inside: avoid; }
.tabla th { background: var(--suave); color: var(--tinta); font-weight: 600; padding: 1.2mm 1.2mm; border-bottom: 1px solid #c3c2b7; }
.tabla td { padding: 0.9mm 1.2mm; border-bottom: 1px solid var(--linea); text-align: center; }
.tabla td:first-child { text-align: left; }
.fig1 { font-size: 7.4pt; } .fig1 td, .fig1 th { padding: 0.8mm 0.6mm; }
.fig1 td.max { font-weight: 700; color: var(--azul-osc); background: #e8f1fc; }
.fig1 th.g0 { border-bottom: 3px solid #2a78d6; } .fig1 th.g1 { border-bottom: 3px solid #eb6834; }
.fig1 th.g2 { border-bottom: 3px solid #1baf7a; } .fig1 th.g3 { border-bottom: 3px solid #eda100; }
.vars td { text-align: left; } .vars td:nth-child(2) { white-space: nowrap; }
.nota { background: #eef5fd; border-left: 3px solid var(--azul); padding: 2mm 3mm; margin: 2mm 0 3mm; font-size: 9.4pt; }
.alerta { background: #fff6e0; border-left: 3px solid #eda100; padding: 2mm 3mm; margin: 2mm 0 3mm; font-size: 9.4pt; }
.pendiente { background: #fff1b8; padding: 0 1.2mm; border-radius: 2px; font-weight: 600; }
.portada { height: 255mm; display: flex; flex-direction: column; justify-content: space-between; text-align: center; }
h1.salto { page-break-before: always; margin-top: 0; }
.portada p { text-align: center; }
.portada h1 { page-break-before: avoid; border: none; font-size: 24pt; margin-top: 18mm; }
.portada .sub { font-size: 13pt; color: var(--tinta2); }
.portada table { margin: 0 auto; font-size: 10.5pt; border-collapse: collapse; min-width: 120mm; }
.portada td { padding: 1.5mm 4mm; border-bottom: 1px solid var(--linea); text-align: left; }
.flujo { page-break-inside: avoid; display: flex; flex-wrap: wrap; gap: 2mm; align-items: stretch; margin: 2mm 0 4mm; }
.flujo div { flex: 1 1 28mm; background: var(--suave); border-top: 3px solid var(--azul); padding: 2mm; font-size: 8.4pt;
             border-radius: 2px; }
.flujo b { display: block; color: var(--azul-osc); margin-bottom: 0.8mm; }
.dos { display: flex; gap: 4mm; } .dos > * { flex: 1; }
.toc li { text-align: left; }
"""

# ----------------------------------------------------------------------------- documento
html = f"""<!doctype html><html lang="es"><head><meta charset="utf-8">
<title>Documento de diseño · Práctica 2 · Grupo 01 Equipo 07</title><style>{CSS}</style></head><body>

<section class="portada">
<div>
<p class="sub">Universidad Nacional de Colombia · Sede Medellín<br>Introducción a la Inteligencia Artificial — 3010476</p>
<h1>Trabajo Práctico 2<br>Aprendizaje de Máquinas</h1>
<p class="sub">Clasificación y agrupamiento del nivel de congestión vehicular<br>con datos de movilidad urbana inteligente (IoT)</p>
<p class="sub">Documento de diseño</p>
</div>
<div>
<table>
<tr><td><b>Grupo / Equipo</b></td><td>Grupo 01 — Equipo 07</td></tr>
<tr><td><b>Profesor</b></td><td>Jaime Alberto Guzmán Luna</td></tr>
<tr><td><b>Integrantes</b></td><td>{integrantes_html()}</td></tr>
<tr><td><b>Cédula utilizada</b></td><td>{dato(CEDULA, "[Cédula completa del integrante elegido]")}<br>
<small>últimos dígitos ≠ 0: 1 y 9 → 6350 registros, random_state = 19</small></td></tr>
<tr><td><b>Video pitch (YouTube)</b></td><td>{video_html()}</td></tr>
<tr><td><b>Dataset original</b></td><td><a href="https://www.kaggle.com/datasets/mobeenfatimah/cityflow-smart-urban-mobility-and-traffic-iot">kaggle.com/datasets/mobeenfatimah/<br>cityflow-smart-urban-mobility-and-traffic-iot</a></td></tr>
<tr><td><b>Código</b></td><td><code>Practica2_Grupo01_Equipo07.ipynb</code> (Google Colab)</td></tr>
</table>
</div>
<p class="sub">Medellín, 2026</p>
</section>

<h1 class="salto">Contenido</h1>
<ol class="toc">
<li>Descripción general y objetivos</li>
<li>Justificación del dataset</li>
<li>Arquitectura de la solución y reproducibilidad</li>
<li>Módulo de aprendizaje supervisado: preprocesamiento, modelos, resultados (figura 1), análisis, curvas ROC/PR y visualizaciones</li>
<li>Módulo de aprendizaje no supervisado: datasets (figura 2), K-Means y DBSCAN, resultados (figura 3), análisis y visualización</li>
<li>Conclusiones generales</li>
<li>Video pitch</li>
<li>Anexos: visualizaciones de todos los casos</li>
</ol>

<!-- ======================================================================= 1 -->
<h1>1. Descripción general y objetivos</h1>
<p>El trabajo implementa y compara técnicas de aprendizaje de máquinas sobre un dataset abierto de
<b>movilidad urbana inteligente</b>. La tarea supervisada es predecir el <b>nivel de congestión</b> de una intersección
(<i>Low, Moderate, High, Severe</i>) a partir de mediciones de sensores IoT, cámaras, clima y contexto urbano. Después se
retira la etiqueta y se buscan agrupamientos naturales con K-Means y DBSCAN.</p>
<p><b>Motivación del dominio.</b> Anticipar la congestión permite a una central de tráfico reprogramar semáforos,
redirigir vehículos, priorizar vehículos de emergencia y reducir emisiones. Por eso los errores no cuestan lo mismo:
no detectar una congestión severa es mucho más costoso que una falsa alarma (ver sección 4.6).</p>
<ul>
<li><b>Supervisado:</b> Árboles de decisión, K vecinos más cercanos (KNN), Máquinas de vectores de soporte (SVM) y
Redes neuronales (Keras), en los 8 casos de preprocesamiento de la figura 1. Se combinan escalado sí/no, outliers sí/no
y balanceo sí/no.</li>
<li><b>No supervisado:</b> K-Means y DBSCAN en los mismos 8 casos (figura 2), con el método del codo, el silhouette,
el gráfico de k-distancia y validación externa contra la etiqueta original.</li>
</ul>

<!-- ======================================================================= 2 -->
<h1 class="salto">2. Justificación del dataset</h1>
<p><b>Fuente:</b> Kaggle — <i>CityFlow: Smart Urban Mobility and Traffic IoT</i>
(<code>mobeenfatimah/cityflow-smart-urban-mobility-and-traffic-iot</code>). Es un archivo CSV de acceso libre
(<code>smart_city_traffic_mobility.csv</code>) con <b>204 000 registros horarios y 47 columnas</b>, que se descarga
automáticamente desde el notebook con <code>kagglehub</code>.</p>
<h2>2.1 Verificación de requisitos</h2>
<table class="tabla"><thead><tr><th>Requisito del enunciado</th><th>Dataset</th><th>Cumple</th></tr></thead><tbody>
<tr><td>Al menos 5000 registros</td><td>204 000 registros (se usa un subconjunto de 6350)</td><td>✔</td></tr>
<tr><td>Al menos 4 clases en la variable objetivo</td><td><code>congestion_level</code>: Low, Moderate, High, Severe</td><td>✔</td></tr>
<tr><td>Variables numéricas y categóricas</td><td>34 numéricas y 6 categóricas (5 nominales y 1 ordinal) tras la limpieza</td><td>✔</td></tr>
<tr><td>Dominio propio del equipo</td><td>Movilidad urbana / tráfico IoT</td><td>✔</td></tr>
<tr><td>Subconjunto: 5000 + (último dígito ≠ 0 de la cédula × 150)</td><td>Dígito 9 → 5000 + 1350 = <b>6350 registros</b></td><td>✔</td></tr>
<tr><td>Semilla: dos últimos dígitos ≠ 0 de la cédula, distinta de 42</td><td><b>random_state = 19</b></td><td>✔</td></tr>
</tbody></table>
<div class="nota">Cédula utilizada: {dato(CEDULA, "[cédula completa]")}. Sus dos últimos dígitos distintos de cero
son 1 y 9. La semilla 19 se usa en el muestreo del subconjunto, la partición, los modelos, SMOTENC, la inyección de outliers,
las inicializaciones de K-Means y las de la red neuronal.</div>

<h2>2.2 Descripción de las variables</h2>
<table class="tabla vars"><thead><tr><th>Variable(s)</th><th>Tipo</th><th>Descripción</th></tr></thead>
<tbody>{tabla_vars}</tbody></table>
<h2>2.3 Variables descartadas</h2>
<ul>
<li><b><code>congestion_score</code> — fuga de información.</b> El análisis exploratorio mostró que la etiqueta se deriva
por umbrales exactos de este puntaje (Low &lt; 25 ≤ Moderate &lt; 55 ≤ High &lt; 80 ≤ Severe). Un árbol de profundidad 3
que solo usa esta variable obtiene 100 % de accuracy. Conservarla habría vuelto trivial el problema.</li>
<li><b><code>traffic_flow_rate</code></b>: es un duplicado exacto de <code>vehicle_count</code> (correlación 1.0).</li>
<li><b><code>record_id</code>, <code>timestamp</code>, <code>road_id</code>, <code>intersection_id</code></b>: identificadores o
información redundante. La hora y el día ya están en <code>hour</code>, <code>day_of_week</code> y <code>peak_period</code>.
La ubicación ya está en <code>city_zone</code> y en las coordenadas.</li>
</ul>
<p>Quedan <b>40 variables predictoras</b>. Tras el One-Hot Encoding resultan <b>56 columnas</b> de entrada para los modelos.</p>

<!-- ======================================================================= 3 -->
<h1 class="salto">3. Arquitectura de la solución y reproducibilidad</h1>
<p>La solución es un único notebook de Google Colab (<code>Practica2_Grupo01_Equipo07.ipynb</code>) dividido en módulos.
Cada paso es una función documentada y reutilizable.</p>
<div class="flujo">
<div><b>0 · Instalación</b>Celda <code>%pip</code> con kagglehub e imbalanced-learn</div>
<div><b>1 · Configuración</b>Cédula → 6350 registros y semilla 19; <code>fijar_semillas()</code></div>
<div><b>Módulo común</b>Carga, selección de variables, subconjunto, EDA, nulos, partición, outliers, SMOTENC, CC/ED y los 8 casos</div>
<div><b>Módulo supervisado</b>Árbol, KNN, SVM, red neuronal; figura 1, gráficas F1, matrices, ROC/PR y visualizaciones</div>
<div><b>Módulo no supervisado</b>K-Means, DBSCAN; figura 3, silhouette, k-distancia, PCA 2D y perfiles</div>
</div>
<table class="tabla"><thead><tr><th>Función</th><th>Responsabilidad</th></tr></thead><tbody>
<tr><td><code>cargar_dataset</code>, <code>seleccionar_variables</code>, <code>construir_subconjunto</code></td><td>Descarga, limpieza y muestreo del subconjunto 70/10/10/10 de 6350 registros</td></tr>
<tr><td><code>resumen_variables</code>, <code>imputar_nulos</code>, <code>particionar</code></td><td>EDA, imputación (mediana/moda) y partición estratificada 70/15/15</td></tr>
<tr><td><code>TratadorOutliersZ</code>, <code>inyectar_outliers</code></td><td>Detección Z-score, recorte e inyección documentada del 5 % de outliers</td></tr>
<tr><td><code>balancear_smotenc</code></td><td>Balanceo de clases con SMOTENC</td></tr>
<tr><td><code>construir_transformador</code>, <code>preparar_caso</code></td><td>One-Hot/Ordinal + StandardScaler opcional (fit solo con entrenamiento) y armado de cada caso</td></tr>
<tr><td><code>entrenar_arbol</code>, <code>entrenar_knn</code>, <code>entrenar_svm</code>, <code>RedNeuronal</code></td><td>Entrenamiento y selección de hiperparámetros de cada técnica</td></tr>
<tr><td><code>metricas</code>, <code>puntajes</code>, <code>graficar_*</code>, <code>barras_agrupadas</code></td><td>Evaluación y gráficas</td></tr>
<tr><td><code>barrido_kmeans</code>, <code>codo</code>, <code>k_distancias</code>, <code>evaluar_dbscan</code>, <code>asignar_dbscan</code></td><td>Agrupamiento, selección de K y ε, y asignación de registros nuevos</td></tr>
</tbody></table>
<p><b>Reproducibilidad:</b> semilla 19 en Python, NumPy y TensorFlow (<code>set_random_seed</code> y
<code>enable_op_determinism</code>), y en el muestreo, la partición, los modelos, SMOTENC, la inyección de outliers y K-Means.
El notebook se ejecutó de principio a fin sin errores. Una segunda ejecución completa produjo las mismas tablas de
resultados. Todas las figuras y tablas se guardan en <code>resultados/</code>.</p>
<div class="nota"><b>"3 datos de prueba".</b> El enunciado pide "la evaluación de 3 datos de prueba" por caso y técnica.
Se interpretó así: además de la evaluación completa sobre el conjunto de prueba, se muestran las predicciones de cada
modelo, en cada caso, sobre <b>los mismos tres registros de prueba</b> (uno <i>Low</i>, uno <i>High</i> y uno
<i>Severe</i>, elegidos con la semilla). En el módulo no supervisado se muestra el cluster asignado a esos tres registros.</div>

<!-- ======================================================================= 4 -->
<h1 class="salto">4. Módulo de aprendizaje supervisado</h1>
<h2>4.1 Análisis exploratorio y valores nulos</h2>
<ul>
<li><b>Dimensiones del subconjunto:</b> 6350 registros × 41 columnas (40 predictoras y el objetivo).</li>
<li><b>Tipos:</b> 5 nominales, 1 ordinal, 10 binarias, 7 numéricas discretas y 17 numéricas continuas (tabla de la
sección 2.2).</li>
<li><b>Nulos:</b> 0 en todas las variables. Aun así se incluye la función <code>imputar_nulos</code> (mediana para
numéricas por ser robusta a outliers, moda para categóricas) para que el preprocesamiento sea robusto.</li>
<li><b>Distribución de clases:</b> el subconjunto se construyó desbalanceado, 70 % Low y 10 % para cada una de las demás
(4445 / 635 / 635 / 635).</li>
<li><b>Correlación con el nivel de congestión</b> (codificado 0–3): las variables más asociadas son
<code>traffic_density</code> (r = 0.93), <code>vehicle_count</code> (0.88), <code>green_light_duration</code> (0.86),
los conteos por tipo de vehículo (0.79–0.82) y <code>emission_estimate</code> (0.77). La alta correlación de la duración
del verde sugiere que la semaforización es adaptativa: alarga el verde cuando hay congestión. Hay pares casi colineales (<code>lanes</code>–<code>speed_limit</code>,
<code>queue_length</code>–<code>average_wait_time</code>, <code>emission_estimate</code>–<code>fuel_waste_estimate</code>). Se
conservaron porque no afectan a los árboles y su efecto en los demás modelos forma parte del análisis.</li>
</ul>
<div class="dos">{figura("eda_distribucion_clases", "Distribución de clases del subconjunto (desbalanceado 70/10/10/10).")}
{figura("eda_histogramas", "Distribución de las variables continuas por clase.")}</div>
{figura("eda_correlacion", "Matriz de correlación de Pearson (numéricas y objetivo codificado ordinalmente).", 82)}
<p>Las estadísticas descriptivas completas están en el notebook (sección 4) y en <code>resultados/tablas/eda_descriptivas.csv</code>.
Varias variables (<code>queue_length</code>, <code>average_wait_time</code>, <code>emission_estimate</code>, <code>rainfall_mm</code>)
tienen colas muy largas, y casi todas las de volumen son bimodales (tráfico fluido frente a congestionado).</p>

<h2>4.2 Partición y preprocesamiento</h2>
<p><b>Partición estratificada</b> (semilla 19) en entrenamiento, validación y prueba:</p>
{tabla_html(particion)}
<p>El conjunto de <b>prueba es el mismo en los 8 casos</b> y no se modifica. Validación y prueba solo reciben la
transformación CC/ED ajustada con el entrenamiento. Todos los demás procedimientos (tratamiento de outliers, inyección,
SMOTENC, codificación y escalado) se ajustan <b>únicamente con el entrenamiento</b>.</p>

<h3>Conversión de categóricas (CC)</h3>
<ul>
<li><b>One-Hot Encoding</b> para las nominales (<code>city_zone</code>, <code>road_type</code>, <code>peak_period</code>,
<code>weather_condition</code>, <code>iot_sensor_health</code>). No tienen orden, y un código entero haría creer a KNN, SVM y la red
que, por ejemplo, "Tech Park" está "entre" dos zonas. Se usa <code>handle_unknown="ignore"</code>.</li>
<li><b>Codificación ordinal</b> para <code>road_condition</code> con el orden Good &lt; Wet &lt; Poor, porque sí existe una
gradación real del estado del pavimento.</li>
<li>Las binarias ya están en 0/1.</li>
</ul>
<h3>Escalado (ED)</h3>
<p>Se usa <b>StandardScaler</b> sobre todas las columnas que no son one-hot. Se prefirió frente a MinMaxScaler porque en los casos
con outliers MinMax comprime casi todos los datos en un rango muy estrecho: el máximo lo fija un valor atípico. La
estandarización es menos sensible a eso y es coherente con la detección por Z-score.</p>
<h3>Valores atípicos</h3>
<ul>
<li><b>Detección:</b> Z-score con |z| &gt; 3 sobre las 15 variables continuas de tráfico, ambientales y de emisiones. No se usó
IQR × 1.5 porque, al ser distribuciones bimodales y de cola larga, marcaba como atípico el 23.5 % de las filas, casi todas
registros reales de congestión severa. En el entrenamiento, el Z-score detectó <b>379 filas (8.53 %) con 1368 valores
atípicos</b>.</li>
<li><b>Sin outliers:</b> los valores detectados se <b>recortan</b> a media ± 3σ (<i>winsorización</i>). No se eliminan porque
eliminarlos borraría sobre todo registros <i>Severe</i> y sesgaría las clases.</li>
<li><b>Con outliers:</b> sobre el entrenamiento ya tratado se <b>inyectan outliers sintéticos</b> en exactamente el 5 % de los
registros. En cada registro elegido se modifican 1 o 2 variables continuas al azar con el valor media ± U(3.5, 6)·σ. La
dirección respeta el dominio: hacia arriba en conteos y tiempos, hacia abajo en visibilidad, en cualquier sentido para la
temperatura, y se excluye la ocupación de parqueaderos por ser un porcentaje acotado. En el caso 8 se alteraron
<b>{n_out_filas} registros ({len(iny)} valores)</b>, es decir, el 5.00 % de 12 444. La tabla completa está en
<code>outliers_inyectados_caso8.csv</code>.</li>
</ul>
<h3>Balanceo de clases</h3>
<ul>
<li><b>Desbalanceado</b> (BALANCEO = NO): el entrenamiento conserva la distribución 70/10/10/10 (3111 / 444 / 445 / 445).</li>
<li><b>Balanceado</b> (BALANCEO = SI): <b>SMOTENC</b> sobremuestrea las minoritarias hasta 3111 registros por clase
(12 444 en total).
<ul>
<li>Genera registros sintéticos interpolados entre vecinos de la misma clase, sin duplicar filas como el sobremuestreo
aleatorio, que favorece el sobreajuste.</li>
<li>No descarta datos: el submuestreo dejaría solo unas 1780 filas.</li>
<li>Su variante <i>NC</i> asigna a las categóricas, binarias y discretas de pocos valores la moda de los vecinos en lugar de
inventar valores intermedios.</li>
<li>Los vecinos se buscan con las numéricas estandarizadas, para que no dominen las de mayor magnitud.</li>
</ul></li>
<li>No se usa <code>class_weight</code>: así los cuatro algoritmos y el módulo no supervisado reciben exactamente los
mismos registros.</li>
</ul>
<h3>Orden de aplicación y casos resultantes</h3>
<p>Sobre el entrenamiento se aplica, en este orden:</p>
<ol>
<li>recorte de outliers reales;</li>
<li>SMOTENC (si BALANCEO = SI);</li>
<li>inyección del 5 % de outliers (si OUTLIERS = SI);</li>
<li>CC y ED ajustados con ese entrenamiento.</li>
</ol>
<p>La inyección va después del balanceo para que los atípicos sean exactamente el 5 % del entrenamiento final y SMOTE no los
propague. Los casos <i>i</i> e <i>i+4</i> comparten exactamente los mismos registros y solo difieren en el escalado.</p>
{tabla_html(resumen_casos.rename(columns={"n_train": "registros train", "%filas_outlier": "% filas atípicas", "n_variables": "variables"}))}

<h2>4.3 Modelos y selección de hiperparámetros</h2>
<p>Las métricas son accuracy y precisión, recall y F1 con <b>promedio macro</b> (cada clase pesa igual). Los hiperparámetros se
eligen con el conjunto de validación según el F1-macro, salvo k en KNN, que se elige por validación cruzada de 5 folds como
pide el enunciado.</p>
<ul>
<li><b>Árbol de decisión:</b> se recorre <code>max_depth</code> de 1 a 20 con los criterios Gini y Entropía. El modelo
solución usa Gini con la profundidad de mejor F1 en validación.</li>
<li><b>KNN:</b> k ∈ {{1, 3, …, 31}}. Se elige el k con mejor accuracy promedio en CV estratificada de 5 folds. <i>Detalle
metodológico:</i> en los casos balanceados, hacer la CV sobre el conjunto ya sobremuestreado filtra información, porque un
registro sintético de validación tiene a sus "padres" en el entrenamiento y la CV favorecería k = 1. Por eso el SMOTENC, la
inyección de outliers y el ajuste CC/ED se repiten <b>dentro de cada fold</b>, y la validación de cada fold contiene solo
registros reales.</li>
<li><b>SVM:</b> kernel RBF con <code>gamma="scale"</code> y C ∈ {{1, 10, 100}}.</li>
<li><b>Red neuronal:</b> descrita en la sección 4.4.</li>
</ul>
{tabla_html(HIPER)}
<p>Observaciones: en tres de los cuatro casos con outliers (3, 4 y 7) se eligió un C menor (10 o 1) que en los casos limpios
(100). Un margen más blando tolera mejor los puntos atípicos. Sin escalar, la CV de KNN prefiere k = 1 en los casos
balanceados, porque los registros sintéticos densifican las regiones de las clases minoritarias.</p>

<h2>4.4 Red neuronal: arquitectura y proceso de configuración</h2>
<p>Es un perceptrón multicapa en Keras (envuelto en la clase <code>RedNeuronal</code>, con interfaz fit/predict/predict_proba):</p>
<ul>
<li>entrada de 56 variables, capas densas ReLU y salida <i>softmax</i> de 4 neuronas;</li>
<li>pérdida <i>sparse categorical cross-entropy</i>, optimizador Adam (lr = 1e-3), lotes de 64 y hasta 200 épocas.</li>
</ul>
<p><b>Regularización</b> (técnicas vistas en clase, aplicadas en todos los casos):</p>
<ul>
<li>penalización L2 (λ = 1e-4) en las capas densas;</li>
<li>Dropout;</li>
<li>Batch Normalization (en las arquitecturas candidatas más profundas).</li>
</ul>
<p><b>Callbacks:</b></p>
<ul>
<li><code>EarlyStopping</code>: vigila <code>val_loss</code> con paciencia 15 y restaura los mejores pesos.</li>
<li><code>ReduceLROnPlateau</code>: multiplica el lr por 0.5 tras 5 épocas sin mejora (mínimo 1e-5).</li>
</ul>
<p><b>Proceso de configuración:</b> se compararon tres arquitecturas en el caso 8 (el más completo) con el conjunto de validación.
La de mejor F1-macro se fijó para los 8 casos, para que la comparación entre preprocesamientos sea justa.</p>
{tabla_html(red_cfg, fmt={"f1_val": f4, "acc_val": f4})}
<p>Se eligió la <b>arquitectura A</b>: Dense(64, ReLU, L2) → Dropout(0.2) → Dense(4, softmax), con 3908 parámetros.
Las redes más profundas no mejoran: el problema es casi separable con pocas variables (sobre todo <code>traffic_density</code>), y
la mayor capacidad solo acelera el sobreajuste, como muestra que el early stopping las detuviera en 25–27 épocas. Las curvas de
aprendizaje de los 8 casos están en el anexo A. En todas, la pérdida de validación se estabiliza y el early stopping
restaura el mejor punto antes de que se separe de la de entrenamiento.</p>

<!-- ------------------------------------------------------------------ resultados -->
<h2>4.5 Resultados — tabla de la figura 1</h2>
<p>Métricas sobre el conjunto de prueba (953 registros, 70/10/10/10). Precisión, recall y F1 con promedio macro. En negrita, el
máximo de cada métrica por algoritmo.</p>
{tabla_figura1()}
<p>Predicciones sobre los tres registros de prueba fijos: los modelos aciertan <b>{aciertos_3} de 96</b> predicciones
(8 casos × 4 técnicas × 3 registros). El registro <i>Low</i> (556) se clasifica bien siempre. El <i>Severe</i> (385) lo
clasifica como <i>Low</i> el SVM en {svm_385_low} de los 8 casos. Ese es justamente el tipo de error más costoso en el dominio
(sección 4.6). El árbol, KNN y la red lo aciertan en casi todos los casos. La tabla completa está en
<code>predicciones_3_registros_supervisado.csv</code> y en el notebook.</p>

<h3>Gráficas comparativas</h3>
<div class="dos">{figura("sup_f1_maximo", "F1-score máximo por algoritmo en los 8 casos.")}
{figura("sup_f1_medio_ed", "F1-score medio por normalización (escalado) y algoritmo.")}</div>
<div class="dos">{figura("sup_f1_medio_outliers", "F1-score medio por manejo de datos atípicos y algoritmo.")}
{figura("sup_f1_medio_balanceo", "F1-score medio por balanceo y algoritmo.")}</div>

<h2>4.6 Análisis de resultados</h2>
<p><b>Visión general.</b> El mejor F1-macro global lo obtiene el <b>{mejor_global.algoritmo} en el caso {int(mejor_global.caso)}
({mejor_global.f1:.4f})</b>. Le siguen muy de cerca el árbol de decisión ({f1max.loc["Árbol de decisión", "f1"]:.4f}) y la red
neuronal ({f1max.loc["Red neuronal", "f1"]:.4f}). KNN queda claramente por debajo ({f1max.loc["KNN", "f1"]:.4f} en su mejor
caso). El árbol, el SVM y la red superan el 0.93 de accuracy en los 8 casos, porque la densidad de tráfico separa muy bien
los niveles de congestión. Las diferencias aparecen en las clases minoritarias, sobre todo <i>Moderate</i>, la más difícil:
queda "entre" el tráfico fluido y el congestionado.</p>

<h3>Sensibilidad a la escala (KNN, SVM y redes neuronales frente a árboles)</h3>
<ul>
<li><b>Árbol: idéntico con y sin escalado</b> (F1 medio {f1_ed.loc["Árbol de decisión"].iloc[0]:.4f} en ambos). Es lo que
predice la teoría: un árbol divide por umbrales sobre una variable a la vez, y una transformación monótona como la
estandarización solo cambia el valor del umbral, no el orden de los registros.</li>
<li><b>KNN: el escalado lo perjudica mucho</b> (F1 medio {f1_ed.loc["KNN"].iloc[0]:.3f} → {f1_ed.loc["KNN"].iloc[1]:.3f}).
Sin escalar, la distancia euclidiana la dominan las variables de gran magnitud (<code>vehicle_count</code> con σ ≈ 750,
<code>queue_length</code>, <code>emission_estimate</code>, <code>traffic_density</code>). Esas son precisamente las más
predictivas: el árbol asigna cerca del 90 % de la importancia a <code>traffic_density</code>. Al estandarizar, las 56 columnas
pesan igual, incluidas 21 one-hot, 10 binarias, coordenadas y clima, casi irrelevantes para la congestión. La señal útil se
diluye en un espacio de alta dimensión (maldición de la dimensionalidad) y los vecinos dejan de parecerse en lo que importa.
El daño se concentra en las minoritarias: en el caso 5, el recall de <i>Moderate</i> cae a {rec(5, "KNN", "Moderate"):.2f} y el
de <i>High</i> a {rec(5, "KNN", "High"):.2f}.</li>
<li><b>SVM</b> pierde menos ({f1_ed.loc["SVM"].iloc[0]:.3f} → {f1_ed.loc["SVM"].iloc[1]:.3f}). El kernel RBF también usa
distancias, pero la maximización del margen y la elección de C compensan en parte.</li>
<li><b>Red neuronal</b> casi no cambia ({f1_ed.loc["Red neuronal"].iloc[0]:.3f} → {f1_ed.loc["Red neuronal"].iloc[1]:.3f}). Aprende
un peso por entrada, así que puede reescalar internamente cada variable, y Adam adapta el paso por parámetro.</li>
</ul>
<p><b>Conclusión sobre la escala:</b> los tres modelos basados en distancias o gradientes son sensibles a la escala y el árbol
no, como predice la teoría. Pero escalar no siempre es mejor. Hace que todas las variables "voten" igual, y si muchas son poco
informativas conviene acompañarlo de selección de variables o ponderación.</p>

<h3>Efecto de los outliers</h3>
<ul>
<li><b>Árbol: prácticamente inmune</b> ({f1_out.loc["Árbol de decisión"].iloc[0]:.4f} → {f1_out.loc["Árbol de decisión"].iloc[1]:.4f}).
Sus cortes dependen del orden de los valores, y un valor extremo cae simplemente en la rama "mayor que".</li>
<li><b>SVM: el más afectado</b> ({f1_out.loc["SVM"].iloc[0]:.3f} → {f1_out.loc["SVM"].iloc[1]:.3f}). Hay tres mecanismos:
<ul>
<li>los puntos extremos aumentan la varianza de los datos y por tanto reducen <code>gamma="scale"</code>, lo que suaviza la frontera;</li>
<li>quedan aislados y pueden convertirse en vectores de soporte que deforman el margen;</li>
<li>en los casos escalados, el StandardScaler ajustado con outliers comprime al resto de los datos.</li>
</ul>
Por eso la validación eligió C más pequeños.</li>
<li><b>Red neuronal</b> pierde un punto ({f1_out.loc["Red neuronal"].iloc[0]:.3f} → {f1_out.loc["Red neuronal"].iloc[1]:.3f}).
<b>KNN</b> casi nada ({f1_out.loc["KNN"].iloc[0]:.3f} → {f1_out.loc["KNN"].iloc[1]:.3f}): un registro con un valor extremo queda
lejos de todos y rara vez es vecino de un punto de prueba.</li>
</ul>

<h3>Efecto del desbalance y por qué el accuracy engaña</h3>
<p>Con un conjunto de prueba 70/10/10/10, un clasificador trivial que siempre responda "Low" obtiene <b>70 % de accuracy</b>, pero
su F1-macro es solo <b>{TRIVIAL_F1:.3f}</b>, porque no detecta ninguna congestión. Un caso real de la tabla: KNN en el caso 5
tiene accuracy {val(5, "KNN", "accuracy"):.3f}, que parece bueno, pero deja sin detectar la mitad de las congestiones
<i>Moderate</i> (recall {rec(5, "KNN", "Moderate"):.2f}). Por eso todo el análisis usa métricas macro y el reporte por
clase.</p>
<p>Al balancear con SMOTENC, el F1-macro medido sobre la prueba (que conserva la distribución real) <b>baja levemente en las cuatro
técnicas</b>. El árbol pasa de {f1_bal.loc["Árbol de decisión"].iloc[0]:.3f} a {f1_bal.loc["Árbol de decisión"].iloc[1]:.3f} y el
SVM de {f1_bal.loc["SVM"].iloc[0]:.3f} a {f1_bal.loc["SVM"].iloc[1]:.3f}. Las causas:</p>
<ul>
<li>El modelo entrenado con clases iguales asume que <i>Moderate</i>, <i>High</i> y <i>Severe</i> son tan frecuentes como
<i>Low</i>, y en la prueba no lo son. Predice más las minoritarias: <b>sube su recall pero baja su precisión</b>. Por ejemplo, el
recall de <i>Moderate</i> del SVM pasa de {rec(1, "SVM", "Moderate"):.2f} (caso 1) a {rec(2, "SVM", "Moderate"):.2f} (caso 2), y
el de KNN de {rec(1, "KNN", "Moderate"):.2f} a {rec(2, "KNN", "Moderate"):.2f}. En cambio, el recall de <i>Low</i> baja y aparecen
falsas alarmas.</li>
<li>Los registros sintéticos se generan en la zona de solapamiento Low–Moderate. Allí la frontera real depende del
<code>congestion_score</code> que se eliminó, y los sintéticos añaden ruido justo en esa zona.</li>
</ul>
<p>Balancear es, por tanto, un intercambio de precisión por recall en las minoritarias. Es útil si se prioriza detectar la
congestión, como en este dominio, aunque el F1-macro global baje un poco.</p>

<h3>Costo de los errores y métrica priorizada</h3>
<ul>
<li><b>Falso negativo</b> (predecir <i>Low</i> o <i>Moderate</i> cuando hay congestión <i>High</i> o <i>Severe</i>): la central no
actúa. No se reprograman semáforos ni se desvía el tráfico, las ambulancias quedan atrapadas, y crecen las colas, las emisiones
y el riesgo de accidentes. <b>Es el error más costoso.</b></li>
<li><b>Falso positivo</b> (alertar congestión donde no la hay): provoca un desvío innecesario o una alerta que el operador
descarta. Tiene costo, sobre todo la fatiga de alertas, pero es menor.</li>
</ul>
<p>Por eso se prioriza el <b>recall de las clases <i>High</i> y <i>Severe</i></b>, y el <b>F1-macro</b> como métrica global de
selección: mantiene la precisión bajo control para que el sistema no se llene de falsas alarmas. Con este criterio, en el
caso 8 el árbol y la red neuronal son los modelos más seguros: recall <i>Severe</i> de {rec(8, "Árbol de decisión", "Severe"):.3f}
y {rec(8, "Red neuronal", "Severe"):.3f}, frente a {rec(8, "SVM", "Severe"):.3f} del SVM y {rec(8, "KNN", "Severe"):.3f} de KNN.</p>

<h2>4.7 Matrices de confusión y reporte por clase (caso 8)</h2>
{figura("sup_caso8_matrices_confusion", "Matrices de confusión del caso 8 (ED SI · Outliers SI · Balanceo SI) en el conjunto de prueba.")}
{tabla_por_clase_caso8()}
<p>Los errores se concentran entre clases vecinas en el orden de la congestión (Low↔Moderate, High↔Severe). Casi nunca se
confunde <i>Low</i> con <i>Severe</i>, lo que confirma que los modelos capturan el orden natural del problema. <i>Moderate</i> es
la clase con peor precisión en todos los modelos, porque recibe los <i>Low</i> que están cerca del umbral. Las matrices
normalizadas de KNN y SVM en los 8 casos, en el anexo A, muestran cómo el escalado degrada a KNN en <i>Moderate</i> y
<i>High</i>.</p>

<h2>4.8 Curvas ROC y Precision-Recall (caso 8)</h2>
{figura("sup_caso8_roc_pr", "Curvas ROC y Precision-Recall macro-promedio (uno-contra-resto) de las 4 técnicas — caso 8.")}
{tabla_html(roc[["AUC_ROC_macro", "AP_macro", "AUC_Moderate", "AP_Moderate", "AUC_Severe", "AP_Severe"]], fmt={c: f4 for c in ["AUC_ROC_macro", "AP_macro", "AUC_Moderate", "AP_Moderate", "AUC_Severe", "AP_Severe"]})}
{figura("sup_caso8_roc_pr_por_clase", "Curvas ROC (zoom en FPR ≤ 0.4) y Precision-Recall por clase — caso 8.")}
<p><b>Análisis.</b></p>
<ul>
<li><b>La red neuronal domina en ambas curvas</b> (AUC macro {roc.loc["Red neuronal", "AUC_ROC_macro"]:.4f},
AP macro {roc.loc["Red neuronal", "AP_macro"]:.4f}). Sus probabilidades <i>softmax</i> son continuas y bien ordenadas. Que su
F1 en el caso 8 sea inferior al del árbol se debe al <b>umbral</b>, no a la capacidad de discriminar: entrenada con clases
balanceadas, la regla argmax sobrepredice <i>Moderate</i> en una prueba donde esa clase es solo el 10 %. Un ajuste del umbral
por clase mejoraría su F1.</li>
<li><b>El árbol</b> tiene curvas casi lineales por tramos, porque sus probabilidades salen de pocas hojas. Aun así logra
AUC {roc.loc["Árbol de decisión", "AUC_ROC_macro"]:.3f}.</li>
<li><b>El SVM</b> (puntajes de <code>decision_function</code>) es excelente en <i>High</i> y <i>Low</i>, pero débil en
<i>Moderate</i> (AP {roc.loc["SVM", "AP_Moderate"]:.3f}).</li>
<li><b>KNN</b> es el peor (AP <i>Moderate</i> {roc.loc["KNN", "AP_Moderate"]:.3f}): en el caso 8 está escalado, la condición que
más lo perjudica.</li>
<li><b>ROC frente a PR:</b> todas las AUC-ROC superan 0.91 y parecen muy similares, mientras que las curvas PR separan con
claridad a los modelos. Con clases desbalanceadas, la ROC es optimista porque la tasa de falsos positivos se diluye entre
muchos negativos (<i>Low</i>). La PR muestra directamente la precisión sobre la clase positiva, así que es la curva más
informativa para este problema.</li>
</ul>

<h2>4.9 Visualización de resultados por técnica</h2>
<p>Para cada uno de los 8 casos el notebook genera dos figuras. La primera es una figura compuesta con:</p>
<ul>
<li>importancia de características del árbol;</li>
<li><code>max_depth</code> vs accuracy en entrenamiento y validación (Gini vs Entropía);</li>
<li>k vs accuracy (CV 5 folds);</li>
<li>vecinos de KNN para un registro de prueba;</li>
<li>regiones de decisión del SVM;</li>
<li>curva de pérdida de la red.</li>
</ul>
<p>La segunda es el árbol solución con Gini. Aquí se muestran los casos 1 y 8. Los demás están en el anexo A.</p>
{figura("sup_caso1_visualizaciones", "Visualizaciones del caso 1 (sin escalado, sin outliers, desbalanceado).")}
{figura("sup_caso8_visualizaciones", "Visualizaciones del caso 8 (escalado, con outliers, balanceado).")}
<ul>
<li><b>Sobreajuste y subajuste del árbol.</b> Con profundidad 1–2 el árbol <b>subajusta</b>: solo separa <i>Low</i> del
resto. Desde una profundidad de alrededor de 7 la accuracy de entrenamiento tiende a 1.0 mientras la de validación se estanca
o baja levemente: es la zona de <b>sobreajuste</b>. La profundidad elegida (4–11 según el caso) está en el codo de la
validación. Gini y Entropía dan curvas muy parecidas, porque ambas medidas de impureza ordenan los cortes de forma similar.
En algunos casos la Entropía logra hasta 1.5 puntos más de validación en profundidades grandes, una diferencia pequeña
frente al efecto de la profundidad.</li>
<li><b>KNN.</b> La curva de k muestra que la accuracy cae al aumentar k en los casos balanceados (vecindarios grandes mezclan
clases) y es más plana en los desbalanceados. En el gráfico de vecinos (PCA), los k vecinos del registro de consulta caen
en la región de su clase.</li>
<li><b>SVM.</b> Las regiones de decisión se calcularon proyectando una malla del plano PCA al espacio original
(<code>inverse_transform</code>) y clasificándola con el SVM entrenado con todas las variables. PCA solo se usa para
visualizar. En el plano, las regiones siguen el eje de congestión (PC1): <i>Low</i> a la izquierda, <i>Severe</i> a la
derecha y franjas angostas de <i>Moderate</i> y <i>High</i> entre ellas.</li>
</ul>
<div class="dos">{figura("sup_caso1_arbol", "Árbol solución (Gini) del caso 1, en unidades originales.")}</div>
<p><b>Interpretación de las reglas del árbol (caso 1, unidades reales, profundidad 4):</b></p>
<ul>
<li><code>traffic_density ≤ 38.65</code> → <b>Low</b> o <b>Moderate</b>. Dentro de esta rama, una velocidad media baja
(≤ 31.8 km/h) junto con emisiones altas indica <i>Moderate</i>; con velocidad alta es <i>Low</i>. En la franja de densidad
30–39, un clima distinto de despejado empuja hacia <i>Moderate</i>.</li>
<li><code>38.65 &lt; traffic_density ≤ 85.4</code> → <b>High</b> si <code>average_speed ≤ 28.45</code>; si no,
<b>Moderate</b>. Es decir, la misma densidad se vuelve congestión alta cuando el flujo se frena.</li>
<li><code>85.4 &lt; traffic_density ≤ 119.25</code> → <b>High</b> o <b>Severe</b> según el combustible desperdiciado
(<code>fuel_waste_estimate &gt; 78</code> → <i>Severe</i>).</li>
<li><code>traffic_density &gt; 119.25</code> → casi siempre <b>Severe</b>. La excepción son los registros con poco combustible
desperdiciado, que solo son <i>Severe</i> en la hora punta de la mañana.</li>
</ul>
<p>Las reglas tienen sentido físico. La congestión la define sobre todo la <b>densidad</b> (vehículos por km de carril), y la
<b>velocidad</b> y el <b>desperdicio de combustible</b> (indicador de <i>stop-and-go</i>) deciden entre niveles contiguos. El
clima y la franja horaria solo matizan casos frontera.</p>

<h2>4.10 Conclusiones del módulo supervisado</h2>
<ol>
<li>Las cuatro técnicas resuelven bien el problema (F1-macro de 0.93 a 0.96 en su mejor caso) porque la densidad de tráfico es
muy discriminante. El <b>SVM sin escalar y sin outliers</b> fue el mejor ({f1max.loc["SVM", "f1"]:.4f}), y el
<b>árbol</b> el más estable: el preprocesamiento casi no le afecta.</li>
<li><b>Escalado:</b> el árbol es invariante; KNN, SVM y la red son sensibles. En este dataset el escalado empeoró a KNN porque
iguala el peso de muchas variables poco informativas. Escalar debe ir acompañado de selección o ponderación de variables.</li>
<li><b>Outliers:</b> el SVM es el más vulnerable y el árbol el más robusto. Recortar los atípicos reales con Z-score fue
preferible a eliminarlos, porque eliminarlos borraba congestión severa real.</li>
<li><b>Desbalance:</b> el accuracy es engañoso: un modelo trivial obtendría 70 %. Balancear intercambia precisión por recall en
las minoritarias, lo que se justifica en el dominio porque un falso negativo de congestión es el error más costoso.</li>
<li><b>Curvas:</b> la red neuronal es el mejor clasificador probabilístico (AUC 0.995, AP 0.97). Su F1 depende del umbral.
La curva PR es más informativa que la ROC con clases desbalanceadas.</li>
</ol>

<!-- ======================================================================= 5 -->
<h1 class="salto">5. Módulo de aprendizaje no supervisado</h1>
<h2>5.1 Datasets (figura 2) y correspondencia con el módulo supervisado</h2>
<p>Se usa el mismo dataset sin la variable objetivo. Cada caso <i>i</i> de la figura 2 contiene <b>exactamente los registros de
entrenamiento del caso <i>i</i> supervisado</b>, con el mismo recorte de outliers, la misma inyección del 5 %, el mismo
SMOTENC y el mismo transformador CC/ED. Después se descarta <code>congestion_level</code>. Así, el "balanceo" de un caso no
supervisado significa que sus registros provienen del entrenamiento balanceado (12 444 registros, 25 % de cada clase) o del
desbalanceado (4445 registros, 70/10/10/10). La etiqueta original se guarda aparte y <b>solo se usa para la validación
externa</b> (ARI, NMI y tablas de contingencia), nunca para entrenar.</p>
<table class="tabla"><thead><tr><th>#</th><th>Normalización</th><th>Outliers</th><th>Balanceo</th><th>Registros</th><th>Variables</th><th>Origen</th></tr></thead><tbody>
{"".join(f"<tr><td>{c}</td><td>CC(SI) y ED({r.ED})</td><td>{r.OUTLIERS}</td><td>{r.BALANCEO}</td><td>{int(r.n_train)}</td><td>56</td><td>Entrenamiento del caso {c} supervisado, sin objetivo</td></tr>" for c, r in resumen_casos.iterrows())}
</tbody></table>

<h2>5.2 Configuración de K-Means y DBSCAN</h2>
<h3>K-Means</h3>
<ul>
<li>Se recorre K = 2…10 con inicialización <b>k-means++</b>, <code>n_init = 10</code> y <code>random_state = 19</code>.</li>
<li>Para cada K se grafican la <b>inercia (SSE)</b>, para el método del codo, y el <b>silhouette</b>.</li>
<li>El silhouette del barrido se estima con una muestra de 4000 puntos (su costo es O(n²)). El de la tabla final usa todos los
puntos.</li>
<li>El <b>K óptimo es el de mayor silhouette</b>, contrastado con el codo de la inercia, que se detecta con el método
<i>Kneedle</i>.</li>
</ul>
<h3>DBSCAN</h3>
<ul>
<li><b>MinPts = 2 × número de variables = 2 × 56 = 112</b>, según la regla general. Es un valor alto, apropiado para un
conjunto grande y con ruido.</li>
<li><b>ε</b> se toma en el <b>codo del gráfico de k-distancia</b> (distancia al 112-ésimo vecino, ordenada). El codo es el
punto más alejado de la recta que une los extremos de la curva normalizada.</li>
<li>El silhouette se calcula <b>excluyendo el ruido</b>.</li>
<li>Si el ε del codo produce menos de 2 clusters (silhouette indefinido), se hace un análisis de sensibilidad entre
0.3× y 1.5× el ε del codo y se toma el ε con 2 o más clusters, a lo sumo 50 % de ruido y mayor silhouette. Esto ocurrió en
los casos 4, 5 y 7, y se reporta en la tabla.</li>
</ul>

<h2>5.3 Resultados — tabla de la figura 3</h2>
{tabla_figura3()}
{figura("nosup_silhouette_casos", "Silhouette score por caso: K-Means (izquierda) y DBSCAN (derecha), 8 barras cada uno.")}

<h2>5.4 Análisis de resultados</h2>
<h3>Efecto del escalado: el silhouette alto puede engañar</h3>
<ul>
<li><b>Sin escalado</b> (casos 1–4) el silhouette es alto: 0.63–0.79 en K-Means y 0.55–0.86 en DBSCAN. Pero no indica una
estructura rica. La distancia euclidiana la dominan <code>vehicle_count</code>, <code>queue_length</code> y
<code>emission_estimate</code>, con varianzas de cientos o miles de unidades, así que el agrupamiento es casi unidimensional
por "volumen de tráfico". Como esos volúmenes son bimodales, la separación en dos grupos parece excelente. La inercia
(2.5·10⁸–6.7·10⁸) está en unidades de vehículos² y no es comparable con la de los casos escalados.</li>
<li><b>Con escalado</b> (casos 5–8) todas las variables pesan igual y el silhouette baja a 0.25–0.41. En 56 dimensiones,
la mayoría poco relacionadas con la congestión, no hay grupos globulares nítidos.</li>
<li>K-Means usa distancia euclidiana, así que el escalado decide qué variables definen los clusters. Sin escalar, los clusters
se parecen más a la etiqueta real (ARI {det_ns.loc[1, "km_ARI"]:.2f} en el caso 1 frente a {det_ns.loc[5, "km_ARI"]:.2f} en el
caso 5), porque las variables dominantes son justo las de congestión. Es el mismo fenómeno que se vio con KNN.</li>
</ul>
<h3>Efecto de los outliers</h3>
<ul>
<li>Los outliers inyectados <b>aumentan la inercia</b> (caso 1 → 3: {fig3.loc[1, "K-MEANS INERTIA"]:.3g} →
{fig3.loc[3, "K-MEANS INERTIA"]:.3g}, +{100 * (fig3.loc[3, "K-MEANS INERTIA"] / fig3.loc[1, "K-MEANS INERTIA"] - 1):.0f} %).
K-Means minimiza distancias al cuadrado, así que unos pocos puntos extremos pesan mucho y desplazan los centroides.</li>
<li>El silhouette de K-Means baja levemente (0.79 → 0.78 y 0.41 → 0.38).</li>
<li><b>DBSCAN es robusto por diseño:</b> los atípicos quedan como ruido. En el caso 3 el ruido sube a
{det_ns.loc[3, "db_%ruido"]:.1f} % (frente a {det_ns.loc[1, "db_%ruido"]:.1f} % en el caso 1) y el silhouette de los clusters
se mantiene.</li>
</ul>
<h3>Efecto del balanceo</h3>
<ul>
<li>Los conjuntos balanceados (casos 2, 4, 6 y 8) tienen <b>menor silhouette</b> en K-Means (0.79 → 0.65 sin escalar,
0.41 → 0.27 escalado).</li>
<li>SMOTENC añade miles de registros <i>Moderate</i> y <i>High</i>, que ocupan la zona intermedia entre el tráfico fluido y la
congestión severa. Eso convierte la estructura bimodal en un <b>continuo</b> y los clusters quedan menos separados.</li>
<li>La inercia crece también porque hay casi el triple de puntos (12 444 frente a 4445). La inercia solo es comparable entre
casos del mismo tamaño y la misma escala.</li>
</ul>
<h3>Número de clusters</h3>
<p>El silhouette elige <b>K = 2</b> en 6 de los 8 casos (K = 3 en los balanceados sin escalar), mientras que el codo de la
inercia sugiere K = 4–5. Esa discrepancia es informativa. La estructura dominante de los datos es binaria: <b>tráfico fluido
frente a congestión</b>. Los niveles intermedios <i>Moderate</i> y <i>High</i> no forman grupos compactos propios, son
transiciones. Las 4 clases supervisadas son umbrales sobre un puntaje continuo, no grupos naturales. Esto explica los ARI
moderados (0.20–0.54).</p>
<h3>DBSCAN: sensibilidad a ε y densidades variables</h3>
<table class="tabla"><thead><tr><th>Caso 8</th>{"".join(f"<th>{f}× codo</th>" for f in s8.index)}</tr></thead><tbody>
<tr><td>ε</td>{"".join(f"<td>{v:.2f}</td>" for v in s8.eps)}</tr>
<tr><td>clusters</td>{"".join(f"<td>{int(v)}</td>" for v in s8.clusters)}</tr>
<tr><td>% ruido</td>{"".join(f"<td>{v:.1f}</td>" for v in s8["%ruido"])}</tr>
<tr><td>silhouette (sin ruido)</td>{"".join("<td>N/D</td>" if pd.isna(v) else f"<td>{v:.3f}</td>" for v in s8.silhouette)}</tr>
</tbody></table>
<ul>
<li>La sensibilidad del caso 8 (tabla) muestra el comportamiento descrito en clase. Con <b>ε pequeño</b> (0.3–0.5× el codo)
casi todo es ruido (79–100 %) o aparecen muchos micro-clusters. Con <b>ε grande</b> (1.25–1.5×) el ruido cae por debajo del
3 % y las regiones se <b>fusionan</b> en uno o dos clusters.</li>
<li>En los casos 4, 5 y 7, el ε del codo daba un único cluster. Por eso se usó un ε más pequeño del análisis de
sensibilidad, a costa de un 23–35 % de ruido.</li>
<li>El ARI de DBSCAN con escalado es casi 0 (por ejemplo, {det_ns.loc[8, "db_ARI"]:.3f} en el caso 8). La región de tráfico
fluido es muy densa y la de congestión severa está dispersa. <b>Un único ε no puede capturar ambas densidades</b>: o la región
dispersa queda como ruido, o se fusiona con la densa. Es la limitación clásica de DBSCAN con densidades variables.</li>
<li>El silhouette de DBSCAN sin escalar (0.86) supera al de K-Means. Pero se calcula sin el ruido, es decir, sin los puntos
frontera más difíciles, así que es una medida optimista.</li>
</ul>

<h2>5.5 Visualización 2D e interpretación de los clusters</h2>
<p>Para cada caso se proyectan los datos a 2D con PCA. Se muestran los clusters de K-Means con sus <b>centroides</b> (X negras),
los de DBSCAN con el <b>ruido en gris</b> y, como referencia, la etiqueta real (usada solo para validar).</p>
{figura("nosup_caso1", "Caso 1 (sin escalado): codo, silhouette vs K, k-distancia y clusters en PCA 2D.")}
{figura("nosup_caso8", "Caso 8 (escalado, outliers, balanceado): codo, silhouette vs K, k-distancia y clusters en PCA 2D.")}
<p><b>Perfil de los clusters de K-Means en el caso 5</b> (medias en unidades originales):</p>
{tabla_html(p5[["n", "vehicle_count", "average_speed", "traffic_density", "queue_length", "average_wait_time", "heavy_vehicle_count", "rush_hour", "clase_real_mayoritaria"]].round(2))}
<ul>
<li><b>¿Tienen sentido en el dominio?</b> Sí. El <b>cluster 0</b> (n = {int(p5.loc[0, "n"])}) es <b>tráfico fluido</b>: unos
{p5.loc[0, "vehicle_count"]:.0f} vehículos/h, {p5.loc[0, "average_speed"]:.0f} km/h, densidad {p5.loc[0, "traffic_density"]:.0f}
y colas de {p5.loc[0, "queue_length"]:.0f} m; el 77 % de sus registros son <i>Low</i>. El <b>cluster 1</b>
(n = {int(p5.loc[1, "n"])}) es <b>congestión severa</b>: unos {p5.loc[1, "vehicle_count"]:.0f} vehículos/h,
{p5.loc[1, "average_speed"]:.0f} km/h, densidad {p5.loc[1, "traffic_density"]:.0f}, colas de {p5.loc[1, "queue_length"]:.0f} m y
esperas 17 veces mayores; el 92 % son <i>Severe</i>. Son los dos regímenes de la teoría del flujo vehicular (flujo libre y
flujo congestionado).</li>
<li><b>DBSCAN en el caso 8</b> encuentra un patrón distinto. Además del gran grupo principal, aísla un pequeño cluster de
{int(perfil_db8.loc[1, "n"])} registros de <b>madrugada</b> (hora media {perfil_db8.loc[1, "hour"]:.1f}, unos
{perfil_db8.loc[1, "vehicle_count"]:.0f} vehículos/h, sin hora punta, 94 % <i>Low</i>): el tráfico nocturno, una región muy
densa y homogénea. Los {int(perfil_db8.loc[-1, "n"])} puntos de ruido incluyen los outliers inyectados.</li>
<li><b>Tres registros de prueba:</b> el registro <i>Low</i> cae siempre en el cluster de flujo libre. El <i>Severe</i> va al
cluster de congestión de K-Means en los 6 casos con K = 2, y DBSCAN lo marca como <b>ruido</b> en los 8. Esto confirma que la
congestión severa ocupa una región de baja densidad.</li>
</ul>

<h2>5.6 Comparación K-Means frente a DBSCAN</h2>
<table class="tabla"><thead><tr><th>Aspecto</th><th>K-Means</th><th>DBSCAN</th></tr></thead><tbody>
<tr><td>Forma de los clusters</td><td>Esféricos y convexos alrededor de un centroide (minimiza la SSE)</td><td>Arbitrarios: une regiones densas conectadas</td></tr>
<tr><td>Número de clusters</td><td>Hay que definir K (codo y silhouette). Aquí se obtuvo K = 2–3</td><td>Lo determina el algoritmo a partir de ε y MinPts. Aquí se obtuvieron de 2 a 4 clusters</td></tr>
<tr><td>Ruido y outliers</td><td>Asigna todos los puntos; los outliers desplazan centroides y aumentan la inercia</td><td>Los etiqueta como ruido (−1); 1.5–35 % según el caso</td></tr>
<tr><td>Densidades variables</td><td>No le afecta la densidad, pero sí el tamaño y la forma de los grupos</td><td>Dificultad seria: un único ε no sirve para la región densa (Low) y la dispersa (Severe)</td></tr>
<tr><td>Sensibilidad a parámetros</td><td>Baja con k-means++ y n_init = 10 (resultados estables)</td><td>Alta: pequeños cambios de ε pasan de 100 % de ruido a un solo cluster</td></tr>
<tr><td>Escala</td><td>Muy sensible (distancia euclidiana)</td><td>Muy sensible (ε es una distancia)</td></tr>
<tr><td>Coincidencia con la etiqueta (ARI)</td><td>0.20–0.54</td><td>0.00–0.37</td></tr>
</tbody></table>

<h2>5.7 Conclusiones del módulo no supervisado</h2>
<ol>
<li>La estructura natural de los datos es <b>flujo libre frente a congestión</b> (K = 2), coherente con la teoría del tráfico.
Las cuatro clases supervisadas son cortes de un continuo, no grupos naturales.</li>
<li>El <b>escalado</b> define qué significa "parecido": sin él, los clusters los dominan las variables de volumen. El silhouette
es alto pero refleja casi una sola dimensión. Comparar silhouettes entre escalas distintas exige cuidado.</li>
<li>Los <b>outliers</b> afectan a K-Means (más inercia, centroides desplazados) y DBSCAN los aísla como ruido.</li>
<li>El <b>balanceo</b> rellena la zona intermedia y reduce la separación entre clusters.</li>
<li><b>K-Means</b> fue más estable y más alineado con el dominio en este problema. <b>DBSCAN</b> es útil para detectar
anomalías y regímenes densos (como el tráfico nocturno), pero sufre con las densidades variables y es muy sensible a ε.</li>
</ol>

<!-- ======================================================================= 6 -->
<h1 class="salto">6. Conclusiones generales</h1>
<ol>
<li>El dataset CityFlow cumple todos los requisitos y permitió construir un problema realista. Fue clave detectar y eliminar la
<b>fuga de información</b> (<code>congestion_score</code>): sin ese paso todos los modelos habrían obtenido 100 % sin aprender
nada útil.</li>
<li>En el módulo supervisado, las cuatro técnicas alcanzan un F1-macro de 0.93–0.96 en su mejor caso. El <b>SVM</b> obtuvo el mejor resultado
puntual, el <b>árbol</b> fue el más robusto e interpretable (sus reglas coinciden con la intuición física del tráfico) y la
<b>red neuronal</b> fue el mejor clasificador probabilístico (AUC 0.995).</li>
<li>El preprocesamiento no es neutro y su efecto depende del algoritmo:
<ul>
<li>el escalado no afecta al árbol y sí a KNN, SVM y la red, y en este dataset perjudicó a KNN;</li>
<li>los outliers afectan sobre todo al SVM y a K-Means;</li>
<li>el balanceo intercambia precisión por recall.</li>
</ul>
Estos resultados confirman la teoría y muestran que cada decisión debe validarse con datos.</li>
<li>Con clases desbalanceadas el <b>accuracy engaña</b>. Las métricas macro, el reporte por clase y la curva Precision-Recall
son imprescindibles. En el dominio, el recall de <i>High</i> y <i>Severe</i> es la prioridad, porque un falso negativo de
congestión es el error más costoso.</li>
<li>El aprendizaje no supervisado, sin ver la etiqueta, redescubre la distinción principal entre flujo libre y congestión
(ARI de hasta 0.54). Muestra que los niveles intermedios no son grupos naturales y aísla regímenes de interés, como el tráfico
nocturno, y registros anómalos.</li>
<li>Se cuidó la validez metodológica:
<ul>
<li>un conjunto de prueba único y sin modificar;</li>
<li>todas las transformaciones ajustadas solo con el entrenamiento;</li>
<li>validación cruzada sin fuga de SMOTE;</li>
<li>PCA usado solo para visualizar;</li>
<li>semillas fijas.</li>
</ul></li>
</ol>

<!-- ======================================================================= 7 -->
<h1>7. Video pitch</h1>
<p><b>Enlace (YouTube):</b> {video_html()}</p>
<p>Guion sugerido (máximo 5 minutos; deben participar los cuatro integrantes):</p>
<table class="tabla"><thead><tr><th>Tiempo</th><th>Integrante</th><th>Contenido</th></tr></thead><tbody>
<tr><td>0:00–0:45</td><td>1</td><td>Problema: congestión urbana y costo de no detectarla. Dataset CityFlow, verificación de requisitos, cédula → 6350 registros y semilla 19. Fuga de información en <code>congestion_score</code>.</td></tr>
<tr><td>0:45–1:45</td><td>2</td><td>Preprocesamiento: One-Hot/Ordinal, StandardScaler, Z-score y 5 % de outliers inyectados, SMOTENC, los 8 casos y la prueba única. Demo: ejecución del notebook en Colab.</td></tr>
<tr><td>1:45–3:15</td><td>3</td><td>Supervisado: figura 1 y gráficas de F1 (escala, outliers, balanceo), por qué el accuracy engaña, matrices y curvas ROC/PR del caso 8, reglas del árbol.</td></tr>
<tr><td>3:15–4:30</td><td>4</td><td>No supervisado: codo y silhouette, k-distancia, figura 3, clusters de flujo libre y congestión, K-Means frente a DBSCAN.</td></tr>
<tr><td>4:30–5:00</td><td>Todos</td><td>Conclusiones principales.</td></tr>
</tbody></table>

<!-- ======================================================================= anexos -->
<h1 class="salto">Anexo A. Visualizaciones supervisadas de los casos 2 a 7</h1>
{"".join(figura(f"sup_caso{c}_visualizaciones", f"Visualizaciones del caso {c}.") for c in range(2, 8))}
{figura("sup_matrices_knn_8casos", "Matrices de confusión normalizadas de KNN en los 8 casos.")}
{figura("sup_matrices_svm_8casos", "Matrices de confusión normalizadas del SVM en los 8 casos.")}
{figura("sup_red_curvas_8casos", "Accuracy por época de la red neuronal en los 8 casos (con early stopping).")}
{"".join(figura(f"sup_caso{c}_arbol", f"Árbol solución (Gini) del caso {c}.") for c in (2, 3, 4, 5, 6, 7, 8))}

<h1 class="salto">Anexo B. Visualizaciones no supervisadas de los casos 2 a 7</h1>
{"".join(figura(f"nosup_caso{c}", f"Aprendizaje no supervisado — caso {c}.") for c in range(2, 8))}

</body></html>
"""

SALIDA_HTML.write_text(html, encoding="utf-8")
print("HTML:", SALIDA_HTML)
faltan = [n for n, v in (("INTEGRANTES", INTEGRANTES), ("CEDULA", CEDULA), ("URL_VIDEO", URL_VIDEO)) if not v]
if faltan:
    print("AVISO: faltan datos del equipo (aparecen resaltados en el documento):", ", ".join(faltan))

navegadores = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
               r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
               r"C:\Program Files\Google\Chrome\Application\chrome.exe",
               shutil.which("msedge") or "", shutil.which("chrome") or "", shutil.which("google-chrome") or ""]
nav = next((n for n in navegadores if n and os.path.exists(n)), None)
if nav:
    subprocess.run([nav, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={SALIDA_PDF}", SALIDA_HTML.as_uri()], check=True, timeout=180)
    print("PDF:", SALIDA_PDF)
else:
    print("No se encontró Edge/Chrome: abra el HTML en un navegador e imprímalo a PDF.")
