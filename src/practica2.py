# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Trabajo Práctico 2 — Aprendizaje de Máquinas
# **Universidad Nacional de Colombia, Sede Medellín — Introducción a la Inteligencia Artificial (3010476)**
#
# **Grupo 01 — Equipo 07**
#
# | Integrante | Documento |
# |---|---|
# | _[Nombre integrante 1]_ | _[Cédula usada para el subconjunto]_ |
# | _[Nombre integrante 2]_ | |
# | _[Nombre integrante 3]_ | |
# | _[Nombre integrante 4]_ | |
#
# **Dominio:** movilidad urbana inteligente / tráfico IoT.
# **Dataset:** *CityFlow — Smart Urban Mobility and Traffic IoT* (Kaggle,
# `mobeenfatimah/cityflow-smart-urban-mobility-and-traffic-iot`).
# **Variable objetivo:** `congestion_level` (4 clases ordinales: Low, Moderate, High, Severe).
#
# ## Flujo general del notebook
# 1. **Instalación y configuración**: librerías, semilla y tamaño del subconjunto derivados de la cédula.
# 2. **Módulo común (preprocesamiento)**: carga, limpieza, subconjunto, EDA, partición 70/15/15,
#    funciones de outliers, balanceo, conversión de categóricas y escalado, y construcción de los 8 casos.
# 3. **Módulo de aprendizaje supervisado**: Árboles de decisión, KNN, SVM y Redes Neuronales en los 8 casos
#    (tabla de la figura 1, gráficas comparativas, matrices de confusión, curvas ROC y Precision-Recall,
#    visualizaciones por técnica).
# 4. **Módulo de aprendizaje no supervisado**: K-Means y DBSCAN en los 8 casos (tabla de la figura 3,
#    codo, silhouette, gráfico de k-distancia, visualización 2D con PCA y validación externa).
#
# Todas las funciones están documentadas y son reutilizables. El notebook se ejecuta de principio a fin con
# semillas fijas.

# %% [markdown]
# ## 0. Instalación de librerías
# En Google Colab ya vienen instaladas `scikit-learn`, `tensorflow`, `pandas`, `matplotlib` y `seaborn`;
# se instalan además `kagglehub` (descarga del dataset) e `imbalanced-learn` (SMOTENC).

# %%
# %pip install -q kagglehub imbalanced-learn scikit-learn pandas matplotlib seaborn tensorflow

# %%
import os
import time
import random
import warnings

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.svm import SVC
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay, classification_report,
                             roc_curve, auc, precision_recall_curve, average_precision_score,
                             silhouette_score, adjusted_rand_score, normalized_mutual_info_score)
from imblearn.over_sampling import SMOTENC

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, callbacks

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 200)
print("TensorFlow", tf.__version__)

# %% [markdown]
# ## 1. Configuración del equipo (cédula → tamaño del subconjunto y semilla)
# Regla del enunciado: **registros = 5000 + (último dígito ≠ 0 de la cédula) × 150** y
# **random state = dos últimos dígitos ≠ 0 de la misma cédula** (distinto de 42).
#
# Los dos últimos dígitos distintos de cero de la cédula elegida son **1 y 9**:
# * último dígito ≠ 0 → **9** → 5000 + 9 × 150 = **6350 registros**
# * semilla → **19**

# %%
ULTIMO_DIGITO_CEDULA = 9                       # último dígito distinto de cero
SEMILLA = 19                                   # dos últimos dígitos distintos de cero
N_REGISTROS = 5000 + ULTIMO_DIGITO_CEDULA * 150
assert SEMILLA != 42, "El enunciado prohíbe usar la semilla 42"

DATASET_KAGGLE = "mobeenfatimah/cityflow-smart-urban-mobility-and-traffic-iot"
OBJETIVO = "congestion_level"
CLASES = ["Low", "Moderate", "High", "Severe"]  # orden natural (ordinal) de la congestión
DIR_FIG = "resultados/figuras"
DIR_TAB = "resultados/tablas"
os.makedirs(DIR_FIG, exist_ok=True)
os.makedirs(DIR_TAB, exist_ok=True)


def fijar_semillas(semilla=SEMILLA):
    """Fija las semillas de Python, NumPy y TensorFlow para que todo el notebook sea reproducible."""
    os.environ["PYTHONHASHSEED"] = str(semilla)
    random.seed(semilla)
    np.random.seed(semilla)
    tf.keras.utils.set_random_seed(semilla)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


fijar_semillas()
print(f"Registros del subconjunto: {N_REGISTROS} | Semilla (random_state): {SEMILLA}")

# %% [markdown]
# ### Estilo de las gráficas
# Paleta categórica validada para daltonismo (azul, naranja, aqua, amarillo) asignada siempre en el mismo
# orden a los algoritmos. Las clases de congestión son **ordinales**, por eso se representan con una
# rampa secuencial de un solo tono (azul claro → azul oscuro) y, en los diagramas de dispersión, además con
# un marcador distinto por clase (para no depender solo del color).

# %%
COLOR_ALG = {"Árbol de decisión": "#2a78d6", "KNN": "#eb6834", "SVM": "#1baf7a", "Red neuronal": "#eda100"}
COLOR_PAR = ["#2a78d6", "#eb6834"]                          # comparaciones de dos niveles (NO / SI)
COLOR_CLASE = {"Low": "#86b6ef", "Moderate": "#3987e5", "High": "#1c5cab", "Severe": "#0d366b"}
MARCA_CLASE = {"Low": "o", "Moderate": "s", "High": "^", "Severe": "D"}
COLOR_RUIDO = "#898781"
TINTA, TINTA2, REJILLA = "#0b0b0b", "#52514e", "#e1e0d9"

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": TINTA2, "text.color": TINTA,
    "xtick.color": TINTA2, "ytick.color": TINTA2, "axes.grid": True, "grid.color": REJILLA,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
    "lines.linewidth": 2, "figure.dpi": 100,
})


def guardar_figura(fig, nombre):
    """Guarda la figura en `resultados/figuras/<nombre>.png`, la muestra y la cierra."""
    fig.savefig(os.path.join(DIR_FIG, f"{nombre}.png"), dpi=130, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def guardar_tabla(df, nombre):
    """Guarda un DataFrame como CSV en `resultados/tablas/<nombre>.csv`."""
    df.to_csv(os.path.join(DIR_TAB, f"{nombre}.csv"), encoding="utf-8")


# %% [markdown]
# ---
# # MÓDULO COMÚN — Datos y preprocesamiento
# ## 2. Carga del dataset y selección de variables

# %%
def cargar_dataset():
    """Descarga (o reutiliza la caché de) el dataset de Kaggle con kagglehub y lo devuelve como DataFrame."""
    import kagglehub
    ruta = kagglehub.dataset_download(DATASET_KAGGLE)
    archivo = [f for f in os.listdir(ruta) if f.endswith(".csv")][0]
    return pd.read_csv(os.path.join(ruta, archivo))


df_original = cargar_dataset()
print("Dimensiones del dataset original:", df_original.shape)
display(df_original.head())

# %% [markdown]
# **Columnas descartadas y justificación**
#
# | Columna | Motivo |
# |---|---|
# | `congestion_score` | **Fuga de información**: la etiqueta se obtiene por umbrales exactos de este puntaje (Low < 25 ≤ Moderate < 55 ≤ High < 80 ≤ Severe). Con ella cualquier modelo acierta el 100 %. |
# | `traffic_flow_rate` | Duplicado exacto de `vehicle_count` (correlación 1.0). |
# | `record_id` | Identificador sin información predictiva. |
# | `timestamp` | Su información útil ya está en `hour`, `day_of_week`, `is_weekend`, `peak_period`. |
# | `road_id`, `intersection_id` | Identificadores de alta cardinalidad (≈100 niveles); la ubicación ya está en `city_zone`, `latitude` y `longitude`. |

# %%
COLUMNAS_DESCARTADAS = ["congestion_score", "traffic_flow_rate", "record_id", "timestamp",
                        "road_id", "intersection_id"]

# Tipología de variables usada en todo el preprocesamiento
NOMINALES = ["city_zone", "road_type", "peak_period", "weather_condition", "iot_sensor_health"]
ORDINALES = {"road_condition": ["Good", "Wet", "Poor"]}          # orden de severidad del pavimento
BINARIAS = ["is_weekend", "is_holiday", "rush_hour", "nearby_school", "nearby_hospital", "nearby_market",
            "construction_activity", "public_event", "accident_reported", "emergency_vehicle_detected"]
DISCRETAS_POCOS_VALORES = ["lanes", "speed_limit", "traffic_camera_count", "signal_cycle_seconds",
                           "green_light_duration", "day_of_week"]
CONTINUAS = ["vehicle_count", "average_speed", "heavy_vehicle_count", "motorcycle_count",
             "public_transport_count", "traffic_density", "queue_length", "average_wait_time",
             "temperature", "rainfall_mm", "visibility_km", "air_quality_index", "parking_occupancy",
             "emission_estimate", "fuel_waste_estimate"]
OTRAS_NUMERICAS = ["latitude", "longitude", "hour"]
NUMERICAS = BINARIAS + DISCRETAS_POCOS_VALORES + CONTINUAS + OTRAS_NUMERICAS


def seleccionar_variables(df):
    """Elimina las columnas con fuga de información, duplicadas o identificadoras."""
    return df.drop(columns=COLUMNAS_DESCARTADAS)


df_limpio = seleccionar_variables(df_original)
assert set(NOMINALES + list(ORDINALES) + NUMERICAS + [OBJETIVO]) == set(df_limpio.columns)
print("Variables predictoras:", df_limpio.shape[1] - 1,
      f"({len(NOMINALES) + len(ORDINALES)} categóricas, {len(NUMERICAS)} numéricas)")

# %% [markdown]
# ## 3. Construcción del subconjunto único (6350 registros, desbalanceado 70/10/10/10)
# El subconjunto se extrae con la semilla del equipo. Se construye directamente con la distribución
# **desbalanceada** que pide el enunciado: la clase mayoritaria (`Low`, el estado más frecuente del tráfico)
# con el 70 % de los registros y las otras tres clases con el 10 % cada una. Esta es la versión
# **"BALANCEO = NO"**; la versión balanceada se obtiene después, solo sobre el conjunto de entrenamiento.

# %%
def construir_subconjunto(df, n, proporciones, semilla=SEMILLA):
    """Muestrea `n` registros respetando `proporciones` (dict clase -> fracción) con la semilla dada."""
    cuotas = {c: int(round(n * p)) for c, p in proporciones.items()}
    cuotas[max(cuotas, key=cuotas.get)] += n - sum(cuotas.values())     # ajuste por redondeo
    partes = [df[df[OBJETIVO] == c].sample(k, random_state=semilla) for c, k in cuotas.items()]
    return pd.concat(partes).sample(frac=1, random_state=semilla).reset_index(drop=True)


PROPORCIONES = {"Low": 0.70, "Moderate": 0.10, "High": 0.10, "Severe": 0.10}
df = construir_subconjunto(df_limpio, N_REGISTROS, PROPORCIONES)
print("Subconjunto:", df.shape)
display(df[OBJETIVO].value_counts().reindex(CLASES).to_frame("registros")
        .assign(porcentaje=lambda t: (100 * t.registros / len(df)).round(2)))

# %% [markdown]
# ## 4. Análisis exploratorio y valores nulos

# %%
def resumen_variables(df):
    """Tabla con tipo, tipo lógico, número de valores distintos y nulos de cada variable."""
    tipo_logico = {**{c: "categórica nominal" for c in NOMINALES},
                   **{c: "categórica ordinal" for c in ORDINALES},
                   **{c: "binaria" for c in BINARIAS},
                   **{c: "numérica discreta" for c in DISCRETAS_POCOS_VALORES + ["hour"]},
                   **{c: "numérica continua" for c in CONTINUAS + ["latitude", "longitude"]},
                   OBJETIVO: "objetivo (ordinal, 4 clases)"}
    return pd.DataFrame({"dtype": df.dtypes.astype(str), "tipo": pd.Series(tipo_logico),
                         "valores_distintos": df.nunique(), "nulos": df.isna().sum()}).loc[df.columns]


print("Dimensiones:", df.shape)
tabla_variables = resumen_variables(df)
display(tabla_variables)
guardar_tabla(tabla_variables, "eda_variables")

# %%
descriptivas = df[NUMERICAS].describe().T
display(descriptivas.round(3))
guardar_tabla(descriptivas.round(4), "eda_descriptivas")
display(df[NOMINALES + list(ORDINALES)].describe().T)


# %%
def imputar_nulos(df):
    """Imputa nulos: mediana en numéricas (robusta a outliers) y moda en categóricas.

    El dataset no tiene nulos, pero la función se deja en el flujo para que el preprocesamiento
    sea robusto si se usa otra muestra.
    """
    df = df.copy()
    for c in df.columns:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median() if c in NUMERICAS else df[c].mode()[0])
    return df


print("Total de nulos antes de imputar:", int(df.isna().sum().sum()))
df = imputar_nulos(df)

# %%
fig, ax = plt.subplots(figsize=(6, 3.2))
conteo = df[OBJETIVO].value_counts().reindex(CLASES)
ax.bar(CLASES, conteo.values, color=[COLOR_CLASE[c] for c in CLASES], width=0.6, edgecolor="#fcfcfb", linewidth=2)
for i, v in enumerate(conteo.values):
    ax.text(i, v + 40, f"{v} ({100 * v / len(df):.0f} %)", ha="center", color=TINTA2, fontsize=9)
ax.set_title("Distribución de clases del subconjunto (congestion_level)")
ax.set_ylabel("Registros")
ax.grid(axis="x", visible=False)
guardar_figura(fig, "eda_distribucion_clases")

# %%
fig, axes = plt.subplots(3, 5, figsize=(16, 8.5))
for ax, col in zip(axes.ravel(), CONTINUAS):
    for c in CLASES:
        ax.hist(df.loc[df[OBJETIVO] == c, col], bins=30, histtype="step", linewidth=1.4,
                color=COLOR_CLASE[c], label=c, density=True)
    ax.set_title(col, fontsize=9)
    ax.tick_params(labelsize=7)
axes[0, 0].legend(fontsize=7)
fig.suptitle("Distribución de las variables continuas por clase (densidad)", fontweight="bold")
fig.tight_layout()
guardar_figura(fig, "eda_histogramas")

# %%
df_corr = df[NUMERICAS].copy()
df_corr["road_condition"] = df["road_condition"].map({v: i for i, v in enumerate(ORDINALES["road_condition"])})
df_corr[OBJETIVO + "_ord"] = df[OBJETIVO].map({c: i for i, c in enumerate(CLASES)})
corr = df_corr.corr()
fig, ax = plt.subplots(figsize=(15, 12.5))
sns.heatmap(corr, cmap=sns.diverging_palette(250, 15, s=75, l=45, as_cmap=True), center=0, vmin=-1, vmax=1,
            square=True, linewidths=0.3, linecolor="#fcfcfb", cbar_kws={"shrink": 0.6}, ax=ax)
ax.set_title("Matriz de correlación (Pearson) de variables numéricas y objetivo ordinal")
ax.tick_params(labelsize=7)
guardar_figura(fig, "eda_correlacion")

top_corr = corr[OBJETIVO + "_ord"].drop(OBJETIVO + "_ord").sort_values(key=abs, ascending=False)
print("Correlación con el nivel de congestión (top 12):")
display(top_corr.head(12).round(3).to_frame("r"))
guardar_tabla(corr.round(4), "eda_correlacion")

# %% [markdown]
# ## 5. Partición estratificada 70 / 15 / 15
# Se divide una sola vez. El conjunto de **prueba es idéntico en los 8 casos**. El conjunto de validación
# se usa para elegir hiperparámetros y para el *early stopping* de la red neuronal.

# %%
def particionar(df, semilla=SEMILLA):
    """Divide de forma estratificada en entrenamiento (70 %), validación (15 %) y prueba (15 %)."""
    X, y = df.drop(columns=OBJETIVO), df[OBJETIVO]
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=semilla)
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=semilla)
    return [d.reset_index(drop=True) for d in (X_tr, X_va, X_te, y_tr, y_va, y_te)]


X_train_raw, X_val_raw, X_test_raw, y_train_raw, y_val, y_test = particionar(df)
tabla_particion = pd.DataFrame({n: s.value_counts().reindex(CLASES) for n, s in
                                [("entrenamiento", y_train_raw), ("validación", y_val), ("prueba", y_test)]})
tabla_particion.loc["Total"] = tabla_particion.sum()
display(tabla_particion)
guardar_tabla(tabla_particion, "particion")

# %% [markdown]
# ## 6. Funciones de preprocesamiento
# ### 6.1 Manejo de valores atípicos — Z-score (|z| > 3)
# * **Método:** Z-score con umbral 3 sobre las 15 variables continuas (`CONTINUAS`). Media y desviación se
#   ajustan **solo con entrenamiento**. Se eligió Z-score y no IQR porque varias variables tienen colas
#   largas y mezclan dos regímenes (tráfico fluido y congestionado). Con IQR × 1.5, el 23.5 % de las filas
#   saldrían marcadas como atípicas, la mayoría registros reales de congestión severa.
# * **Sin outliers:** los valores detectados se **tratan con recorte** (*winsorización*) a media ± 3σ. Se
#   recorta en lugar de eliminar porque eliminar borraría sobre todo registros `Severe` y alteraría el
#   balance de clases.
# * **Con outliers:** sobre el entrenamiento ya tratado se **inyectan outliers sintéticos** en exactamente
#   el 5 % de los registros. Para cada registro elegido se modifican 1 o 2 variables continuas al azar con
#   el valor media ± U(3.5, 6)·σ. La dirección respeta el dominio: hacia arriba en conteos y tiempos, hacia
#   abajo en visibilidad, en cualquier sentido para la temperatura. `parking_occupancy` se excluye porque
#   es un porcentaje acotado.

# %%
class TratadorOutliersZ:
    """Detecta outliers por Z-score (|z| > umbral) y los recorta a media ± umbral·σ.

    Las medias y desviaciones se ajustan (fit) únicamente con el conjunto de entrenamiento.
    """

    def __init__(self, columnas=CONTINUAS, umbral=3.0):
        self.columnas, self.umbral = columnas, umbral

    def fit(self, X):
        self.media_ = X[self.columnas].mean()
        self.std_ = X[self.columnas].std()
        return self

    def mascara(self, X):
        """Matriz booleana (filas × columnas) con True donde |z| > umbral.

        Se usa una tolerancia de 1e-6 para que los valores recortados exactamente en media ± 3σ no se
        cuenten como atípicos por errores de redondeo de punto flotante.
        """
        z = (X[self.columnas] - self.media_) / self.std_
        return z.abs() > self.umbral + 1e-6

    def transform(self, X):
        X = X.copy()
        lim_inf = self.media_ - self.umbral * self.std_
        lim_sup = self.media_ + self.umbral * self.std_
        X[self.columnas] = X[self.columnas].clip(lower=lim_inf, upper=lim_sup, axis=1)
        return X


DIRECCION_INYECCION = {"visibility_km": -1, "temperature": 0}     # -1 abajo, 0 cualquiera, +1 (defecto) arriba
CANDIDATAS_INYECCION = [c for c in CONTINUAS if c != "parking_occupancy"]


def inyectar_outliers(X, tratador, fraccion=0.05, semilla=SEMILLA):
    """Inyecta outliers sintéticos en `fraccion` de las filas de X.

    Devuelve (X_modificado, registro) donde `registro` documenta fila, variable, valor original y nuevo.
    """
    rng = np.random.default_rng(semilla)
    X = X.copy()
    n_out = int(round(fraccion * len(X)))
    filas = rng.choice(len(X), size=n_out, replace=False)
    registro = []
    for fila in filas:
        variables = rng.choice(CANDIDATAS_INYECCION, size=rng.integers(1, 3), replace=False)
        for var in variables:
            direccion = DIRECCION_INYECCION.get(var, 1) or rng.choice([-1, 1])
            nuevo = tratador.media_[var] + direccion * rng.uniform(3.5, 6.0) * tratador.std_[var]
            registro.append({"fila": int(fila), "variable": var, "original": X.at[fila, var], "nuevo": nuevo})
            X.at[fila, var] = nuevo
    return X, pd.DataFrame(registro)


# %% [markdown]
# ### 6.2 Balanceo de clases — SMOTENC
# * **Desbalanceado (BALANCEO = NO):** el entrenamiento conserva la distribución 70/10/10/10 del subconjunto.
# * **Balanceado (BALANCEO = SI):** **SMOTENC** sobremuestrea las clases minoritarias hasta igualar a la
#   mayoritaria. Se eligió SMOTENC porque genera registros sintéticos interpolados entre vecinos de la
#   misma clase en lugar de duplicar filas (el sobremuestreo aleatorio favorece el sobreajuste) y no
#   descarta información (el submuestreo dejaría solo unas 1780 filas). La variante *NC* trata
#   correctamente las variables categóricas: les asigna la moda de los vecinos en vez de interpolar.
#   También se tratan así las binarias y las discretas de pocos valores.
# * Para que la búsqueda de vecinos no esté dominada por las variables de mayor magnitud, las numéricas se
#   estandarizan temporalmente dentro de la función y luego se devuelven a su escala original. Las enteras se
#   redondean.
# * `class_weight` no se usa: el balanceo se hace en los datos para que los cuatro algoritmos (y el
#   agrupamiento) reciban exactamente los mismos registros.

# %%
CATEGORICAS_SMOTE = NOMINALES + list(ORDINALES) + BINARIAS + DISCRETAS_POCOS_VALORES


def balancear_smotenc(X, y, semilla=SEMILLA):
    """Balancea (X, y) con SMOTENC hasta igualar todas las clases con la mayoritaria."""
    columnas = list(X.columns)
    numericas = [c for c in columnas if c not in CATEGORICAS_SMOTE]
    escalador = StandardScaler().fit(X[numericas])
    X_tmp = X.copy()
    X_tmp[numericas] = escalador.transform(X[numericas])
    idx_cat = [columnas.index(c) for c in CATEGORICAS_SMOTE]
    smote = SMOTENC(categorical_features=idx_cat, random_state=semilla, k_neighbors=5)
    X_bal, y_bal = smote.fit_resample(X_tmp, y)
    X_bal = pd.DataFrame(X_bal, columns=columnas)
    X_bal[numericas] = escalador.inverse_transform(X_bal[numericas].astype(float))
    for c in columnas:                                   # restaurar tipos originales
        if c in CATEGORICAS_SMOTE and X[c].dtype != object and not pd.api.types.is_string_dtype(X[c]):
            X_bal[c] = X_bal[c].astype(X[c].dtype)
        elif pd.api.types.is_integer_dtype(X[c]):
            X_bal[c] = np.round(X_bal[c].astype(float)).astype(X[c].dtype)
        elif pd.api.types.is_float_dtype(X[c]):
            X_bal[c] = X_bal[c].astype(float)
    return X_bal, pd.Series(y_bal, name=OBJETIVO).reset_index(drop=True)


# %% [markdown]
# ### 6.3 Normalización: conversión de categóricas (CC) y escalado (ED)
# * **CC (siempre):** *One-Hot Encoding* para las nominales (`city_zone`, `road_type`, `peak_period`,
#   `weather_condition`, `iot_sensor_health`). No tienen orden, y un código entero les inventaría
#   distancias falsas. *Codificación ordinal* para `road_condition` (Good < Wet < Poor), cuyo orden de
#   severidad sí es real. Las binarias ya son 0/1.
# * **ED (casos 5–8):** `StandardScaler` sobre todas las columnas no one-hot. Se prefirió frente a
#   MinMaxScaler porque con outliers (inyectados en los casos 3, 4, 7 y 8) MinMax comprime casi todos los
#   datos en un rango muy estrecho. La estandarización es menos sensible a eso y además es coherente con
#   la detección por Z-score. Se ajusta (fit) solo con entrenamiento y se aplica (transform) a validación y
#   prueba.

# %%
def construir_transformador(escalar):
    """ColumnTransformer con One-Hot (nominales), Ordinal (ordinales) y, si `escalar`, StandardScaler."""
    pasos = [("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), NOMINALES)]
    ordinal = OrdinalEncoder(categories=[ORDINALES[c] for c in ORDINALES])
    if escalar:
        pasos.append(("ordinal", Pipeline([("cod", ordinal), ("esc", StandardScaler())]), list(ORDINALES)))
        pasos.append(("numericas", StandardScaler(), NUMERICAS))
    else:
        pasos.append(("ordinal", ordinal, list(ORDINALES)))
        pasos.append(("numericas", "passthrough", NUMERICAS))
    return ColumnTransformer(pasos, verbose_feature_names_out=False)


# %% [markdown]
# ## 7. Construcción de los 8 casos (figura 1)
# Orden de aplicación sobre el **entrenamiento**, siempre en este orden:
# 1. Tratamiento de outliers reales (recorte Z-score) → base limpia común.
# 2. Balanceo SMOTENC (si BALANCEO = SI).
# 3. Inyección del 5 % de outliers (si OUTLIERS = SI). Se hace después del balanceo para que los atípicos
#    sean exactamente el 5 % del entrenamiento final y SMOTE no los propague por interpolación.
# 4. Ajuste del transformador CC (+ ED si aplica) con el entrenamiento y transformación de validación y prueba.
#
# Validación y prueba no se modifican (son registros reales). Solo reciben la transformación CC/ED ajustada
# con el entrenamiento. Por eso **la prueba es la misma en los 8 casos**. Los casos *i* e *i+4* comparten
# exactamente los mismos registros de entrenamiento y solo se diferencian en el escalado.

# %%
CASOS = [
    {"caso": i + 1, "escalado": esc, "outliers": out, "balanceo": bal}
    for i, (esc, out, bal) in enumerate([(e, o, b) for e in (False, True) for o in (False, True) for b in (False, True)])
]


def etiqueta_caso(c):
    """Texto corto del caso, p. ej. 'C8 · ED(SI) · Out(SI) · Bal(SI)'."""
    s = lambda v: "SI" if v else "NO"
    return f"C{c['caso']} · ED({s(c['escalado'])}) · Out({s(c['outliers'])}) · Bal({s(c['balanceo'])})"


tratador_outliers = TratadorOutliersZ().fit(X_train_raw)
mascara_real = tratador_outliers.mascara(X_train_raw)
print(f"Outliers reales detectados en entrenamiento (|z|>3): {int(mascara_real.any(axis=1).sum())} filas "
      f"({100 * mascara_real.any(axis=1).mean():.2f} %), {int(mascara_real.values.sum())} valores")
display(mascara_real.sum().to_frame("valores_atipicos").T)
X_train_base = tratador_outliers.transform(X_train_raw)


def construir_registros_entrenamiento(outliers, balanceo):
    """Devuelve (X_train, y_train, registro_inyección) para la combinación outliers/balanceo dada."""
    X, y = X_train_base.copy(), y_train_raw.copy()
    if balanceo:
        X, y = balancear_smotenc(X, y)
    registro = pd.DataFrame()
    if outliers:
        X, registro = inyectar_outliers(X, tratador_outliers, fraccion=0.05)
    return X.reset_index(drop=True), y.reset_index(drop=True), registro


def preparar_caso(caso):
    """Construye todas las matrices de un caso: entrenamiento modificado y val/prueba transformados."""
    X_tr_df, y_tr, registro = construir_registros_entrenamiento(caso["outliers"], caso["balanceo"])
    transformador = construir_transformador(caso["escalado"]).fit(X_tr_df)
    return {
        **caso, "etiqueta": etiqueta_caso(caso), "transformador": transformador,
        "X_train_df": X_tr_df, "y_train": y_tr, "registro_outliers": registro,
        "X_train": transformador.transform(X_tr_df), "X_val": transformador.transform(X_val_raw),
        "X_test": transformador.transform(X_test_raw), "columnas": list(transformador.get_feature_names_out()),
    }


datos_casos = {c["caso"]: preparar_caso(c) for c in CASOS}

resumen_casos = []
for k, d in datos_casos.items():
    dist = d["y_train"].value_counts().reindex(CLASES)
    pct_out = 100 * tratador_outliers.mascara(d["X_train_df"]).any(axis=1).mean()
    resumen_casos.append({"caso": k, "ED": "SI" if d["escalado"] else "NO", "OUTLIERS": "SI" if d["outliers"] else "NO",
                          "BALANCEO": "SI" if d["balanceo"] else "NO", "n_train": len(d["y_train"]),
                          **{f"%{c}": round(100 * dist[c] / len(d["y_train"]), 1) for c in CLASES},
                          "%filas_outlier": round(pct_out, 2), "n_variables": d["X_train"].shape[1]})
resumen_casos = pd.DataFrame(resumen_casos).set_index("caso")
display(resumen_casos)
guardar_tabla(resumen_casos, "resumen_casos")

# %%
reg = datos_casos[8]["registro_outliers"]
print(f"Caso 8: {reg.fila.nunique()} registros con outliers inyectados "
      f"({100 * reg.fila.nunique() / len(datos_casos[8]['y_train']):.2f} % del entrenamiento), {len(reg)} valores modificados.")
display(reg.groupby("variable").size().sort_values(ascending=False).to_frame("valores_inyectados").T)
display(reg.head(8).round(2))
guardar_tabla(reg.round(3), "outliers_inyectados_caso8")

# %% [markdown]
# ### Tres registros de prueba fijos
# Para cada caso y cada técnica se muestra la predicción sobre los **mismos tres registros del conjunto de
# prueba** (uno de cada una de tres clases distintas), elegidos con la semilla del equipo.

# %%
rng_muestras = np.random.default_rng(SEMILLA)
IDX_PRUEBA_3 = [int(rng_muestras.choice(np.where(y_test.values == c)[0])) for c in ["Low", "High", "Severe"]]
display(X_test_raw.loc[IDX_PRUEBA_3, ["city_zone", "road_type", "hour", "vehicle_count", "average_speed",
                                     "traffic_density", "queue_length"]].assign(real=y_test[IDX_PRUEBA_3].values))

# %% [markdown]
# ---
# # MÓDULO DE APRENDIZAJE SUPERVISADO
# ## 8. Funciones de entrenamiento y evaluación
# * **Métricas:** accuracy, y precisión, recall y F1 con promedio **macro** (cada clase pesa igual). Con un
#   test 70/10/10/10, el promedio ponderado o el accuracy premiarían a un modelo que solo acierta `Low`.
# * **Selección de hiperparámetros:** con el conjunto de validación (F1-macro), salvo KNN, donde el
#   enunciado pide validación cruzada de 5 folds.

# %%
NOMBRES_ALG = list(COLOR_ALG)


def metricas(y_true, y_pred):
    """Accuracy y precisión/recall/F1 macro."""
    return {"accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "f1": f1_score(y_true, y_pred, average="macro", zero_division=0)}


def entrenar_arbol(d, profundidades=range(1, 21)):
    """Árbol de decisión: curvas de max_depth (Gini y Entropía) en train/val y modelo final Gini.

    Devuelve (modelo_final, curvas DataFrame, profundidad_elegida).
    """
    filas = []
    for criterio in ("gini", "entropy"):
        for p in profundidades:
            m = DecisionTreeClassifier(criterion=criterio, max_depth=p, random_state=SEMILLA).fit(d["X_train"], d["y_train"])
            filas.append({"criterio": criterio, "max_depth": p,
                          "acc_train": accuracy_score(d["y_train"], m.predict(d["X_train"])),
                          "acc_val": accuracy_score(y_val, m.predict(d["X_val"])),
                          "f1_val": f1_score(y_val, m.predict(d["X_val"]), average="macro")})
    curvas = pd.DataFrame(filas)
    gini = curvas[curvas.criterio == "gini"]
    mejor = int(gini.loc[gini.f1_val.idxmax(), "max_depth"])
    modelo = DecisionTreeClassifier(criterion="gini", max_depth=mejor, random_state=SEMILLA).fit(d["X_train"], d["y_train"])
    return modelo, curvas, mejor


def entrenar_knn(d, ks=range(1, 32, 2)):
    """KNN: elige k por accuracy promedio de validación cruzada estratificada de 5 folds.

    La CV se hace sobre los registros reales del entrenamiento (base tratada) y el balanceo SMOTENC, la
    inyección de outliers y el ajuste de CC/ED se repiten DENTRO de cada fold, solo con la parte de
    entrenamiento del fold. Hacer la CV directamente sobre el conjunto ya balanceado filtraría información:
    un registro sintético de validación tendría a sus 'padres' en el entrenamiento y la CV favorecería k=1.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
    aciertos = {k: [] for k in ks}
    for i, (idx_tr, idx_va) in enumerate(cv.split(X_train_base, y_train_raw)):
        X_f, y_f = X_train_base.iloc[idx_tr].reset_index(drop=True), y_train_raw.iloc[idx_tr].reset_index(drop=True)
        if d["balanceo"]:
            X_f, y_f = balancear_smotenc(X_f, y_f, semilla=SEMILLA + i)
        if d["outliers"]:
            X_f, _ = inyectar_outliers(X_f, tratador_outliers, fraccion=0.05, semilla=SEMILLA + i)
        transf = construir_transformador(d["escalado"]).fit(X_f)
        Xtr_f, Xva_f = transf.transform(X_f), transf.transform(X_train_base.iloc[idx_va])
        for k in ks:
            pred = KNeighborsClassifier(n_neighbors=k).fit(Xtr_f, y_f).predict(Xva_f)
            aciertos[k].append(accuracy_score(y_train_raw.iloc[idx_va], pred))
    curva = pd.DataFrame([{"k": k, "acc_cv": np.mean(v), "std_cv": np.std(v)} for k, v in aciertos.items()])
    mejor = int(curva.loc[curva.acc_cv.idxmax(), "k"])
    return KNeighborsClassifier(n_neighbors=mejor).fit(d["X_train"], d["y_train"]), curva, mejor


def entrenar_svm(d, valores_C=(1, 10, 100)):
    """SVM con kernel RBF (gamma='scale'): elige C por F1-macro en validación."""
    res = []
    for C in valores_C:
        m = SVC(kernel="rbf", C=C, gamma="scale", cache_size=1000, random_state=SEMILLA).fit(d["X_train"], d["y_train"])
        res.append((f1_score(y_val, m.predict(d["X_val"]), average="macro"), C, m))
    f1v, C, modelo = max(res, key=lambda t: t[0])
    return modelo, pd.DataFrame([(c, f) for f, c, _ in res], columns=["C", "f1_val"]), C


# %% [markdown]
# ### Red neuronal (Keras)
# **Arquitectura y proceso de configuración.** Es un perceptrón multicapa totalmente conectado con salida
# *softmax* de 4 neuronas y pérdida *sparse categorical cross-entropy*, optimizado con Adam
# (lr = 1e-3, lotes de 64).
#
# **Regularización vista en clase:**
# * penalización **L2** en las capas densas (en todas las arquitecturas)
# * **Dropout** (en todas las arquitecturas)
# * **Batch Normalization** (en las arquitecturas B y C)
#
# **Callbacks:**
# * **EarlyStopping**: vigila `val_loss` con paciencia 15 y restaura los mejores pesos.
# * **ReduceLROnPlateau**: multiplica el lr por 0.5 tras 5 épocas sin mejora.
#
# Antes de entrenar los 8 casos se comparan 3 arquitecturas en el caso 8 con el conjunto de validación
# (sección 9). La ganadora se usa en todos los casos para que la comparación entre preprocesamientos sea
# justa.

# %%
ARQUITECTURAS = {
    "A: 64 · L2 · Dropout 0.2": {"capas": [64], "dropout": 0.2, "l2": 1e-4, "bn": False},
    "B: 128-64 · L2 · BN · Dropout 0.3": {"capas": [128, 64], "dropout": 0.3, "l2": 1e-4, "bn": True},
    "C: 256-128-64 · L2 · BN · Dropout 0.4": {"capas": [256, 128, 64], "dropout": 0.4, "l2": 1e-4, "bn": True},
}
ID_CLASE = {c: i for i, c in enumerate(CLASES)}


def construir_red(n_entradas, capas, dropout, l2, bn):
    """Crea y compila un MLP Keras con regularización L2, BatchNormalization y Dropout."""
    modelo = keras.Sequential([layers.Input(shape=(n_entradas,))])
    for unidades in capas:
        modelo.add(layers.Dense(unidades, activation="relu",
                                kernel_regularizer=regularizers.l2(l2) if l2 else None))
        if bn:
            modelo.add(layers.BatchNormalization())
        modelo.add(layers.Dropout(dropout))
    modelo.add(layers.Dense(len(CLASES), activation="softmax"))
    modelo.compile(optimizer=keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return modelo


class RedNeuronal:
    """Envoltorio con interfaz tipo scikit-learn (fit / predict / predict_proba) para el MLP de Keras."""

    def __init__(self, config, epocas=200, lote=64):
        self.config, self.epocas, self.lote = config, epocas, lote

    def fit(self, X, y, X_val, y_val):
        fijar_semillas()
        self.modelo = construir_red(X.shape[1], **self.config)
        cbs = [callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
               callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5)]
        self.historia = self.modelo.fit(X.astype("float32"), y.map(ID_CLASE).values,
                                        validation_data=(X_val.astype("float32"), y_val.map(ID_CLASE).values),
                                        epochs=self.epocas, batch_size=self.lote, callbacks=cbs, verbose=0).history
        return self

    def predict_proba(self, X):
        return self.modelo.predict(X.astype("float32"), verbose=0)

    def predict(self, X):
        return np.array(CLASES)[self.predict_proba(X).argmax(axis=1)]


# %% [markdown]
# ## 9. Configuración de la red neuronal (comparación de arquitecturas en el caso 8)

# %%
comparacion_redes = []
for nombre, cfg in ARQUITECTURAS.items():
    red = RedNeuronal(cfg).fit(datos_casos[8]["X_train"], datos_casos[8]["y_train"], datos_casos[8]["X_val"], y_val)
    comparacion_redes.append({"arquitectura": nombre, "parámetros": red.modelo.count_params(),
                              "épocas": len(red.historia["loss"]),
                              "f1_val": f1_score(y_val, red.predict(datos_casos[8]["X_val"]), average="macro"),
                              "acc_val": accuracy_score(y_val, red.predict(datos_casos[8]["X_val"]))})
comparacion_redes = pd.DataFrame(comparacion_redes).set_index("arquitectura")
display(comparacion_redes.round(4))
guardar_tabla(comparacion_redes.round(4), "configuracion_red")
ARQ_ELEGIDA = comparacion_redes.f1_val.idxmax()
print("Arquitectura elegida:", ARQ_ELEGIDA)
construir_red(datos_casos[8]["X_train"].shape[1], **ARQUITECTURAS[ARQ_ELEGIDA]).summary()

# %% [markdown]
# ## 10. Funciones de visualización supervisada (sección 2.4)
# PCA se usa **solo para visualizar**. Los modelos se entrenan siempre con todas las variables.
# * **KNN:** se proyecta el entrenamiento a 2D. Se marca un registro de prueba y sus *k* vecinos,
#   calculados por el modelo en el espacio completo.
# * **SVM:** se crea una malla en el plano PCA y se lleva al espacio original con `inverse_transform`. El
#   SVM entrenado (en todas las variables) predice cada punto. El color de la malla es la región de
#   decisión del modelo real sobre el plano principal de los datos.

# %%
def dispersion_clases(ax, Z, y, alpha=0.35, s=10):
    """Dibuja puntos 2D coloreados (rampa ordinal) y con marcador distinto por clase."""
    for c in CLASES:
        m = np.asarray(y) == c
        ax.scatter(Z[m, 0], Z[m, 1], s=s, alpha=alpha, color=COLOR_CLASE[c], marker=MARCA_CLASE[c],
                   label=c, edgecolors="none")


def leyenda(ax, **kwargs):
    """Leyenda con marcadores opacos y de tamaño uniforme (los puntos del gráfico pueden ser transparentes)."""
    ley = ax.legend(**kwargs)
    for h in ley.legend_handles:
        if hasattr(h, "set_alpha"):
            h.set_alpha(1)
        if hasattr(h, "set_sizes"):
            h.set_sizes([30])
    return ley


def limites_robustos(Z, margen=0.05):
    """Límites del gráfico entre los percentiles 1 y 99 (evita que los outliers aplasten la vista)."""
    lo, hi = np.percentile(Z, 1, axis=0), np.percentile(Z, 99, axis=0)
    r = hi - lo
    return lo - margen * r, hi + margen * r


def graficar_visualizaciones_caso(d, modelos, curvas, num):
    """Figura compuesta del caso: importancia (árbol), max_depth vs accuracy, k vs accuracy,
    vecinos KNN (PCA) y regiones de decisión SVM (PCA)."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    arbol, knn, svm = modelos["Árbol de decisión"], modelos["KNN"], modelos["SVM"]

    # (a) Importancia de características
    imp = pd.Series(arbol.feature_importances_, index=d["columnas"]).sort_values().tail(12)
    axes[0, 0].barh(imp.index, imp.values, color=COLOR_ALG["Árbol de decisión"], height=0.6)
    axes[0, 0].set_title("Árbol: importancia de características (top 12)")
    axes[0, 0].tick_params(axis="y", labelsize=8)
    axes[0, 0].grid(axis="y", visible=False)

    # (b) max_depth vs accuracy (Gini vs Entropía, train vs validación)
    ax = axes[0, 1]
    cv = curvas["arbol"]
    for criterio, color in (("gini", COLOR_PAR[0]), ("entropy", COLOR_PAR[1])):
        t = cv[cv.criterio == criterio]
        ax.plot(t.max_depth, t.acc_train, color=color, linestyle="--", linewidth=1.5, label=f"{criterio} · train")
        ax.plot(t.max_depth, t.acc_val, color=color, label=f"{criterio} · validación")
    ax.axvline(curvas["prof"], color=TINTA2, linewidth=1, linestyle=":")
    ax.annotate(f"elegido: {curvas['prof']}", (curvas["prof"], ax.get_ylim()[0]), xytext=(4, 6),
                textcoords="offset points", color=TINTA2, fontsize=8)
    ax.set_title("Árbol: max_depth vs accuracy")
    ax.set_xlabel("max_depth"); ax.set_ylabel("accuracy"); ax.legend(fontsize=8)

    # (c) k vs accuracy CV
    ax = axes[0, 2]
    ax.plot(curvas["knn"].k, curvas["knn"].acc_cv, color=COLOR_ALG["KNN"], marker="o", markersize=5)
    ax.axvline(curvas["k"], color=TINTA2, linewidth=1, linestyle=":")
    ax.set_title(f"KNN: k vs accuracy (CV 5 folds) — k elegido = {curvas['k']}")
    ax.set_xlabel("k"); ax.set_ylabel("accuracy promedio CV")

    # PCA (solo visualización)
    pca = PCA(n_components=2, random_state=SEMILLA).fit(d["X_train"])
    Z_tr = pca.transform(d["X_train"])
    lo, hi = limites_robustos(Z_tr)

    # (d) Vecinos KNN de un registro de prueba
    ax = axes[1, 0]
    dispersion_clases(ax, Z_tr, d["y_train"], alpha=0.18, s=8)
    consulta = d["X_test"][IDX_PRUEBA_3[1]:IDX_PRUEBA_3[1] + 1]
    _, vecinos = knn.kneighbors(consulta)
    Zq, Zv = pca.transform(consulta), Z_tr[vecinos[0]]
    for p in Zv:
        ax.plot([Zq[0, 0], p[0]], [Zq[0, 1], p[1]], color=TINTA2, linewidth=0.6, alpha=0.6)
    ax.scatter(Zv[:, 0], Zv[:, 1], s=60, facecolors="none", edgecolors=COLOR_ALG["KNN"], linewidths=1.6,
               label=f"{knn.n_neighbors} vecinos")
    ax.scatter(Zq[:, 0], Zq[:, 1], s=220, marker="*", color=COLOR_ALG["KNN"], edgecolors=TINTA, linewidths=0.8,
               label=f"consulta (real: {y_test[IDX_PRUEBA_3[1]]}, pred: {knn.predict(consulta)[0]})")
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
    ax.set_title("KNN: vecinos de un registro de prueba (PCA 2D)")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); leyenda(ax, fontsize=7, loc="best")

    # (e) Regiones de decisión SVM
    ax = axes[1, 1]
    xx, yy = np.meshgrid(np.linspace(lo[0], hi[0], 220), np.linspace(lo[1], hi[1], 220))
    malla = pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()])
    pred = pd.Series(svm.predict(malla)).map(ID_CLASE).values.reshape(xx.shape)
    cmap = matplotlib.colors.ListedColormap([COLOR_CLASE[c] for c in CLASES])
    ax.contourf(xx, yy, pred, levels=np.arange(-0.5, 4), cmap=cmap, alpha=0.35)
    dispersion_clases(ax, pca.transform(d["X_test"]), y_test, alpha=0.8, s=12)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
    ax.set_title(f"SVM (RBF, C={curvas['C']}): regiones de decisión + prueba (PCA 2D)")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); leyenda(ax, fontsize=7)

    # (f) Curva de aprendizaje de la red
    ax = axes[1, 2]
    h = modelos["Red neuronal"].historia
    ax.plot(h["loss"], color=COLOR_ALG["Red neuronal"], linestyle="--", linewidth=1.5, label="pérdida train")
    ax.plot(h["val_loss"], color=COLOR_ALG["Red neuronal"], label="pérdida validación")
    ax.set_title(f"Red neuronal: pérdida por época ({len(h['loss'])} épocas, early stopping)")
    ax.set_xlabel("época"); ax.set_ylabel("cross-entropy"); ax.legend(fontsize=8)
    ax.set_yscale("log")

    fig.suptitle(f"Visualización de resultados — {d['etiqueta']}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    guardar_figura(fig, f"sup_caso{num}_visualizaciones")


def graficar_arbol(modelo, d, num, profundidad_visible=3):
    """Dibuja el árbol solución (Gini) mostrando los primeros niveles e imprime sus reglas."""
    fig, ax = plt.subplots(figsize=(22, 8))
    plot_tree(modelo, max_depth=profundidad_visible, feature_names=d["columnas"], class_names=list(modelo.classes_),
              filled=False, impurity=True, proportion=True, rounded=True, fontsize=8, ax=ax)
    ax.set_title(f"Árbol de decisión solución (Gini, max_depth={modelo.get_depth()}) — primeros "
                 f"{profundidad_visible} niveles — {d['etiqueta']}")
    guardar_figura(fig, f"sup_caso{num}_arbol")
    reglas = export_text(modelo, feature_names=d["columnas"], max_depth=profundidad_visible, decimals=2)
    print(reglas)
    return reglas


# %% [markdown]
# ## 11. Entrenamiento y evaluación de los 8 casos

# %%
resultados_sup, predicciones_3, modelos_casos, reportes = [], [], {}, {}
for num, d in datos_casos.items():
    t0 = time.time()
    print("=" * 100, f"\n{d['etiqueta']}  —  entrenamiento: {d['X_train'].shape}")
    arbol, curva_arbol, prof = entrenar_arbol(d)
    knn, curva_knn, k = entrenar_knn(d)
    svm, curva_svm, C = entrenar_svm(d)
    red = RedNeuronal(ARQUITECTURAS[ARQ_ELEGIDA]).fit(d["X_train"], d["y_train"], d["X_val"], y_val)
    modelos = {"Árbol de decisión": arbol, "KNN": knn, "SVM": svm, "Red neuronal": red}
    modelos_casos[num] = modelos
    print(f"Hiperparámetros → árbol max_depth={prof} | KNN k={k} | SVM C={C} | red épocas={len(red.historia['loss'])}")

    for nombre, m in modelos.items():
        pred = m.predict(d["X_test"])
        resultados_sup.append({"caso": num, "ED": "SI" if d["escalado"] else "NO",
                               "OUTLIERS": "SI" if d["outliers"] else "NO",
                               "BALANCEO": "SI" if d["balanceo"] else "NO", "algoritmo": nombre, **metricas(y_test, pred)})
        reportes[(num, nombre)] = classification_report(y_test, pred, labels=CLASES, digits=3, output_dict=True)
        p3 = m.predict(d["X_test"][IDX_PRUEBA_3])
        for j, idx in enumerate(IDX_PRUEBA_3):
            predicciones_3.append({"caso": num, "algoritmo": nombre, "registro_prueba": idx,
                                   "real": y_test[idx], "predicho": p3[j]})

    graficar_visualizaciones_caso(d, modelos, {"arbol": curva_arbol, "prof": prof, "knn": curva_knn, "k": k, "C": C}, num)
    graficar_arbol(arbol, d, num)
    tabla_caso = pd.DataFrame([r for r in resultados_sup if r["caso"] == num]).set_index("algoritmo")
    display(tabla_caso[["accuracy", "precision", "recall", "f1"]].round(4))
    print(f"Tiempo del caso: {time.time() - t0:.1f} s")

resultados_sup = pd.DataFrame(resultados_sup)
predicciones_3 = pd.DataFrame(predicciones_3)

# %% [markdown]
# ## 12. Tabla de la figura 1

# %%
figura1 = resultados_sup.pivot_table(index=["caso", "ED", "OUTLIERS", "BALANCEO"], columns="algoritmo",
                                     values=["accuracy", "precision", "recall", "f1"])
figura1 = figura1.swaplevel(0, 1, axis=1).reindex(columns=pd.MultiIndex.from_product(
    [NOMBRES_ALG, ["accuracy", "precision", "recall", "f1"]]))
figura1.columns.names = ["algoritmo", "métrica"]
display(figura1.round(4))
guardar_tabla(figura1.round(4), "figura1_supervisado")
guardar_tabla(resultados_sup.round(4), "resultados_supervisado_largo")

# %% [markdown]
# ### Predicción de los tres registros de prueba en cada caso y técnica

# %%
tabla_p3 = predicciones_3.assign(r=lambda t: "reg " + t.registro_prueba.astype(str) + " (real " + t.real + ")") \
    .pivot_table(index=["caso", "algoritmo"], columns="r", values="predicho", aggfunc="first")
display(tabla_p3)
guardar_tabla(tabla_p3, "predicciones_3_registros_supervisado")
print("Aciertos sobre los 3 registros:",
      f"{(predicciones_3.real == predicciones_3.predicho).sum()} de {len(predicciones_3)}")

# %% [markdown]
# ## 13. Gráficas comparativas de F1-score

# %%
def barras_agrupadas(ax, tabla, colores, titulo, etiqueta_y="F1-score (macro)"):
    """Diagrama de barras agrupadas: filas de `tabla` = algoritmos, columnas = niveles comparados."""
    n = tabla.shape[1]
    ancho = 0.8 / n
    x = np.arange(len(tabla))
    for i, col in enumerate(tabla.columns):
        barras = ax.bar(x + (i - (n - 1) / 2) * ancho, tabla[col].values, width=ancho * 0.92,
                        color=colores[i], label=col, edgecolor="#fcfcfb", linewidth=1)
        for b in barras:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005, f"{b.get_height():.3f}",
                    ha="center", va="bottom", fontsize=7.5, color=TINTA2)
    ax.set_xticks(x, tabla.index)
    ax.set_title(titulo)
    ax.set_ylabel(etiqueta_y)
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=8, loc="lower right")


fig, ax = plt.subplots(figsize=(8, 4.2))
f1_max = resultados_sup.groupby("algoritmo").f1.max().reindex(NOMBRES_ALG)
barras = ax.bar(NOMBRES_ALG, f1_max.values, color=[COLOR_ALG[a] for a in NOMBRES_ALG], width=0.55)
for b, a in zip(barras, NOMBRES_ALG):
    caso = int(resultados_sup.loc[resultados_sup[resultados_sup.algoritmo == a].f1.idxmax(), "caso"])
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.004, f"{b.get_height():.4f}\n(caso {caso})",
            ha="center", va="bottom", fontsize=8.5, color=TINTA2)
ax.set_ylim(min(0.8, f1_max.min() - 0.05), 1.03)
ax.set_title("F1-score máximo por algoritmo (8 casos)")
ax.set_ylabel("F1-score (macro)"); ax.grid(axis="x", visible=False)
guardar_figura(fig, "sup_f1_maximo")

agrupaciones = [("ED", "normalización (escalado)", {"NO": "Sin ED (NO)", "SI": "Con ED (SI)"}),
                ("OUTLIERS", "manejo de datos atípicos", {"NO": "Sin outliers (NO)", "SI": "Con outliers (SI)"}),
                ("BALANCEO", "balanceo", {"NO": "Desbalanceado (NO)", "SI": "Balanceado (SI)"})]
for col, nombre, renombre in agrupaciones:
    tabla = resultados_sup.pivot_table(index="algoritmo", columns=col, values="f1", aggfunc="mean") \
        .reindex(NOMBRES_ALG)[["NO", "SI"]].rename(columns=renombre)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    barras_agrupadas(ax, tabla, COLOR_PAR, f"F1-score medio por {nombre} y algoritmo")
    ax.set_ylim(max(0, tabla.values.min() - 0.1), 1.02)
    guardar_figura(fig, f"sup_f1_medio_{col.lower()}")
    display(tabla.round(4))
    guardar_tabla(tabla.round(4), f"f1_medio_{col.lower()}")

# %% [markdown]
# ## 14. Matrices de confusión y reporte por clase (caso 8 y resumen de todos los casos)

# %%
d8 = datos_casos[8]
fig, axes = plt.subplots(1, 4, figsize=(20, 4.6))
for ax, (nombre, m) in zip(axes, modelos_casos[8].items()):
    cm = confusion_matrix(y_test, m.predict(d8["X_test"]), labels=CLASES)
    ConfusionMatrixDisplay(cm, display_labels=CLASES).plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(nombre); ax.grid(False)
fig.suptitle("Matrices de confusión — caso 8 (ED SI · Outliers SI · Balanceo SI) — conjunto de prueba",
             fontweight="bold")
fig.tight_layout()
guardar_figura(fig, "sup_caso8_matrices_confusion")

for nombre, m in modelos_casos[8].items():
    print(f"--- {nombre} (caso 8) ---")
    print(classification_report(y_test, m.predict(d8["X_test"]), labels=CLASES, digits=3))

# %%
por_clase = pd.DataFrame([{"caso": num, "algoritmo": alg, "clase": c, "recall": rep[c]["recall"],
                           "precision": rep[c]["precision"], "f1": rep[c]["f1-score"]}
                          for (num, alg), rep in reportes.items() for c in CLASES])
recall_clase = por_clase.pivot_table(index=["caso", "algoritmo"], columns="clase", values="recall")[CLASES]
display(recall_clase.round(3))
guardar_tabla(por_clase.round(4), "metricas_por_clase")

fig, axes = plt.subplots(2, 4, figsize=(20, 9))
for ax, num in zip(axes.ravel(), datos_casos):
    m = modelos_casos[num]["SVM"]
    cm = confusion_matrix(y_test, m.predict(datos_casos[num]["X_test"]), labels=CLASES, normalize="true")
    ConfusionMatrixDisplay(cm, display_labels=CLASES).plot(ax=ax, cmap="Blues", colorbar=False, values_format=".2f")
    ax.set_title(datos_casos[num]["etiqueta"], fontsize=9); ax.grid(False)
fig.suptitle("Matrices de confusión normalizadas por clase real — SVM en los 8 casos", fontweight="bold")
fig.tight_layout()
guardar_figura(fig, "sup_matrices_svm_8casos")

fig, axes = plt.subplots(2, 4, figsize=(20, 9))
for ax, num in zip(axes.ravel(), datos_casos):
    m = modelos_casos[num]["KNN"]
    cm = confusion_matrix(y_test, m.predict(datos_casos[num]["X_test"]), labels=CLASES, normalize="true")
    ConfusionMatrixDisplay(cm, display_labels=CLASES).plot(ax=ax, cmap="Blues", colorbar=False, values_format=".2f")
    ax.set_title(datos_casos[num]["etiqueta"], fontsize=9); ax.grid(False)
fig.suptitle("Matrices de confusión normalizadas por clase real — KNN en los 8 casos", fontweight="bold")
fig.tight_layout()
guardar_figura(fig, "sup_matrices_knn_8casos")

# %% [markdown]
# ## 15. Curvas ROC y Precision-Recall — caso 8
# Problema multiclase → estrategia **uno-contra-resto**. Puntajes usados:
# * `predict_proba` para el árbol, KNN y la red.
# * `decision_function` para SVM (evita el costoso ajuste de Platt y el ranking es el mismo).
#
# Se grafica el promedio **macro** (cada clase pesa igual) y además las curvas por clase.

# %%
def puntajes(modelo, X):
    """Puntajes por clase (columnas en el orden de CLASES) para construir curvas ROC/PR."""
    if isinstance(modelo, SVC):
        s = modelo.decision_function(X)
    else:
        s = modelo.predict_proba(X)
    if isinstance(modelo, RedNeuronal):
        return s
    orden = [list(modelo.classes_).index(c) for c in CLASES]
    return s[:, orden]


Y_bin = label_binarize(y_test, classes=CLASES)
rejilla = np.linspace(0, 1, 501)
resumen_curvas, curvas_clase = [], {}
fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(14, 5.6))
for nombre, m in modelos_casos[8].items():
    S = puntajes(m, d8["X_test"])
    tpr_m, prec_m = np.zeros_like(rejilla), np.zeros_like(rejilla)
    curvas_clase[nombre] = []
    for i, c in enumerate(CLASES):
        fpr, tpr, _ = roc_curve(Y_bin[:, i], S[:, i])
        prec, rec, _ = precision_recall_curve(Y_bin[:, i], S[:, i])
        tpr_m += np.interp(rejilla, fpr, tpr)
        # precisión interpolada: para cada recall r, la máxima precisión alcanzable con recall >= r
        # (precision_recall_curve entrega el recall en orden decreciente, por eso se acumula en ese orden)
        prec_interp = np.maximum.accumulate(prec)
        prec_m += np.interp(rejilla, rec[::-1], prec_interp[::-1])
        curvas_clase[nombre].append((fpr, tpr, auc(fpr, tpr), rec, prec, average_precision_score(Y_bin[:, i], S[:, i])))
    tpr_m /= len(CLASES); prec_m /= len(CLASES)
    auc_macro = np.mean([cc[2] for cc in curvas_clase[nombre]])
    ap_macro = average_precision_score(Y_bin, S, average="macro")
    resumen_curvas.append({"algoritmo": nombre, "AUC_ROC_macro": auc_macro, "AP_macro": ap_macro,
                           **{f"AUC_{c}": curvas_clase[nombre][i][2] for i, c in enumerate(CLASES)},
                           **{f"AP_{c}": curvas_clase[nombre][i][5] for i, c in enumerate(CLASES)}})
    ax_roc.plot(rejilla, tpr_m, color=COLOR_ALG[nombre], label=f"{nombre} (AUC={auc_macro:.4f})")
    ax_pr.plot(rejilla, prec_m, color=COLOR_ALG[nombre], label=f"{nombre} (AP={ap_macro:.4f})")
ax_roc.plot([0, 1], [0, 1], color=COLOR_RUIDO, linestyle=":", linewidth=1, label="azar")
ax_roc.set_title("Curva ROC macro-promedio (uno-contra-resto) — caso 8")
ax_roc.set_xlabel("Tasa de falsos positivos"); ax_roc.set_ylabel("Tasa de verdaderos positivos (recall)")
ax_roc.legend(fontsize=8, loc="lower right")
ax_pr.axhline(Y_bin.mean(), color=COLOR_RUIDO, linestyle=":", linewidth=1, label="azar (prevalencia media)")
ax_pr.set_title("Curva Precision-Recall macro-promedio — caso 8")
ax_pr.set_xlabel("Recall"); ax_pr.set_ylabel("Precisión"); ax_pr.legend(fontsize=8, loc="lower left")
fig.tight_layout()
guardar_figura(fig, "sup_caso8_roc_pr")

resumen_curvas = pd.DataFrame(resumen_curvas).set_index("algoritmo")
display(resumen_curvas.round(4))
guardar_tabla(resumen_curvas.round(4), "roc_pr_caso8")

fig, axes = plt.subplots(2, 4, figsize=(20, 9))
for i, c in enumerate(CLASES):
    for nombre in modelos_casos[8]:
        fpr, tpr, a, rec, prec, ap = curvas_clase[nombre][i]
        axes[0, i].plot(fpr, tpr, color=COLOR_ALG[nombre], label=f"{nombre} ({a:.3f})")
        axes[1, i].plot(rec, prec, color=COLOR_ALG[nombre], label=f"{nombre} ({ap:.3f})")
    axes[0, i].set_title(f"ROC — clase {c}"); axes[0, i].set_xlabel("FPR"); axes[0, i].set_ylabel("TPR")
    axes[1, i].set_title(f"Precision-Recall — clase {c}"); axes[1, i].set_xlabel("Recall"); axes[1, i].set_ylabel("Precisión")
    axes[0, i].legend(fontsize=7, loc="lower right"); axes[1, i].legend(fontsize=7, loc="lower left")
    axes[0, i].set_xlim(-0.01, 0.4); axes[0, i].set_ylim(0.6, 1.01)
fig.suptitle("Curvas ROC (zoom) y Precision-Recall por clase — caso 8", fontweight="bold")
fig.tight_layout()
guardar_figura(fig, "sup_caso8_roc_pr_por_clase")

# %% [markdown]
# ### Resumen de las curvas de aprendizaje de la red neuronal en los 8 casos

# %%
fig, axes = plt.subplots(2, 4, figsize=(20, 8), sharey=False)
for ax, num in zip(axes.ravel(), datos_casos):
    h = modelos_casos[num]["Red neuronal"].historia
    ax.plot(h["accuracy"], color=COLOR_ALG["Red neuronal"], linestyle="--", linewidth=1.4, label="train")
    ax.plot(h["val_accuracy"], color=COLOR_ALG["Red neuronal"], label="validación")
    ax.set_title(datos_casos[num]["etiqueta"], fontsize=9)
    ax.set_xlabel("época"); ax.set_ylabel("accuracy"); ax.legend(fontsize=7)
fig.suptitle("Red neuronal: accuracy por época en los 8 casos", fontweight="bold")
fig.tight_layout()
guardar_figura(fig, "sup_red_curvas_8casos")

# %% [markdown]
# ---
# # MÓDULO DE APRENDIZAJE NO SUPERVISADO
# ## 16. Datasets de la figura 2
# **Correspondencia con el módulo supervisado.** Cada caso *i* de la figura 2 usa exactamente los mismos
# registros del entrenamiento del caso *i* de la figura 1, con el mismo tratamiento de outliers, el mismo
# balanceo SMOTENC y el mismo transformador CC/ED. Después se descarta la variable objetivo:
# * casos con BALANCEO = SI: 4 × 3111 registros
# * casos con BALANCEO = NO: los 4445 de la distribución 70/10/10/10
#
# La etiqueta original se guarda aparte y solo se usa para la **validación externa** (ARI, NMI y tabla de
# contingencia), nunca para entrenar. Los tres registros de prueba se asignan a un cluster con el mismo
# transformador.

# %%
resumen_ns = pd.DataFrame([{"caso": n, "registros": d["X_train"].shape[0], "variables": d["X_train"].shape[1],
                            "ED": "SI" if d["escalado"] else "NO", "OUTLIERS": "SI" if d["outliers"] else "NO",
                            "BALANCEO": "SI" if d["balanceo"] else "NO",
                            "origen": f"X_train del caso {n} supervisado (sin {OBJETIVO})"}
                           for n, d in datos_casos.items()]).set_index("caso")
display(resumen_ns)
guardar_tabla(resumen_ns, "figura2_datasets_no_supervisado")

# %% [markdown]
# ## 17. Funciones de agrupamiento
# * **K-Means:** se recorre K = 2…10 con inicialización k-means++, `n_init=10` y la semilla del equipo. Se
#   grafican la inercia (SSE, método del codo) y el silhouette.
#   * El silhouette del barrido se estima con una muestra de 4000 puntos para reducir el costo, que es
#     O(n²). El de la tabla final se calcula con todos los puntos.
#   * El **K óptimo** es el de mayor silhouette, contrastado con el codo de la inercia.
# * **DBSCAN:**
#   * **MinPts = 2 × número de variables.**
#   * **ε** se toma en el codo del gráfico de k-distancia (k = MinPts). El codo se detecta como el punto
#     más alejado de la recta que une los extremos de la curva normalizada (método *Kneedle*).
#   * El silhouette se calcula **excluyendo el ruido**.
#   * Si con el ε del codo salen menos de 2 clusters (silhouette indefinido), se documenta y se reporta
#     además el mejor ε de un análisis de sensibilidad (0.3×–1.5× ε del codo).

# %%
def barrido_kmeans(X, ks=range(2, 11)):
    """Inercia y silhouette (muestra de 4000) para cada K."""
    filas = []
    for k in ks:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=SEMILLA).fit(X)
        sil = silhouette_score(X, km.labels_, sample_size=min(4000, len(X)), random_state=SEMILLA)
        filas.append({"K": k, "inercia": km.inertia_, "silhouette": sil})
    return pd.DataFrame(filas)


def codo(y):
    """Índice del codo de una curva (punto más alejado de la recta entre sus extremos, curva normalizada)."""
    y = np.asarray(y, dtype=float)
    x = np.linspace(0, 1, len(y))
    yn = (y - y.min()) / (y.max() - y.min() + 1e-12)
    a, b = np.array([x[0], yn[0]]), np.array([x[-1], yn[-1]])
    dist = np.abs(np.cross(b - a, np.c_[x, yn] - a)) / np.linalg.norm(b - a)
    return int(dist.argmax())


def k_distancias(X, min_pts):
    """Distancia al MinPts-ésimo vecino más cercano de cada punto, ordenada ascendentemente."""
    nn = NearestNeighbors(n_neighbors=min_pts).fit(X)
    dist, _ = nn.kneighbors(X)
    return np.sort(dist[:, -1])


def evaluar_dbscan(X, eps, min_pts):
    """Ajusta DBSCAN y devuelve (modelo, n_clusters, %ruido, silhouette sin ruido o NaN)."""
    m = DBSCAN(eps=eps, min_samples=min_pts, n_jobs=-1).fit(X)
    etiquetas = m.labels_
    n_cl = len(set(etiquetas)) - (1 if -1 in etiquetas else 0)
    no_ruido = etiquetas != -1
    sil = silhouette_score(X[no_ruido], etiquetas[no_ruido]) if n_cl >= 2 else np.nan
    return m, n_cl, 100 * (~no_ruido).mean(), sil


def asignar_dbscan(modelo, X_train, X_nuevo):
    """Asigna puntos nuevos al cluster del punto núcleo más cercano si está a distancia ≤ ε; si no, ruido (-1)."""
    nucleos = X_train[modelo.core_sample_indices_]
    if len(nucleos) == 0:
        return np.full(len(X_nuevo), -1)
    dist, idx = NearestNeighbors(n_neighbors=1).fit(nucleos).kneighbors(X_nuevo)
    etiquetas = modelo.labels_[modelo.core_sample_indices_][idx[:, 0]]
    return np.where(dist[:, 0] <= modelo.eps, etiquetas, -1)


def graficar_no_supervisado(d, barrido, k_opt, km, kdist, idx_codo, eps, min_pts, db, num):
    """Figura compuesta: codo, silhouette vs K, k-distancia, clusters K-Means y DBSCAN en PCA 2D."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    ax = axes[0, 0]
    ax.plot(barrido.K, barrido.inercia, color=COLOR_PAR[0], marker="o", markersize=5)
    ax.axvline(k_opt, color=TINTA2, linestyle=":", linewidth=1)
    ax.set_title("K-Means: método del codo (inercia/SSE vs K)"); ax.set_xlabel("K"); ax.set_ylabel("inercia")
    ax = axes[0, 1]
    ax.plot(barrido.K, barrido.silhouette, color=COLOR_PAR[1], marker="o", markersize=5)
    ax.axvline(k_opt, color=TINTA2, linestyle=":", linewidth=1)
    ax.set_title(f"K-Means: silhouette vs K — K óptimo = {k_opt}"); ax.set_xlabel("K"); ax.set_ylabel("silhouette")
    ax = axes[0, 2]
    ax.plot(np.arange(len(kdist)), kdist, color=COLOR_PAR[0], linewidth=1.5)
    ax.scatter([idx_codo], [kdist[idx_codo]], color=COLOR_PAR[1], s=60, zorder=3, label=f"codo: ε={kdist[idx_codo]:.3g}")
    if not np.isclose(eps, kdist[idx_codo]):
        ax.axhline(eps, color=COLOR_PAR[1], linestyle=":", linewidth=1.2, label=f"ε usado = {eps:.3g}")
    ax.set_title(f"DBSCAN: gráfico de k-distancia (k = MinPts = {min_pts})")
    ax.set_xlabel("puntos ordenados"); ax.set_ylabel(f"distancia al {min_pts}-ésimo vecino"); ax.legend(fontsize=8)

    pca = PCA(n_components=2, random_state=SEMILLA).fit(d["X_train"])
    Z = pca.transform(d["X_train"])
    lo, hi = limites_robustos(Z)
    paleta = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
    ax = axes[1, 0]
    for c in range(km.n_clusters):
        m = km.labels_ == c
        ax.scatter(Z[m, 0], Z[m, 1], s=7, alpha=0.4, color=paleta[c % 8], label=f"cluster {c} (n={m.sum()})",
                   edgecolors="none")
    Zc = pca.transform(km.cluster_centers_)
    ax.scatter(Zc[:, 0], Zc[:, 1], marker="X", s=220, color=TINTA, edgecolors="#fcfcfb", linewidths=1.5,
               label="centroides")
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
    ax.set_title(f"K-Means (K={km.n_clusters}) en PCA 2D"); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    leyenda(ax, fontsize=7)
    ax = axes[1, 1]
    etq = db.labels_
    for j, c in enumerate(sorted(set(etq) - {-1})[:8]):
        m = etq == c
        ax.scatter(Z[m, 0], Z[m, 1], s=7, alpha=0.45, color=paleta[j], label=f"cluster {c} (n={m.sum()})",
                   edgecolors="none")
    otros = [c for c in sorted(set(etq) - {-1})][8:]
    if otros:
        m = np.isin(etq, otros)
        ax.scatter(Z[m, 0], Z[m, 1], s=7, alpha=0.45, color="#c3c2b7", label=f"otros {len(otros)} clusters")
    m = etq == -1
    ax.scatter(Z[m, 0], Z[m, 1], s=9, alpha=0.6, color=COLOR_RUIDO, marker="x", linewidths=0.7,
               label=f"ruido (n={m.sum()})")
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
    ax.set_title(f"DBSCAN (ε={eps:.3g}, MinPts={min_pts}) en PCA 2D"); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    leyenda(ax, fontsize=7)
    ax = axes[1, 2]
    dispersion_clases(ax, Z, d["y_train"], alpha=0.35, s=7)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
    ax.set_title("Referencia: etiqueta real (solo validación externa)"); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    leyenda(ax, fontsize=7)
    fig.suptitle(f"Aprendizaje no supervisado — {d['etiqueta']}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    guardar_figura(fig, f"nosup_caso{num}")


# %% [markdown]
# ## 18. Entrenamiento y evaluación de K-Means y DBSCAN en los 8 casos

# %%
resultados_ns, sensibilidad_db, asignaciones_3, modelos_ns = [], [], [], {}
for num, d in datos_casos.items():
    t0 = time.time()
    X = d["X_train"]
    print("=" * 100, f"\n{d['etiqueta']} — {X.shape}")

    # --- K-Means ---
    barrido = barrido_kmeans(X)
    k_opt = int(barrido.loc[barrido.silhouette.idxmax(), "K"])
    km = KMeans(n_clusters=k_opt, init="k-means++", n_init=10, random_state=SEMILLA).fit(X)
    sil_km = silhouette_score(X, km.labels_)
    tamanos = np.bincount(km.labels_)
    print(f"K-Means → K óptimo={k_opt} (codo de inercia en K={barrido.K[codo(barrido.inercia)]}) | "
          f"inercia={km.inertia_:.4g} | silhouette={sil_km:.4f} | tamaños={tamanos.tolist()}")
    display(barrido.round(4).set_index("K").T)

    # --- DBSCAN ---
    min_pts = 2 * X.shape[1]
    kdist = k_distancias(X, min_pts)
    i_codo = codo(kdist)
    eps_codo = float(kdist[i_codo])
    db, n_cl, ruido, sil_db = evaluar_dbscan(X, eps_codo, min_pts)
    sens = []
    for f in (0.3, 0.4, 0.5, 0.6, 0.75, 1.0, 1.25, 1.5):
        m_, n_, r_, s_ = evaluar_dbscan(X, f * eps_codo, min_pts)
        sens.append({"caso": num, "factor": f, "eps": f * eps_codo, "clusters": n_, "%ruido": r_, "silhouette": s_})
        sensibilidad_db.append(sens[-1])
    sens = pd.DataFrame(sens)
    eps_usado, criterio_eps = eps_codo, "codo"
    if n_cl < 2:
        validos = sens[(sens.clusters >= 2) & (sens["%ruido"] <= 50)]
        if len(validos):
            mejor = validos.loc[validos.silhouette.idxmax()]
            eps_usado, criterio_eps = float(mejor.eps), f"sensibilidad ({mejor.factor}× codo)"
            db, n_cl, ruido, sil_db = evaluar_dbscan(X, eps_usado, min_pts)
    print(f"DBSCAN → MinPts={min_pts} | ε codo={eps_codo:.4g} | ε usado={eps_usado:.4g} ({criterio_eps}) | "
          f"clusters={n_cl} | ruido={ruido:.2f} % | silhouette (sin ruido)={sil_db:.4f}")
    display(sens.drop(columns="caso").round(4).set_index("factor").T)

    # --- validación externa (etiqueta real, solo para evaluar) ---
    y_tr = d["y_train"].values
    ari_km, nmi_km = adjusted_rand_score(y_tr, km.labels_), normalized_mutual_info_score(y_tr, km.labels_)
    ari_db, nmi_db = adjusted_rand_score(y_tr, db.labels_), normalized_mutual_info_score(y_tr, db.labels_)
    print("Tabla de contingencia K-Means (filas = clase real, columnas = cluster):")
    display(pd.crosstab(pd.Categorical(y_tr, CLASES), km.labels_, rownames=["clase"], colnames=["cluster"]))

    # --- tres registros de prueba ---
    X3 = d["X_test"][IDX_PRUEBA_3]
    for j, idx in enumerate(IDX_PRUEBA_3):
        asignaciones_3.append({"caso": num, "registro_prueba": idx, "real": y_test[idx],
                               "cluster_kmeans": int(km.predict(X3)[j]),
                               "cluster_dbscan": int(asignar_dbscan(db, X, X3)[j])})

    resultados_ns.append({"caso": num, "ED": "SI" if d["escalado"] else "NO", "OUTLIERS": "SI" if d["outliers"] else "NO",
                          "BALANCEO": "SI" if d["balanceo"] else "NO", "K_optimo": k_opt, "km_inercia": km.inertia_,
                          "km_silhouette": sil_km, "km_tamanos": tamanos.tolist(), "km_ARI": ari_km, "km_NMI": nmi_km,
                          "db_minpts": min_pts, "db_eps_codo": eps_codo, "db_eps": eps_usado, "db_criterio_eps": criterio_eps,
                          "db_clusters": n_cl, "db_%ruido": ruido, "db_silhouette": sil_db, "db_ARI": ari_db, "db_NMI": nmi_db})
    modelos_ns[num] = {"kmeans": km, "dbscan": db}
    graficar_no_supervisado(d, barrido, k_opt, km, kdist, i_codo, eps_usado, min_pts, db, num)
    print(f"Tiempo del caso: {time.time() - t0:.1f} s")

resultados_ns = pd.DataFrame(resultados_ns).set_index("caso")
sensibilidad_db = pd.DataFrame(sensibilidad_db)
asignaciones_3 = pd.DataFrame(asignaciones_3)

# %% [markdown]
# ## 19. Tabla de la figura 3

# %%
figura3 = resultados_ns[["ED", "OUTLIERS", "BALANCEO", "km_inercia", "km_silhouette", "db_silhouette"]].rename(
    columns={"km_inercia": "K-MEANS INERTIA", "km_silhouette": "K-MEANS SILHOUETTE", "db_silhouette": "DBSCAN SILHOUETTE"})
display(figura3.round(4))
guardar_tabla(figura3.round(4), "figura3_no_supervisado")
detalle_ns = resultados_ns.drop(columns=["ED", "OUTLIERS", "BALANCEO"])
display(detalle_ns.round(4))
guardar_tabla(detalle_ns.round(4), "detalle_no_supervisado")
guardar_tabla(sensibilidad_db.round(4), "sensibilidad_dbscan")
display(asignaciones_3.set_index(["caso", "registro_prueba"]))
guardar_tabla(asignaciones_3, "asignaciones_3_registros_no_supervisado")

# %% [markdown]
# ## 20. Silhouette por caso — K-Means y DBSCAN

# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 4.8))
for ax, col, nombre, color in ((axes[0], "km_silhouette", "K-Means", COLOR_PAR[0]),
                               (axes[1], "db_silhouette", "DBSCAN", COLOR_PAR[1])):
    vals = resultados_ns[col]
    etiquetas = [f"C{n}\n{'ED' if r.ED == 'SI' else 'sin ED'}\nOut {r.OUTLIERS}\nBal {r.BALANCEO}"
                 for n, r in resultados_ns.iterrows()]
    barras = ax.bar(range(8), vals.fillna(0).values, color=color, width=0.6)
    for b, v in zip(barras, vals.values):
        ax.text(b.get_x() + b.get_width() / 2, max(b.get_height(), 0) + 0.01,
                "N/D" if np.isnan(v) else f"{v:.3f}", ha="center", va="bottom", fontsize=8.5, color=TINTA2)
    ax.set_xticks(range(8), etiquetas, fontsize=8)
    ax.set_ylim(min(0, np.nanmin(vals.values) - 0.05), 1.05)
    ax.axhline(0, color="#c3c2b7", linewidth=1)
    ax.set_title(f"Silhouette score por caso — {nombre}"); ax.set_ylabel("silhouette")
    ax.grid(axis="x", visible=False)
fig.tight_layout()
guardar_figura(fig, "nosup_silhouette_casos")

# %% [markdown]
# ## 21. Interpretación de los clusters en el dominio (perfil de variables originales)
# Se describen los clusters de K-Means de los casos 5 y 8 (con escalado) con las medias de las variables de
# tráfico en su **escala original**. El objetivo es ver si los grupos tienen sentido para la movilidad
# urbana: tráfico fluido frente a congestionado, hora punta, tipo de vía.

# %%
VARS_PERFIL = ["vehicle_count", "average_speed", "traffic_density", "queue_length", "average_wait_time",
               "heavy_vehicle_count", "hour", "rush_hour", "lanes", "parking_occupancy"]
perfiles = {}
for num in (5, 8):
    d, km = datos_casos[num], modelos_ns[num]["kmeans"]
    perfil = d["X_train_df"][VARS_PERFIL].assign(cluster=km.labels_).groupby("cluster").mean()
    perfil["n"] = np.bincount(km.labels_)
    perfil["road_type_mas_comun"] = d["X_train_df"].assign(cluster=km.labels_).groupby("cluster").road_type.agg(
        lambda s: s.value_counts().index[0])
    perfil["clase_real_mayoritaria"] = pd.Series(d["y_train"].values).groupby(km.labels_).agg(
        lambda s: f"{s.value_counts().index[0]} ({100 * s.value_counts().iloc[0] / len(s):.0f} %)")
    perfiles[num] = perfil
    print(f"Perfil de clusters K-Means — {d['etiqueta']}")
    display(perfil.round(2))
    guardar_tabla(perfil.round(3), f"perfil_clusters_caso{num}")

    db = modelos_ns[num]["dbscan"]
    perfil_db = d["X_train_df"][VARS_PERFIL].assign(cluster=db.labels_).groupby("cluster").mean()
    perfil_db["n"] = pd.Series(db.labels_).value_counts().sort_index().values
    perfil_db["clase_real_mayoritaria"] = pd.Series(d["y_train"].values).groupby(db.labels_).agg(
        lambda s: f"{s.value_counts().index[0]} ({100 * s.value_counts().iloc[0] / len(s):.0f} %)")
    print(f"Perfil de clusters DBSCAN (−1 = ruido) — {d['etiqueta']}")
    display(perfil_db.round(2))
    guardar_tabla(perfil_db.round(3), f"perfil_dbscan_caso{num}")

# %% [markdown]
# ## 22. Resumen final
# Las tablas de las figuras 1 y 3 y todas las gráficas quedaron guardadas en `resultados/`. El análisis de
# resultados y las conclusiones están en el documento de diseño (PDF).

# %%
print("Mejor F1 por algoritmo:")
display(resultados_sup.loc[resultados_sup.groupby("algoritmo").f1.idxmax(), ["algoritmo", "caso", "accuracy", "f1"]]
        .set_index("algoritmo").round(4))
print("Archivos generados:", len(os.listdir(DIR_FIG)), "figuras,", len(os.listdir(DIR_TAB)), "tablas")
