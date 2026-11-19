# Trabajo Práctico 2 — Aprendizaje de Máquinas (Grupo 01 · Equipo 07)

Introducción a la Inteligencia Artificial (3010476) — Universidad Nacional de Colombia, Sede Medellín.

Clasificación supervisada (Árboles de decisión, KNN, SVM y Redes Neuronales) y agrupamiento no supervisado
(K-Means y DBSCAN) sobre el dataset **CityFlow — Smart Urban Mobility and Traffic IoT** (Kaggle), con la
variable objetivo `congestion_level` (4 clases).

| Parámetro | Valor |
|---|---|
| Dígitos de la cédula usados | últimos dígitos ≠ 0: **1** y **9** |
| Registros del subconjunto | 5000 + 9 × 150 = **6350** |
| Semilla (`random_state`) | **19** |

## Estructura

```
Practica2_Grupo01_Equipo07.ipynb   Notebook entregable (Google Colab), ejecutado con todas sus salidas
src/practica2.py                   Fuente del notebook en formato jupytext "percent" (se versiona en git)
resultados/figuras/                Todas las gráficas generadas por el notebook
resultados/tablas/                 Tablas (figura 1, figura 3, métricas por clase, perfiles, etc.) en CSV
informe/                           Documento de diseño (HTML fuente y PDF) y script que lo genera
entrega/                           ZIP final para Google Classroom
Practica2-2026-2.pdf               Enunciado
```

## Cómo ejecutar

**Google Colab:** subir `Practica2_Grupo01_Equipo07.ipynb` y ejecutar todo (*Entorno de ejecución → Ejecutar todas*).
La primera celda instala `kagglehub` e `imbalanced-learn`, y el dataset se descarga automáticamente.

**Local (Windows):**

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install kagglehub pandas numpy matplotlib seaborn scikit-learn imbalanced-learn tensorflow jupyter jupytext
.\.venv\Scripts\jupytext.exe --to ipynb --output Practica2_Grupo01_Equipo07.ipynb src\practica2.py
.\.venv\Scripts\jupyter.exe nbconvert --to notebook --execute --inplace Practica2_Grupo01_Equipo07.ipynb
.\.venv\Scripts\python.exe informe\generar_informe.py      # regenera el documento de diseño
```

Si se edita el código, se modifica `src/practica2.py` y se regenera el `.ipynb` con jupytext.
