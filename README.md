# 🚴 Predicción de Fatiga en Ciclistas

Aplicación web desarrollada con **Streamlit** que predice el nivel de fatiga de un ciclista en tiempo real usando tres modelos de Machine Learning: KNN, Regresión Lineal y Árbol de Decisión.

---

## 📁 Estructura del proyecto

```
proyecto/
├── train.py                  # Entrena y guarda los modelos
├── app.py                    # Aplicación Streamlit
├── Datos/
│   └── dataset_ciclismo_fatiga.csv
└── modelos/
    ├── metricas.pkl
    ├── modelo_fatiga_knn.pkl
    ├── modelo_fatiga_lr.pkl
    └── modelo_fatiga_dt.pkl
```

---

## ⚙️ Requisitos

- Python 3.8+
- pandas
- scikit-learn
- joblib
- streamlit

Instala las dependencias con:

```bash
pip install pandas scikit-learn joblib streamlit
```

---

## 🚀 Cómo ejecutar

### 1. Entrenar los modelos

Primero debes correr `train.py` para generar los archivos `.pkl`:

```bash
python train.py
```

Esto entrenará los tres modelos y guardará en `modelos/`:
- Los modelos entrenados (`.pkl`)
- Las métricas de evaluación (`metricas.pkl`)

### 2. Abrir la aplicación

```bash
streamlit run app.py
```

---

## 🤖 Modelos utilizados

| Modelo | Descripción |
|---|---|
| **KNN** | K-Nearest Neighbors con k=25 y estandarización |
| **Regresión Lineal** | Modelo lineal con estandarización |
| **Árbol de Decisión** | Árbol con profundidad máxima de 5 |

Todos los modelos usan `Pipeline` de scikit-learn para encapsular el preprocesamiento.

---

## 📊 Variables de entrada

| Variable | Descripción |
|---|---|
| `frecuencia_cardiaca` | Frecuencia cardíaca en bpm |
| `potencia` | Potencia en watts |
| `cadencia` | Cadencia en rpm |
| `tiempo` | Tiempo acumulado en minutos |
| `temperatura` | Temperatura ambiente en °C |
| `pendiente` | Pendiente del terreno en % |
| `velocidad` | Velocidad en km/h |

**Variable objetivo:** `fatiga` (valor entre 0 y 100)

---

## 📈 Métricas de evaluación

La app muestra al inicio las métricas de cada modelo:

- **MSE** — Error cuadrático medio
- **MAE** — Error absoluto medio
- **R²** — Coeficiente de determinación

---

## 🟢 Niveles de fatiga

| Rango | Nivel | Interpretación |
|---|---|---|
| 0 – 20 | Muy baja | Sin fatiga significativa |
| 21 – 40 | Baja | Esfuerzo leve |
| 41 – 60 | Media | Fatiga moderada |
| 61 – 80 | Alta | Fatiga evidente |
| 81 – 100 | Muy alta | Fatiga extrema / agotamiento |

---

## 📝 Notas

- El entrenamiento solo se necesita hacer **una vez** o cuando cambie el dataset.
- La app únicamente realiza **inferencia** — no modifica los modelos.
- El dataset debe estar en `Datos/dataset_ciclismo_fatiga.csv`.
