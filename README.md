# 🚴 Monitor de Fatiga Ciclista (Machine Learning)

Esta aplicación web, construida con **Streamlit**, permite estimar el nivel de agotamiento de un ciclista en tiempo real. El sistema utiliza tres modelos de aprendizaje automático para contrastar resultados: KNN, Regresión Lineal y Árboles de Decisión.

---

## 📁 Estructura del Proyecto

```text
MODEL_PREDICT_SL1/
├── .streamlit/
│   └── config.toml           # Configuración de la interfaz Streamlit
├── Data/
│   └── dataset_ciclismo_fatiga.csv
├── modelos/
│   ├── metricas.pkl          # Resultados de validación
│   ├── modelo_fatiga_dt.pkl
│   ├── modelo_fatiga_knn.pkl
│   └── modelo_fatiga_lr.pkl
├── predict.py                # Lógica de predicción
├── README.md                 # Documentación del proyecto
├── requirements.txt          # Lista de dependencias
└── train.py                  # Script de entrenamiento y exportación
