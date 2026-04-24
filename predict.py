import pandas as pd
import joblib
import streamlit as st
import os
import subprocess

st.set_page_config(
    page_title="Fatiga Ciclismo",
    page_icon="🚴",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #071a0e 0%, #0a1f12 50%, #071510 100%);
    color: #e8eaf0;
}

.hero-block {
    background: linear-gradient(90deg, #0f2d1a 0%, #0a2214 100%);
    border-left: 4px solid #22c55e;
    border-radius: 0 12px 12px 0;
    padding: 20px 28px;
    margin-bottom: 28px;
}
.hero-block h1 {
    font-family: 'Space Mono', monospace;
    font-size: 1.9rem;
    color: #ffffff;
    margin: 0 0 4px 0;
}
.hero-block p { color: #6b9e7a; font-size: 0.95rem; margin: 0; }

.metric-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(34,197,94,0.2);
    border-radius: 14px;
    padding: 20px 18px;
    text-align: center;
    height: 100%;
}
.metric-card h4 {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #22c55e;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin: 0 0 12px 0;
}
.metric-card .big-num { font-size: 1.5rem; font-weight: 700; color: #f0f4ff; }
.metric-stat { display:flex;justify-content:space-between;font-size:0.82rem;color:#6b9e7a;margin:4px 0; }
.metric-stat span.val { color:#c7d2f0;font-weight:600; }

.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #22c55e;
    margin-bottom: 16px;
}

.fatigue-result {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 14px;
    padding: 22px 20px;
    text-align: center;
}
.fatigue-result h3 {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: #6b9e7a;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin: 0 0 10px 0;
}
.fatigue-val { font-size:2.8rem;font-weight:700;line-height:1;margin:4px 0; }
.bar-wrap { background:rgba(255,255,255,0.08);border-radius:999px;height:8px;margin:14px 0 10px;overflow:hidden; }
.bar-fill { height:100%;border-radius:999px; }

.my-divider { border:none;border-top:1px solid rgba(255,255,255,0.08);margin:28px 0; }

div.stButton > button {
    background: linear-gradient(135deg, #16a34a, #15803d);
    color: white; border: none; border-radius: 10px;
    padding: 14px 32px; font-family: 'DM Sans', sans-serif;
    font-weight: 700; font-size: 1rem; width: 100%;
}
div.stButton > button:hover { opacity: 0.9; }

div[data-baseweb="input"] > div,
div[data-baseweb="base-input"] > input {
    background: rgba(255,255,255,0.05) !important;
    border-color: rgba(34,197,94,0.15) !important;
    color: #e8eaf0 !important;
    border-radius: 8px !important;
}
label { color: #c7d2f0 !important; font-size: 0.88rem !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; max-width: 860px; }
</style>
""", unsafe_allow_html=True)

# ── Entrenar si no existen los modelos ───────────────────────────────────────
if not os.path.exists("modelos/modelo_fatiga_knn.pkl"):
    with st.spinner("Entrenando modelos por primera vez..."):
        resultado = subprocess.run(["python", "train.py"], capture_output=True, text=True)
    if resultado.returncode != 0:
        st.error(f"Error al entrenar: {resultado.stderr}")
        st.stop()

# ── Carga de modelos ──────────────────────────────────────────────────────────
try:
    modelo_knn = joblib.load("modelos/modelo_fatiga_knn.pkl")
    modelo_lr  = joblib.load("modelos/modelo_fatiga_lr.pkl")
    modelo_dt  = joblib.load("modelos/modelo_fatiga_dt.pkl")
except Exception as e:
    st.error(f"No se pudo cargar un modelo: {e}")
    st.stop()

@st.cache_data
def cargar_metricas():
    return joblib.load("modelos/metricas.pkl")

try:
    metricas = cargar_metricas()
except:
    metricas = None

# ── Helpers ───────────────────────────────────────────────────────────────────
def nivel_fatiga(valor):
    if valor <= 20:   return "Muy baja",  "#22c55e", "Sin fatiga significativa"
    elif valor <= 40: return "Baja",      "#84cc16", "Esfuerzo leve"
    elif valor <= 60: return "Media",     "#f59e0b", "Fatiga moderada"
    elif valor <= 80: return "Alta",      "#f97316", "Fatiga evidente"
    else:             return "Muy alta",  "#ef4444", "Fatiga extrema / agotamiento"

def result_html(titulo, valor):
    nivel, color, desc = nivel_fatiga(valor)
    pct = int(min(max(valor, 0), 100))
    return f"""
    <div class="fatigue-result">
      <h3>{titulo}</h3>
      <div class="fatigue-val" style="color:{color}">{valor:.1f}</div>
      <div style="color:#6b9e7a;font-size:0.8rem;">/ 100</div>
      <div class="bar-wrap"><div class="bar-fill" style="width:{pct}%;background:{color}"></div></div>
      <div style="font-size:0.88rem;font-weight:600;color:{color}">{nivel}</div>
      <div style="color:#6b9e7a;font-size:0.82rem;margin-top:4px">{desc}</div>
    </div>"""

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-block">
  <h1>Prediccion de Fatiga en Ciclistas</h1>
  <p>Comparacion de modelos · KNN · Regresion Lineal · Arbol de Decision</p>
</div>
""", unsafe_allow_html=True)

# ── METRICAS DE MODELOS ───────────────────────────────────────────────────────
if metricas:
    st.markdown('<p class="section-label">Rendimiento de los modelos (test set)</p>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    modelos_info = [
        (c1, "KNN",               "knn", "K=25 · StandardScaler"),
        (c2, "Regresion Lineal",  "lr",  "LinearRegression · StandardScaler"),
        (c3, "Arbol de Decision", "dt",  "max_depth=5"),
    ]
    for col, nombre, key, detalle in modelos_info:
        m = metricas[key]
        with col:
            st.markdown(f"""
            <div class="metric-card">
              <h4>{nombre}</h4>
              <div class="big-num">{m['r2']*100:.1f}%</div>
              <div style="color:#6b9e7a;font-size:0.75rem;margin:6px 0 12px">R² — varianza explicada</div>
              <div class="metric-stat"><span>MSE</span><span class="val">{m['mse']:.2f}</span></div>
              <div class="metric-stat"><span>MAE</span><span class="val">{m['mae']:.2f}</span></div>
              <div style="color:#2d5c3a;font-size:0.72rem;margin-top:10px">{detalle}</div>
            </div>""", unsafe_allow_html=True)

st.markdown('<hr class="my-divider">', unsafe_allow_html=True)

# ── INPUTS (columna unica centrada) ──────────────────────────────────────────
st.markdown('<p class="section-label">Parametros del ciclista</p>', unsafe_allow_html=True)

_, form_col, _ = st.columns([1, 2, 1])
with form_col:
    bmp         = st.number_input("Frecuencia cardiaca (bpm)",  min_value=0.0, step=1.0,  format="%.0f")
    watts       = st.number_input("Potencia (watts)",           min_value=0.0, step=1.0,  format="%.0f")
    rpm         = st.number_input("Cadencia (rpm)",             min_value=0.0, step=1.0,  format="%.0f")
    tiempo      = st.number_input("Tiempo acumulado (min)",     min_value=0.0, step=0.1)
    temperatura = st.number_input("Temperatura (°C)",           min_value=-20.0, max_value=60.0, step=0.1)
    pendiente   = st.number_input("Pendiente (%)",              min_value=-30.0, max_value=30.0, step=0.1)
    velocidad   = st.number_input("Velocidad (km/h)",           min_value=0.0, step=0.1)

st.markdown('<hr class="my-divider">', unsafe_allow_html=True)

# ── PREDICCION ────────────────────────────────────────────────────────────────
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    predecir = st.button("Predecir Fatiga")

if predecir:
    entrada = [[bmp, watts, rpm, tiempo, temperatura, pendiente, velocidad]]
    fatiga_knn = modelo_knn.predict(entrada)[0]
    fatiga_lr  = modelo_lr.predict(entrada)[0]
    fatiga_dt  = modelo_dt.predict(entrada)[0]

    st.markdown('<p class="section-label" style="margin-top:24px">Resultados de prediccion</p>', unsafe_allow_html=True)
    r1, r2, r3 = st.columns(3)
    with r1: st.markdown(result_html("KNN", fatiga_knn), unsafe_allow_html=True)
    with r2: st.markdown(result_html("Regresion Lineal", fatiga_lr), unsafe_allow_html=True)
    with r3: st.markdown(result_html("Arbol de Decision", fatiga_dt), unsafe_allow_html=True)

    promedio = (fatiga_knn + fatiga_lr + fatiga_dt) / 3
    nivel_p, color_p, desc_p = nivel_fatiga(promedio)
    st.markdown(f"""
    <div style="margin-top:20px;background:rgba(34,197,94,0.06);border:1px solid rgba(34,197,94,0.25);
         border-radius:14px;padding:18px 24px;display:flex;align-items:center;gap:20px">
      <div style="font-size:2.2rem;font-weight:700;color:{color_p}">{promedio:.1f}</div>
      <div>
        <div style="font-family:'Space Mono',monospace;font-size:0.7rem;color:#22c55e;
             text-transform:uppercase;letter-spacing:1.5px">Promedio de los 3 modelos</div>
        <div style="font-size:1rem;font-weight:600;color:{color_p};margin-top:2px">{nivel_p} — {desc_p}</div>
      </div>
    </div>""", unsafe_allow_html=True)