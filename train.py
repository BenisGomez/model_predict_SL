import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

MODELO_PATH_KNN = "modelos/modelo_fatiga_knn.pkl"
MODELO_PATH_LR  = "modelos/modelo_fatiga_lr.pkl"
MODELO_PATH_DT  = "modelos/modelo_fatiga_dt.pkl"

print("Entrenando modelo por primera vez...\n")

data = pd.read_csv("Data/dataset_ciclismo_fatiga.csv")
X = data[["frecuencia_cardiaca", "potencia", "cadencia", "tiempo","temperatura", "pendiente", "velocidad"]]
y = data["fatiga"]

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=50
)

# Regresión lineal
modelo_lr = Pipeline([
    ("scaler", StandardScaler()),
    ("modelo", LinearRegression())
])
modelo_lr.fit(x_train, y_train)
y_pred_lr = modelo_lr.predict(x_test)

# Métricas del modelo de regresión lineal
mse_lr = mean_squared_error(y_test, y_pred_lr)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
r2_lr  = r2_score(y_test, y_pred_lr)

# KNN
modelo_knn = Pipeline([
    ("scaler", StandardScaler()),
    ("modelo", KNeighborsRegressor(n_neighbors=25))
])
modelo_knn.fit(x_train, y_train)
y_pred_knn = modelo_knn.predict(x_test)  # ídem

# Métricas del modelo de KNN
mse_knn = mean_squared_error(y_test, y_pred_knn)
mae_knn = mean_absolute_error(y_test, y_pred_knn)
r2_knn  = r2_score(y_test, y_pred_knn)

# Árbol de decisión
modelo_dt = Pipeline([
    ("modelo", DecisionTreeRegressor(max_depth=5))
])
modelo_dt.fit(x_train, y_train)
y_pred_dt = modelo_dt.predict(x_test)

# Métricas del modelo de árbol de decisión
mse_dt = mean_squared_error(y_test, y_pred_dt)
mae_dt = mean_absolute_error(y_test, y_pred_dt)
r2_dt  = r2_score(y_test, y_pred_dt)
    
metricas = {
    "knn": {"mse": mse_knn, "mae": mae_knn, "r2": r2_knn},
    "lr":  {"mse": mse_lr,  "mae": mae_lr,  "r2": r2_lr},
    "dt":  {"mse": mse_dt, "mae": mae_dt, "r2": r2_dt},
}
joblib.dump(metricas, "modelos/metricas.pkl")

joblib.dump(modelo_lr,  MODELO_PATH_LR)
joblib.dump(modelo_knn, MODELO_PATH_KNN)
joblib.dump(modelo_dt,  MODELO_PATH_DT)

print("\nModelos entrenados y guardados correctamente.")