# ============================================================
# ENTRENAMIENTO DEL MODELO
# Problema: clasificar tumores de mama como malignos o benignos
# Dataset: Breast Cancer Wisconsin (Diagnostic), 569 muestras,
# 30 características medidas en imágenes de biopsias (FNA).
# ============================================================

import json
import os

import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Carpetas donde se guardan el dataset y el modelo.
os.makedirs("data", exist_ok=True)
os.makedirs("model", exist_ok=True)

# 1. Obtener el dataset y guardarlo como CSV dentro del proyecto.
datos = load_breast_cancer(as_frame=True)
df = datos.frame  # 30 columnas de características + columna "target"
df.to_csv("data/breast_cancer.csv", index=False)
print("Dataset guardado en data/breast_cancer.csv:", df.shape)

# 2. Leer el CSV y separar características (X) y etiqueta (y).
#    target: 0 = maligno, 1 = benigno
df = pd.read_csv("data/breast_cancer.csv")
X = df.drop(columns=["target"])
y = df["target"]

# 3. Dividir en entrenamiento (80 %) y prueba (20 %).
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Pipeline: estandarizar las variables y aplicar regresión logística.
#    Al guardar el pipeline completo, la API escala los datos
#    nuevos exactamente igual que en el entrenamiento.
modelo = Pipeline([
    ("escalador", StandardScaler()),
    ("clasificador", LogisticRegression(max_iter=1000)),
])

# 5. Entrenar.
modelo.fit(X_train, y_train)

# 6. Evaluar con los datos de prueba.
predicciones = modelo.predict(X_test)
exactitud = accuracy_score(y_test, predicciones)
print(f"Exactitud en prueba: {exactitud:.4f}")
print(classification_report(
    y_test, predicciones, target_names=["maligno", "benigno"]
))

# 7. Guardar el modelo entrenado.
joblib.dump(modelo, "model/modelo_cancer.joblib")

# 8. Guardar información del modelo (nombres de variables, métricas).
info = {
    "dataset": "Breast Cancer Wisconsin (Diagnostic)",
    "modelo": "StandardScaler + LogisticRegression",
    "clases": {"0": "maligno", "1": "benigno"},
    "caracteristicas": list(X.columns),
    "exactitud_prueba": round(exactitud, 4),
}
with open("model/info_modelo.json", "w") as f:
    json.dump(info, f, indent=2)

print("Modelo guardado en model/modelo_cancer.joblib")
