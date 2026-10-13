# ============================================================
# API DE PREDICCIÓN CON FASTAPI
# Carga el modelo entrenado y recibe las 30 mediciones de una
# biopsia para predecir si el tumor es maligno o benigno.
# ============================================================

import json
import logging

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Configurar los mensajes que aparecerán en "docker logs".
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("api-cancer")

# Cargar el modelo y su información una sola vez, al iniciar.
modelo = joblib.load("model/modelo_cancer.joblib")
with open("model/info_modelo.json") as f:
    info = json.load(f)
CARACTERISTICAS = info["caracteristicas"]
logger.info("Modelo cargado. Exactitud en prueba: %s", info["exactitud_prueba"])

app = FastAPI(
    title="Clasificador de tumores de mama",
    description="Predice si un tumor es maligno o benigno a partir de 30 "
                "mediciones de una biopsia (Breast Cancer Wisconsin).",
    version="1.0",
)

# Ejemplo real del dataset (muestra 0, diagnóstico maligno).
EJEMPLO_MALIGNO = [
    17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
    1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
    25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189,
]


class Muestra(BaseModel):
    """Las 30 mediciones de la biopsia, en el orden de GET /info."""
    caracteristicas: list[float] = Field(
        ..., min_length=30, max_length=30,
        examples=[EJEMPLO_MALIGNO],
    )


@app.get("/")
def inicio():
    return {
        "mensaje": "API de clasificación de tumores de mama",
        "endpoints": {
            "/health": "GET - estado de la API",
            "/info": "GET - información del modelo y orden de las variables",
            "/predict": "POST - realizar una predicción",
            "/docs": "GET - documentación interactiva",
        },
    }


@app.get("/health")
def salud():
    return {"status": "ok", "modelo_cargado": modelo is not None}


@app.get("/info")
def informacion():
    return info


@app.post("/predict")
def predecir(muestra: Muestra):
    try:
        # Convertir la lista en una tabla con los nombres de columna originales.
        entrada = pd.DataFrame([muestra.caracteristicas], columns=CARACTERISTICAS)
        clase = int(modelo.predict(entrada)[0])
        probabilidades = modelo.predict_proba(entrada)[0]
        resultado = {
            "prediccion": info["clases"][str(clase)],
            "clase": clase,
            "probabilidad_maligno": round(float(probabilidades[0]), 4),
            "probabilidad_benigno": round(float(probabilidades[1]), 4),
        }
        logger.info("Predicción realizada: %s", resultado)
        return resultado
    except Exception as e:
        logger.error("Error en la predicción: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
