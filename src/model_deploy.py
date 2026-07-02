# ================================================
# IMPORTACIÓN DE LIBRERÍAS
# ================================================

from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd
import uvicorn

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ft_engineering import clean_data, create_features


# ================================================
# CONFIGURACIÓN GENERAL
# ================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"

# ================================================
# ESQUEMA DE ENTRADA PARA PREDICCIÓN
# ================================================

class PredictionRequest(BaseModel):
    """
    Esquema de entrada para predicciones por lote.

    La API espera recibir una lista de registros.
    Cada registro debe venir como un diccionario con las variables
    necesarias para generar una predicción.
    """

    registros: List[Dict[str, Any]] = Field(
        ...,
        description="Lista de registros a predecir"
    )

# ================================================
# CARGA DEL MODELO ENTRENADO
# ================================================

def cargar_modelo():
    """
    Carga el pipeline entrenado guardado en formato pkl.

    El pipeline incluye:
    - Preprocesamiento.
    - Modelo entrenado.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo en la ruta: {MODEL_PATH}"
        )

    modelo = joblib.load(MODEL_PATH)

    return modelo


# Se carga el modelo al iniciar la API.
modelo = cargar_modelo()

# ================================================
# PREPARACIÓN DE DATOS PARA PREDICCIÓN
# ================================================

def preparar_datos_prediccion(df_entrada: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara los datos recibidos por la API antes de enviarlos al modelo.
    """

    df = df_entrada.copy()

    if "Pago_atiempo" in df.columns:
        df = df.drop(columns=["Pago_atiempo"])

    # Algunas variables no forman parte del modelo final porque fueron excluidas por posible fuga de información, pero create_features() las utiliza para
    # construir variables derivadas.
    columnas_auxiliares = {
        "saldo_total": 0,
        "saldo_mora": 0
    }

    for columna, valor_default in columnas_auxiliares.items():
        if columna not in df.columns:
            df[columna] = valor_default

    df = clean_data(df)

    df = create_features(df)

    if "Pago_atiempo" in df.columns:
        df = df.drop(columns=["Pago_atiempo"])

    return df

# ================================================
# CREACIÓN DE LA APLICACIÓN FASTAPI
# ================================================

app = FastAPI(
    title="API de Predicción de Riesgo Crediticio",
    description="API para disponibilizar el modelo entrenado del proyecto MLOps.",
    version="1.3.0"
)


# ================================================
# ENDPOINT RAÍZ
# ================================================

@app.get("/")
def root():
    """
    Endpoint raíz para validar que la API está funcionando.
    """

    return {
        "mensaje": "API de Riesgo Crediticio activa",
        "estado": "ok"
    }


# ================================================
# ENDPOINT DE SALUD
# ================================================

@app.get("/health")
def health():
    """
    Endpoint simple de salud para verificar disponibilidad del servicio
    y confirmar que el modelo fue cargado correctamente.
    """

    return {
        "status": "ok",
        "modelo_cargado": modelo is not None,
        "ruta_modelo": str(MODEL_PATH)
    }


# ================================================
# ENDPOINT DE PREDICCIÓN
# ================================================

@app.post("/predict")
def predict(request: PredictionRequest):
    """
    Recibe una lista de registros en formato JSON, prepara los datos
    y devuelve predicciones por lote.
    """

    try:
        # Se valida que la solicitud contenga registros.
        if not request.registros:
            raise HTTPException(
                status_code=400,
                detail="La solicitud no contiene registros para predecir."
            )

        # Se convierten los registros recibidos a un DataFrame.
        df_entrada = pd.DataFrame(request.registros)

        # Se preparan los datos usando la misma lógica base del entrenamiento.
        df_prediccion = preparar_datos_prediccion(df_entrada)

        # Se generan las predicciones.
        predicciones = modelo.predict(df_prediccion)

        # Si el modelo permite calcular probabilidades, se obtienen.
        probabilidades = None

        if hasattr(modelo, "predict_proba"):
            probabilidades = modelo.predict_proba(df_prediccion)[:, 1]

        # Se arma la respuesta final registro por registro.
        resultados = []

        for indice in range(len(df_prediccion)):

            prediccion = int(predicciones[indice])

            resultado = {
                "indice": indice,
                "prediccion": prediccion,
                "interpretacion": (
                    "Pagará a tiempo"
                    if prediccion == 1
                    else "No pagará a tiempo"
                )
            }

            if probabilidades is not None:
                resultado["probabilidad_pago_a_tiempo"] = round(
                    float(probabilidades[indice]),
                    4
                )

            resultados.append(resultado)

        return {
            "cantidad_registros": len(resultados),
            "predicciones": resultados
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Error al generar predicciones: {str(error)}"
        )

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )