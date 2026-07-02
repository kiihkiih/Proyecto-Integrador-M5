# ================================================
# IMPORTACIÓN DE LIBRERÍAS
# ================================================

import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from ft_engineering import (
    load_data,
    prepare_datasets,
    RANDOM_STATE
)

# ================================================
# CONFIGURACIÓN GENERAL
# ================================================

# Se define la ruta raíz del proyecto tomando como referencia la ubicación de este archivo dentro de la carpeta src/.
# 
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Ruta del dataset utilizado para entrenamiento y evaluación.
DATA_PATH = PROJECT_ROOT / "Base_de_datos.csv"

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "best_model.pkl"

# Se define el valor positivo de la variable objetivo. En este caso, se asume que 1 representa pago a tiempo.
POS_LABEL = 1

# ================================================
# FUNCIÓN DE RESUMEN DE CLASIFICACIÓN
# ================================================

def summarize_classification(model_name, y_true, y_pred, y_proba=None) -> dict:
    """
    Calcula métricas principales para evaluar un modelo de clasificación.

    En este problema existe desbalance de clases, por lo que se reportan
    métricas tanto para la clase mayoritaria como para la clase minoritaria.

    Parameters
    ----------
    model_name : str
        Nombre del modelo evaluado.

    y_true : array-like
        Valores reales de la variable objetivo.

    y_pred : array-like
        Predicciones generadas por el modelo.

    y_proba : array-like, optional
        Probabilidades predichas para la clase positiva.
        Se utiliza para calcular ROC-AUC.

    Returns
    -------
    dict
        Diccionario con métricas de evaluación.
    """

    # Se calculan métricas generales del modelo.
    # Accuracy indica el porcentaje total de aciertos, pero puede ser engañosa cuando existe desbalance de clases.
    # 
    accuracy = accuracy_score(y_true, y_pred)

    # Métricas para la clase 1.
    # En este dataset, la clase 1 representa clientes que pagaron a tiempo.
    precision_clase_1 = precision_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0
    )

    recall_clase_1 = recall_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0
    )

    f1_clase_1 = f1_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0
    )

    # Métricas para la clase 0.
    # Esta clase representa clientes que no pagaron a tiempo.
    # En un contexto financiero, esta clase es especialmente importante porque puede asociarse a mayor riesgo crediticio.
    # 
    precision_clase_0 = precision_score(
        y_true,
        y_pred,
        pos_label=0,
        zero_division=0
    )

    recall_clase_0 = recall_score(
        y_true,
        y_pred,
        pos_label=0,
        zero_division=0
    )

    f1_clase_0 = f1_score(
        y_true,
        y_pred,
        pos_label=0,
        zero_division=0
    )

    # F1 macro promedia el F1 de ambas clases sin ponderar por cantidad de registros.
    # Esto permite evaluar mejor el desempeño cuando la variable objetivo está desbalanceada.
    f1_macro = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    # Se arma el diccionario final de métricas.
    metrics = {
        "modelo": model_name,
        "accuracy": accuracy,
        "precision_clase_1": precision_clase_1,
        "recall_clase_1": recall_clase_1,
        "f1_clase_1": f1_clase_1,
        "precision_clase_0": precision_clase_0,
        "recall_clase_0": recall_clase_0,
        "f1_clase_0": f1_clase_0,
        "f1_macro": f1_macro
    }

    # ROC-AUC requiere probabilidades de la clase positiva.
    # Si el modelo las entrega, se calcula esta métrica.
    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
    else:
        metrics["roc_auc"] = np.nan

    return metrics

# Debido al fuerte desbalance de la variable objetivo, accuracy puede resultar engañosa.
    # Un modelo podría obtener una accuracy alta prediciendo casi siempre la clase mayoritaria.
    #
    # Por este motivo, se calculan métricas separadas para ambas clases:
    # - Clase 1: clientes que pagaron a tiempo.
    # - Clase 0: clientes que no pagaron a tiempo.
    #
    # En un contexto financiero, la clase 0 es especialmente relevante porque representa posibles casos de riesgo crediticio.
    # 

# ================================================
# FUNCIÓN DE CONSTRUCCIÓN DE MODELOS
# ================================================

def build_model(model, preprocessor) -> Pipeline:
    """
    Construye un pipeline de modelado combinando preprocesamiento y modelo.

    Parameters
    ----------
    model : estimator
        Modelo de clasificación de scikit-learn.

    preprocessor : ColumnTransformer
        Preprocesador generado en ft_engineering.py.

    Returns
    -------
    Pipeline
        Pipeline completo con preprocesamiento y modelo.
    """

    # Se construye un Pipeline de scikit-learn.
    # El primer paso transforma las variables usando el preprocessor.
    # El segundo paso entrena el modelo de clasificación.
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    return pipeline

# ================================================
# FUNCIÓN DE DEFINICIÓN DE MODELOS
# ================================================

def get_models() -> dict:
    """
    Define los modelos supervisados iniciales que serán evaluados.

    Returns
    -------
    dict
        Diccionario con nombre del modelo y estimador asociado.
    """

    # Se definen modelos supervisados de clasificación.
    # Logistic Regression funciona como modelo base interpretable.
    # Random Forest permite capturar relaciones no lineales.
    # Gradient Boosting permite construir un ensamble secuencial de árboles.
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            class_weight="balanced"
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=RANDOM_STATE
        )
    }

    return models

# ================================================
# FUNCIÓN DE ENTRENAMIENTO Y EVALUACIÓN
# ================================================

def train_and_evaluate_models(X_train, X_test, y_train, y_test, preprocessor) -> pd.DataFrame:
    """
    Entrena y evalúa distintos modelos de clasificación.

    Parameters
    ----------
    X_train : pd.DataFrame
        Variables predictoras de entrenamiento.

    X_test : pd.DataFrame
        Variables predictoras de prueba.

    y_train : pd.Series
        Variable objetivo de entrenamiento.

    y_test : pd.Series
        Variable objetivo de prueba.

    preprocessor : ColumnTransformer
        Preprocesador generado en ft_engineering.py.

    Returns
    -------
    pd.DataFrame
        Tabla comparativa con las métricas de cada modelo.
    """

    # Se obtiene el diccionario de modelos definidos previamente.
    models = get_models()

    # Lista donde se guardarán las métricas de cada modelo.
    results = []

    trained_models = {}

    # Se entrena y evalúa cada modelo definido.
    for model_name, model in models.items():

        print(f"\nEntrenando modelo: {model_name}")

        # Se construye el pipeline completo:
        # preprocesamiento + modelo.
        pipeline = build_model(model, preprocessor)

        # Se entrena el pipeline con los datos de entrenamiento.
        pipeline.fit(X_train, y_train)

        trained_models[model_name] = pipeline

        # Se generan predicciones sobre el conjunto de prueba.
        y_pred = pipeline.predict(X_test)

        # Si el modelo permite obtener probabilidades, se calcula la probabilidad de la clase positiva para usarla en ROC-AUC.
        
        if hasattr(pipeline.named_steps["model"], "predict_proba"):
            y_proba = pipeline.predict_proba(X_test)[:, 1]
        else:
            y_proba = None

        # Se calculan las métricas principales del modelo.
        
        metrics = summarize_classification(
            model_name=model_name,
            y_true=y_test,
            y_pred=y_pred,
            y_proba=y_proba
        )

        # Se guarda el resultado en la lista.
        
        results.append(metrics)

        # Se imprime la matriz de confusión.
        # Esto permite observar aciertos y errores por clase.
        
        print("\nMatriz de confusión:")
        print(confusion_matrix(y_test, y_pred))

        # Se imprime el reporte de clasificación.
        # Incluye precision, recall y f1-score por clase.
        
        print("\nReporte de clasificación:")
        print(classification_report(y_test, y_pred, zero_division=0))

    # Se convierten los resultados a DataFrame para comparar modelos.
    
    results_df = pd.DataFrame(results)

    # Se ordenan los modelos por F1 macro.
    # Esta métrica es más adecuada que accuracy en escenarios con desbalance, porque considera el desempeño de ambas clases.
    
    results_df = results_df.sort_values(
        by="f1_macro",
        ascending=False
)
    return results_df, trained_models

# ================================================
# FUNCIÓN PARA GUARDAR EL MEJOR MODELO
# ================================================

def save_best_model(results_df: pd.DataFrame, trained_models: dict) -> None:
    """
    Guarda en formato pkl el mejor modelo entrenado según F1 macro.

    Se guarda el pipeline completo, incluyendo:
    - Preprocesamiento.
    - Modelo entrenado.

    Esto permite reutilizar el modelo posteriormente en despliegue,
    sin tener que repetir manualmente las transformaciones de datos.
    """

    # Se crea la carpeta models si todavía no existe.
    MODELS_DIR.mkdir(exist_ok=True)

    # Como results_df ya está ordenado por F1 macro, el primer registro corresponde
    # al mejor modelo según la métrica seleccionada.
    best_model_name = results_df.iloc[0]["modelo"]

    # Se obtiene el pipeline entrenado correspondiente al mejor modelo.
    best_pipeline = trained_models[best_model_name]

    # Se guarda el pipeline completo en formato pkl.
    joblib.dump(best_pipeline, MODEL_PATH)

    print(f"\nMejor modelo guardado correctamente: {best_model_name}")
    print(f"Ruta del modelo: {MODEL_PATH}")

# ================================================
# EJECUCIÓN PRINCIPAL DEL SCRIPT
# ================================================

if __name__ == "__main__":
    
    # Se carga el dataset desde la ruta definida en la configuración general.
    
    df = load_data(DATA_PATH)

    # Se preparan los conjuntos de entrenamiento y prueba utilizando las funciones del archivo ft_engineering.py.
    # Esta etapa incluye limpieza, creación de variables derivadas, separación train/test y construcción del preprocesador.
    
    X_train, X_test, y_train, y_test, preprocessor, feature_groups = prepare_datasets(df)

    # Se imprimen las dimensiones de los conjuntos generados.
    # Esto permite validar que la división train/test se realizó correctamente.
    
    print("Datasets preparados correctamente.")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test: {y_test.shape}")

    # Se entrenan y evalúan los modelos definidos.
    
    results_df, trained_models = train_and_evaluate_models(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        preprocessor=preprocessor
    )

    # Se imprime la tabla comparativa final de métricas.
    
    print("\nComparación final de modelos:")
    print("Nota: los modelos se ordenan por F1 macro debido al desbalance de clases.")
    print(results_df.round(4))

    save_best_model(
        results_df=results_df,
        trained_models=trained_models
    )
   