# ================================================
# IMPORTACIÓN DE LIBRERÍAS
# ================================================


import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


# ================================================
# CONFIGURACIÓN GENERAL
# ================================================

TARGET = "Pago_atiempo"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# ================================================
# FUNCIÓN DE CARGA DE DATOS
# ================================================

def load_data(file_path: str) -> pd.DataFrame:
    """
    Carga el dataset desde un archivo CSV.

    Parameters
    ----------
    file_path : str
        Ruta del archivo CSV.

    Returns
    -------
    pd.DataFrame
        Dataset cargado como DataFrame.
    """
    df = pd.read_csv(file_path)
    return df

# ================================================
# FUNCIÓN DE LIMPIEZA INICIAL
# ================================================

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza una limpieza inicial del dataset:
    - Crea una copia del DataFrame original.
    - Unifica representaciones alternativas de valores nulos.
    - Corrige categorías inválidas en tendencia_ingresos.
    - Convierte fecha_prestamo a formato datetime.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset original.

    Returns
    -------
    pd.DataFrame
        Dataset limpio.
    """
    df = df.copy()

    valores_nulos_equivalentes = [
        "", " ", "NA", "N/A", "na", "n/a",
        "null", "Null", "NULL",
        "None", "none",
        "-", "?",
        "Sin dato", "sin dato"
    ]

    df = df.replace(valores_nulos_equivalentes, np.nan)

    categorias_validas_tendencia = ["Decreciente", "Estable", "Creciente"]

    df["tendencia_ingresos"] = df["tendencia_ingresos"].where(
        df["tendencia_ingresos"].isin(categorias_validas_tendencia),
        np.nan
    )

    df["fecha_prestamo"] = pd.to_datetime(
        df["fecha_prestamo"],
        errors="coerce"
    )

    return df

# ================================================
# FUNCIÓN DE CREACIÓN DE VARIABLES DERIVADAS
# ================================================

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea variables derivadas a partir de las columnas originales.

    Las variables generadas buscan representar:
    - Capacidad de pago.
    - Nivel de endeudamiento.
    - Presencia de mora.
    - Exposición crediticia por sectores.
    - Información temporal del préstamo.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset limpio.

    Returns
    -------
    pd.DataFrame
        Dataset con nuevas variables derivadas.
    """
    df = df.copy()

    # Evita divisiones por cero reemplazando ceros por NaN temporalmente.
    salario_seguro = df["salario_cliente"].replace(0, np.nan)
    saldo_total_seguro = df["saldo_total"].replace(0, np.nan)

    df["ratio_cuota_salario"] = df["cuota_pactada"] / salario_seguro
    df["ratio_capital_salario"] = df["capital_prestado"] / salario_seguro
    df["ratio_mora_saldo_total"] = df["saldo_mora"] / saldo_total_seguro

    df["flag_tiene_mora"] = (df["saldo_mora"].fillna(0) > 0).astype(int)

    df["total_creditos_sectoriales"] = (
        df["creditos_sectorFinanciero"].fillna(0) +
        df["creditos_sectorCooperativo"].fillna(0) +
        df["creditos_sectorReal"].fillna(0)
    )

    df["anio_prestamo"] = df["fecha_prestamo"].dt.year
    df["mes_prestamo"] = df["fecha_prestamo"].dt.month

    return df

# ================================================
# FUNCIÓN DE DEFINICIÓN DE GRUPOS DE VARIABLES
# ================================================

def define_feature_groups(df: pd.DataFrame) -> dict:
    """
    Define los grupos de variables que serán tratados por el preprocesador.

    Se separan las variables en:
    - Numéricas.
    - Categóricas nominales.
    - Categóricas ordinales.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset con variables originales y derivadas.

    Returns
    -------
    dict
        Diccionario con listas de columnas por tipo.
    """

    numeric_features = [
        "capital_prestado",
        "plazo_meses",
        "edad_cliente",
        "salario_cliente",
        "total_otros_prestamos",
        "cuota_pactada",
        "cant_creditosvigentes",
        "huella_consulta",
        "creditos_sectorFinanciero",
        "creditos_sectorCooperativo",
        "creditos_sectorReal",
        "promedio_ingresos_datacredito",
        "ratio_cuota_salario",
        "ratio_capital_salario",
        "total_creditos_sectoriales",
        "anio_prestamo",
        "mes_prestamo"
    ]

    nominal_features = [
        "tipo_credito",
        "tipo_laboral"
    ]

    ordinal_features = [
        "tendencia_ingresos"
    ]

    feature_groups = {
        "numeric_features": numeric_features,
        "nominal_features": nominal_features,
        "ordinal_features": ordinal_features
    }

    return feature_groups

 # Variables excluidas por posible fuga de información o relación demasiado directa
    # con la variable objetivo:
    #
    # - saldo_mora
    # - saldo_total
    # - saldo_principal
    # - saldo_mora_codeudor
    # - ratio_mora_saldo_total
    # - flag_tiene_mora
    # - puntaje
    # - puntaje_datacredito

# ---------------------------------------- OBSERVACION A TENER EN CUENTA --------------------------------------------------
#   Las variables puntaje y puntaje_datacredito se excluyen del primer entrenamiento porque generaban métricas perfectas, lo que sugiere posible fuga de información
# o una relación demasiado directa con la variable objetivo.
# -------------------------------------------------------------------------------------------------------------------------------------------------------------

# ================================================
# FUNCIÓN DE CREACIÓN DEL PREPROCESADOR
# ================================================

def build_preprocessor(feature_groups: dict) -> ColumnTransformer:
    """
    Construye el preprocesador de variables utilizando ColumnTransformer.

    Tratamientos aplicados:
    - Variables numéricas: imputación con mediana.
    - Variables categóricas nominales: imputación con moda y OneHotEncoder.
    - Variable categórica ordinal: imputación con moda y OrdinalEncoder.

    Parameters
    ----------
    feature_groups : dict
        Diccionario con listas de columnas por tipo.

    Returns
    -------
    ColumnTransformer
        Preprocesador listo para integrarse en un pipeline de modelado.
    """

    # Se extraen los grupos de variables definidos previamente.
    # Cada grupo recibirá un tratamiento distinto dentro del ColumnTransformer.
    
    numeric_features = feature_groups["numeric_features"]
    nominal_features = feature_groups["nominal_features"]
    ordinal_features = feature_groups["ordinal_features"]

    # Pipeline para variables numéricas.
    # Se imputan valores faltantes usando la mediana, ya que es una medida robusta frente a valores extremos.
    
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    # Pipeline para variables categóricas nominales.
    # Primero se imputan nulos con la categoría más frecuente. Luego se aplica OneHotEncoder para convertir categorías en variables binarias.
    # handle_unknown="ignore" evita errores si aparecen categorías nuevas en test o producción.
    
    nominal_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Pipeline para la variable categórica ordinal.
    # tendencia_ingresos tiene un orden lógico: Decreciente < Estable < Creciente. 
    # Por eso se usa OrdinalEncoder en lugar de OneHotEncoder.
    
    ordinal_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            categories=[["Decreciente", "Estable", "Creciente"]],
            handle_unknown="use_encoded_value",
            unknown_value=-1
        ))
    ])

    # ColumnTransformer aplica cada pipeline solo a las columnas indicadas.
    # Esto permite procesar numéricas, nominales y ordinales de forma separada dentro de un único objeto reutilizable.
    # 
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("nominal", nominal_transformer, nominal_features),
            ("ordinal", ordinal_transformer, ordinal_features)
        ]
    )

    return preprocessor

# ================================================
# FUNCIÓN DE PREPARACIÓN DE DATASETS
# ================================================

def prepare_datasets(df: pd.DataFrame):
    """
    Prepara los conjuntos de entrenamiento y prueba para modelado.

    Pasos realizados:
    - Limpieza inicial del dataset.
    - Creación de variables derivadas.
    - Separación entre variables predictoras y variable objetivo.
    - Definición de grupos de variables.
    - Creación del preprocesador.
    - División en train y test.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset original.

    Returns
    -------
    tuple
        X_train, X_test, y_train, y_test, preprocessor, feature_groups
    """

    # Se aplica la limpieza inicial definida previamente.
    # Esto incluye unificación de nulos, corrección de tendencia_ingresos y conversión de fecha_prestamo.

    df_limpio = clean_data(df)

    # Se crean variables derivadas a partir de las columnas originales.
    # Estas nuevas variables buscan representar capacidad de pago, exposición crediticia e información temporal.

    df_limpio = create_features(df_limpio)

    # Se definen los grupos de variables que serán tratados por el preprocesador.
    # Esto separa numéricas, categóricas nominales y categóricas ordinales.

    feature_groups = define_feature_groups(df_limpio)

    # Se consolidan las columnas seleccionadas para modelado.
    # Esto evita que variables no deseadas o potencialmente problemáticas queden dentro del conjunto de variables predictoras.

    selected_features = (
        feature_groups["numeric_features"] +
        feature_groups["nominal_features"] +
        feature_groups["ordinal_features"]
    )

    # Se separa la variable objetivo del conjunto de variables predictoras.
    # X contiene únicamente las columnas seleccionadas para entrenamiento.
    # y contiene la variable que queremos predecir.

    X = df_limpio[selected_features]
    y = df_limpio[TARGET]

    # Se construye el preprocesador que luego será usado dentro del pipeline del modelo.

    preprocessor = build_preprocessor(feature_groups)

    # Se realiza la separación entre entrenamiento y prueba.
    # stratify=y mantiene la proporción original de clases de Pago_atiempo en ambos conjuntos,
    # algo importante porque la variable objetivo está desbalanceada.

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    return X_train, X_test, y_train, y_test, preprocessor, feature_groups

# ================================================
# PRUEBA DE TESTEO DEL SCRIPT
# ================================================

if __name__ == "__main__":
    # Se obtiene la ruta raíz del proyecto tomando como referencia
    # la ubicación de este archivo dentro de la carpeta src/.
    project_root = Path(__file__).resolve().parents[1]

    # Ruta del dataset ubicado en la raíz del repositorio.
    file_path = project_root / "Base_de_datos.csv"

    # Se carga el dataset desde el archivo CSV.
    df = load_data(file_path)

    # Se preparan los conjuntos de entrenamiento y prueba.
    # También se obtiene el preprocesador y los grupos de variables.
    X_train, X_test, y_train, y_test, preprocessor, feature_groups = prepare_datasets(df)

    # Se imprimen las dimensiones resultantes para validar
    # que la separación train/test se realizó correctamente.
    print("Preparación de datasets completada correctamente.")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test: {y_test.shape}")

    # Se muestran los grupos de variables que serán utilizados
    # por el preprocesador.
    print("\nGrupos de variables:")
    for group_name, columns in feature_groups.items():
        print(f"{group_name}: {len(columns)} variables")