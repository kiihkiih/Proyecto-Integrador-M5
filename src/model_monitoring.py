# ================================================
# IMPORTACIÓN DE LIBRERÍAS
# ================================================

import pandas as pd
import numpy as np

from pathlib import Path

from scipy.stats import ks_2samp, chi2_contingency
from scipy.spatial.distance import jensenshannon

from ft_engineering import (
    load_data,
    clean_data,
    create_features,
    define_feature_groups,
    TARGET
)


# ================================================
# CONFIGURACIÓN GENERAL
# ================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Se define la ruta del dataset histórico usado como referencia.
DATA_REFERENCIA_PATH = PROJECT_ROOT / "Base_de_datos.csv"

# Se define la ruta del dataset nuevo usado para monitoreo. Esta base representa una nueva población con posible data drift.
DATA_NUEVA_PATH = PROJECT_ROOT / "Base_de_datos_con_Data_Drift_Simulado.xlsx"

# Umbrales iniciales para detección de drift.
PSI_THRESHOLD = 0.20
KS_PVALUE_THRESHOLD = 0.05
JS_THRESHOLD = 0.10
CHI2_PVALUE_THRESHOLD = 0.05


# ================================================
# CARGA DE DATOS PARA MONITOREO
# ================================================

def cargar_datos_monitoreo():
    """
    Carga la base histórica y la base nueva para realizar monitoreo de data drift.

    En este avance se utilizan dos fuentes:
    - Base_de_datos.csv como población histórica o de referencia.
    - Base_de_datos_con_Data_Drift_Simulado.xlsx como población nueva o monitoreada.

    La base histórica representa los datos usados para entrenar el modelo.
    La base nueva representa datos recientes que podrían presentar cambios
    en su distribución.

    Returns
    -------
    tuple
        X_ref, X_new, feature_groups
    """

    # Se carga la base histórica desde CSV.
    df_referencia = load_data(DATA_REFERENCIA_PATH)

    # Se carga la base nueva desde Excel.
    df_nueva = pd.read_excel(DATA_NUEVA_PATH)

    # Se aplica la misma limpieza utilizada durante la etapa de ingeniería de características.
    # Esto asegura consistencia entre entrenamiento y monitoreo.
    df_referencia = clean_data(df_referencia)
    df_nueva = clean_data(df_nueva)

    # Se crean las mismas variables derivadas en ambas bases.
    df_referencia = create_features(df_referencia)
    df_nueva = create_features(df_nueva)

    # Se definen los grupos de variables usados en el entrenamiento.
    feature_groups = define_feature_groups(df_referencia)

    # Se consolidan las columnas que realmente entran al monitoreo.
    selected_features = (
        feature_groups["numeric_features"] +
        feature_groups["nominal_features"] +
        feature_groups["ordinal_features"]
    )

    # Se valida que todas las columnas necesarias existan en la nueva base.
    columnas_faltantes = [
        col for col in selected_features
        if col not in df_nueva.columns
    ]

    if columnas_faltantes:
        raise ValueError(
            f"La base nueva no contiene las siguientes columnas necesarias: {columnas_faltantes}"
        )

    # X_ref representa la población histórica o de referencia.
    # X_new representa la población nueva o monitoreada.
    X_ref = df_referencia[selected_features]
    X_new = df_nueva[selected_features]

    return X_ref, X_new, feature_groups

# ================================================
# CÁLCULO DE POPULATION STABILITY INDEX
# ================================================

def calcular_psi(serie_referencia, serie_nueva, cantidad_bins: int = 10) -> float:
    """
    Calcula el Population Stability Index entre una población de referencia
    y una población nueva para una variable numérica.

    PSI permite medir cuánto cambió la distribución de una variable entre
    dos muestras. Es una métrica común en monitoreo de modelos de riesgo.

    Parameters
    ----------
    serie_referencia : pd.Series
        Datos de la población histórica o de referencia.

    serie_nueva : pd.Series
        Datos de la población nueva o monitoreada.

    cantidad_bins : int
        Cantidad de intervalos utilizados para dividir la variable.

    Returns
    -------
    float
        Valor de PSI calculado.
    """

    # Se eliminan valores nulos para evitar errores en los cálculos.
    serie_referencia = serie_referencia.dropna()
    serie_nueva = serie_nueva.dropna()

    # Si alguna de las dos series queda vacía, no se puede calcular PSI.
    if serie_referencia.empty or serie_nueva.empty:
        return np.nan

    # Se calculan los puntos de corte usando cuantiles de la población de referencia.
    # Esto permite comparar la población nueva contra una escala definida históricamente.
    cuantiles = np.linspace(0, 1, cantidad_bins + 1)
    puntos_corte = serie_referencia.quantile(cuantiles).values

    # Se eliminan puntos duplicados, que pueden aparecer cuando la variable tiene
    # muchos valores repetidos.
    puntos_corte = np.unique(puntos_corte)

    # Si quedan menos de dos puntos de corte, la variable no tiene suficiente variabilidad.
    if len(puntos_corte) < 2:
        return np.nan

    # Se asigna cada valor a un intervalo según los puntos de corte definidos.
    bins_referencia = pd.cut(
        serie_referencia,
        bins=puntos_corte,
        include_lowest=True
    )

    bins_nueva = pd.cut(
        serie_nueva,
        bins=puntos_corte,
        include_lowest=True
    )

    # Se calcula la proporción de observaciones en cada intervalo.
    distribucion_referencia = bins_referencia.value_counts(normalize=True).sort_index()
    distribucion_nueva = bins_nueva.value_counts(normalize=True).sort_index()

    # Se alinean ambas distribuciones para asegurar que tengan los mismos intervalos.
    distribuciones = pd.DataFrame({
        "referencia": distribucion_referencia,
        "nueva": distribucion_nueva
    }).fillna(0)

    # Se reemplazan ceros por un valor pequeño para evitar divisiones por cero o logaritmos indefinidos.

    valor_minimo = 0.0001
    distribuciones["referencia"] = distribuciones["referencia"].replace(0, valor_minimo)
    distribuciones["nueva"] = distribuciones["nueva"].replace(0, valor_minimo)

    # Fórmula de PSI:
    # sum((nueva - referencia) * ln(nueva / referencia))
    
    valores_psi = (
        (distribuciones["nueva"] - distribuciones["referencia"]) *
        np.log(distribuciones["nueva"] / distribuciones["referencia"])
    )

    psi = valores_psi.sum()

    return psi

# ================================================
# CÁLCULO DE KS TEST
# ================================================

def calcular_ks_test(serie_referencia, serie_nueva) -> tuple:
    """
    Calcula el test de Kolmogorov-Smirnov entre una población de referencia
    y una población nueva para una variable numérica.

    El KS Test permite evaluar si dos muestras provienen de la misma distribución.
    Si el p-value es menor al umbral definido, se interpreta como posible drift.

    Parameters
    ----------
    serie_referencia : pd.Series
        Datos de la población histórica o de referencia.

    serie_nueva : pd.Series
        Datos de la población nueva o monitoreada.

    Returns
    -------
    tuple
        Estadístico KS y p-value.
    """

    # Se eliminan valores nulos para evitar errores en el cálculo estadístico.
    serie_referencia = serie_referencia.dropna()
    serie_nueva = serie_nueva.dropna()

    # Si alguna de las dos series queda vacía, no se puede calcular el test.
    if serie_referencia.empty or serie_nueva.empty:
        return np.nan, np.nan

    # Se aplica el test de Kolmogorov-Smirnov para comparar ambas distribuciones.
    estadistico_ks, p_value = ks_2samp(
        serie_referencia,
        serie_nueva
    )

    return estadistico_ks, p_value

# ================================================
# CÁLCULO DE JENSEN-SHANNON DIVERGENCE
# ================================================

def calcular_jensen_shannon(serie_referencia, serie_nueva, cantidad_bins: int = 10) -> float:
    """
    Calcula la divergencia de Jensen-Shannon entre una población de referencia
    y una población nueva.

    Esta métrica compara distribuciones de probabilidad.
    Valores cercanos a 0 indican distribuciones similares.
    Valores más altos indican mayor diferencia entre ambas poblaciones.

    Parameters
    ----------
    serie_referencia : pd.Series
        Datos de la población histórica o de referencia.

    serie_nueva : pd.Series
        Datos de la población nueva o monitoreada.

    cantidad_bins : int
        Cantidad de intervalos utilizados para dividir la variable.

    Returns
    -------
    float
        Valor de divergencia Jensen-Shannon.
    """

    # Se eliminan valores nulos para evitar problemas en el cálculo.
    serie_referencia = serie_referencia.dropna()
    serie_nueva = serie_nueva.dropna()

    # Si alguna de las dos series queda vacía, no se puede calcular la métrica.
    if serie_referencia.empty or serie_nueva.empty:
        return np.nan

    # Se calculan los puntos de corte usando cuantiles de la población de referencia.
    cuantiles = np.linspace(0, 1, cantidad_bins + 1)
    puntos_corte = serie_referencia.quantile(cuantiles).values

    # Se eliminan puntos duplicados por seguridad.
    puntos_corte = np.unique(puntos_corte)

    # Si no hay suficientes puntos de corte, la variable no tiene variabilidad suficiente.
    if len(puntos_corte) < 2:
        return np.nan

    # Se agrupan ambas series en los mismos intervalos.
    bins_referencia = pd.cut(
        serie_referencia,
        bins=puntos_corte,
        include_lowest=True
    )

    bins_nueva = pd.cut(
        serie_nueva,
        bins=puntos_corte,
        include_lowest=True
    )

    # Se calculan distribuciones de probabilidad.
    distribucion_referencia = bins_referencia.value_counts(normalize=True).sort_index()
    distribucion_nueva = bins_nueva.value_counts(normalize=True).sort_index()

    # Se alinean ambas distribuciones.
    distribuciones = pd.DataFrame({
        "referencia": distribucion_referencia,
        "nueva": distribucion_nueva
    }).fillna(0)

    # Se reemplazan ceros por un valor mínimo para evitar problemas numéricos.
    valor_minimo = 0.0001
    distribucion_ref = distribuciones["referencia"].replace(0, valor_minimo)
    distribucion_new = distribuciones["nueva"].replace(0, valor_minimo)

    # Se normalizan nuevamente para asegurar que ambas distribuciones sumen 1.
    distribucion_ref = distribucion_ref / distribucion_ref.sum()
    distribucion_new = distribucion_new / distribucion_new.sum()

    # scipy devuelve la distancia Jensen-Shannon.
    # Para este ejercicio la usamos como indicador de diferencia entre distribuciones.
    js = jensenshannon(
        distribucion_ref,
        distribucion_new
    )

    return js

# ================================================
# CÁLCULO DE CHI-CUADRADO PARA VARIABLES CATEGÓRICAS
# ================================================

def calcular_chi_cuadrado(serie_referencia, serie_nueva) -> tuple:
    """
    Calcula el test de Chi-cuadrado entre una población de referencia
    y una población nueva para una variable categórica.

    Este test permite evaluar si la distribución de categorías cambió
    de forma significativa entre ambas poblaciones.

    Parameters
    ----------
    serie_referencia : pd.Series
        Datos categóricos de la población histórica o de referencia.

    serie_nueva : pd.Series
        Datos categóricos de la población nueva o monitoreada.

    Returns
    -------
    tuple
        Estadístico Chi-cuadrado y p-value.
    """

    # Se reemplazan valores nulos por una categoría explícita.
    # Esto permite incluir los nulos como parte de la distribución monitoreada.
    serie_referencia = serie_referencia.astype("object").fillna("Valor nulo")
    serie_nueva = serie_nueva.astype("object").fillna("Valor nulo")

    # Se calculan las frecuencias absolutas de cada categoría.
    conteo_referencia = serie_referencia.value_counts()
    conteo_nueva = serie_nueva.value_counts()

    # Se arma un DataFrame con ambas distribuciones.
    # fillna(0) asegura que si una categoría aparece en una población
    # pero no en la otra, se complete con cero.
    tabla_frecuencias = pd.DataFrame({
        "referencia": conteo_referencia,
        "nueva": conteo_nueva
    }).fillna(0)

    # Si hay menos de dos categorías, el test no es informativo.
    if tabla_frecuencias.shape[0] < 2:
        return np.nan, np.nan

    # El test Chi-cuadrado se aplica sobre la tabla de frecuencias.
    estadistico_chi2, p_value, grados_libertad, frecuencias_esperadas = chi2_contingency(
        tabla_frecuencias.T
    )

    return estadistico_chi2, p_value

# ================================================
# CLASIFICACIÓN DEL NIVEL DE DRIFT
# ================================================

def clasificar_drift(psi=np.nan, p_value_ks=np.nan, js=np.nan, p_value_chi2=np.nan) -> tuple:
    """
    Clasifica el nivel de drift detectado y registra qué métrica generó la alerta.

    Parameters
    ----------
    psi : float
        Valor de Population Stability Index.

    p_value_ks : float
        P-value del KS Test.

    js : float
        Valor de Jensen-Shannon divergence.

    p_value_chi2 : float
        P-value del test Chi-cuadrado.

    Returns
    -------
    tuple
        nivel_drift, motivo_alerta
    """

    motivos_alerta = []

    # PSI se interpreta por niveles.
    # Valores altos indican cambios importantes en la distribución.
    if pd.notna(psi):
        if psi >= PSI_THRESHOLD:
            motivos_alerta.append("PSI")
        elif psi >= 0.10:
            motivos_alerta.append("PSI leve")

    # KS Test detecta diferencias estadísticamente significativas entre distribuciones numéricas.
    # 
    if pd.notna(p_value_ks) and p_value_ks < KS_PVALUE_THRESHOLD:
        motivos_alerta.append("KS Test")

    # Jensen-Shannon detecta diferencias entre distribuciones de probabilidad.
    if pd.notna(js) and js >= JS_THRESHOLD:
        motivos_alerta.append("Jensen-Shannon")

    # Chi-cuadrado detecta cambios en la distribución de variables categóricas.
    if pd.notna(p_value_chi2) and p_value_chi2 < CHI2_PVALUE_THRESHOLD:
        motivos_alerta.append("Chi-cuadrado")

    # Se define el nivel de drift.
    # Si alguna métrica fuerte dispara alerta, se considera significativo.
    if any(motivo in ["PSI", "KS Test", "Jensen-Shannon", "Chi-cuadrado"] for motivo in motivos_alerta):
        nivel_drift = "Drift significativo"
    elif "PSI leve" in motivos_alerta:
        nivel_drift = "Drift leve"
    else:
        nivel_drift = "Sin drift"

    # Si no hubo motivos de alerta, se deja explícito.
    if not motivos_alerta:
        motivo_alerta = "Sin alerta"
    else:
        motivo_alerta = ", ".join(motivos_alerta)

    return nivel_drift, motivo_alerta

# ================================================
# GENERACIÓN DEL REPORTE DE MONITOREO
# ================================================

def generar_reporte_monitoreo(X_ref, X_new, feature_groups) -> pd.DataFrame:
    """
    Genera una tabla resumen con métricas de data drift para las variables monitoreadas.

    Para variables numéricas se calculan:
    - PSI.
    - KS Test.
    - Jensen-Shannon.

    Para variables categóricas se calculan:
    - Chi-cuadrado.
    - Jensen-Shannon aplicado sobre distribuciones categóricas.

    Parameters
    ----------
    X_ref : pd.DataFrame
        Variables de la población histórica o de referencia.

    X_new : pd.DataFrame
        Variables de la población nueva o monitoreada.

    feature_groups : dict
        Diccionario con los grupos de variables definidos en ft_engineering.py.

    Returns
    -------
    pd.DataFrame
        Tabla resumen con las métricas de monitoreo y alerta de drift.
    """

    # Lista donde se almacenará el resultado de cada variable analizada.
    resultados_monitoreo = []

    # Se recuperan las variables numéricas y categóricas desde feature_groups.
    variables_numericas = feature_groups["numeric_features"]
    variables_categoricas = (
        feature_groups["nominal_features"] +
        feature_groups["ordinal_features"]
    )

    # ------------------------------------------------
    # Monitoreo de variables numéricas
    # ------------------------------------------------

    for variable in variables_numericas:

        # Se calcula PSI.
        psi = calcular_psi(
            serie_referencia=X_ref[variable],
            serie_nueva=X_new[variable]
        )

        # Se calcula KS Test.
        estadistico_ks, p_value_ks = calcular_ks_test(
            serie_referencia=X_ref[variable],
            serie_nueva=X_new[variable]
        )

        # Se calcula Jensen-Shannon.
        js = calcular_jensen_shannon(
            serie_referencia=X_ref[variable],
            serie_nueva=X_new[variable]
        )

        # Se define si existe alerta de drift según los umbrales definidos.

        nivel_drift, motivo_alerta = clasificar_drift(
            psi=psi,
            p_value_ks=p_value_ks,
            js=js,
            p_value_chi2=np.nan
        )

        alerta_drift = nivel_drift != "Sin drift"

        resultados_monitoreo.append({
            "variable": variable,
            "tipo_variable": "numerica",
            "psi": psi,
            "ks_statistic": estadistico_ks,
            "ks_p_value": p_value_ks,
            "jensen_shannon": js,
            "chi2_statistic": np.nan,
            "chi2_p_value": np.nan,
            "nivel_drift": nivel_drift,
            "motivo_alerta": motivo_alerta,
            "alerta_drift": alerta_drift
        })

    # ------------------------------------------------
    # Monitoreo de variables categóricas
    # ------------------------------------------------

    for variable in variables_categoricas:

        # Se calcula Chi-cuadrado.
        estadistico_chi2, p_value_chi2 = calcular_chi_cuadrado(
            serie_referencia=X_ref[variable],
            serie_nueva=X_new[variable]
        )

        # Para variables categóricas, en esta versión se utiliza principalmente
        # Chi-cuadrado para detectar cambios en la distribución de categorías.
        js = np.nan

        # Se clasifica el nivel de drift y el motivo de alerta.
        nivel_drift, motivo_alerta = clasificar_drift(
            psi=np.nan,
            p_value_ks=np.nan,
            js=js,
            p_value_chi2=p_value_chi2
        )

        # Se define la alerta general a partir del nivel de drift.
        alerta_drift = nivel_drift != "Sin drift"

        resultados_monitoreo.append({
            "variable": variable,
            "tipo_variable": "categorica",
            "psi": np.nan,
            "ks_statistic": np.nan,
            "ks_p_value": np.nan,
            "jensen_shannon": js,
            "chi2_statistic": estadistico_chi2,
            "chi2_p_value": p_value_chi2,
            "nivel_drift": nivel_drift,
            "motivo_alerta": motivo_alerta,
            "alerta_drift": alerta_drift
        })

    # Se convierte la lista de resultados en DataFrame.
    reporte_monitoreo = pd.DataFrame(resultados_monitoreo)

    reporte_monitoreo = reporte_monitoreo.sort_values(
        by=["alerta_drift", "tipo_variable", "variable"],
        ascending=[False, True, True]
        ).reset_index(drop=True)

    return reporte_monitoreo

# ================================================
# EJECUCIÓN PRINCIPAL DEL SCRIPT
# ================================================

if __name__ == "__main__":

    # Se cargan los datos preparados para monitoreo.
    X_ref, X_new, feature_groups = cargar_datos_monitoreo()

    # Se imprimen las dimensiones para validar que la carga funcione correctamente.
    print("Datos cargados correctamente para monitoreo.")
    print(f"X_ref: {X_ref.shape}")
    print(f"X_new: {X_new.shape}")

    # Se muestran los grupos de variables disponibles.
    print("\nGrupos de variables disponibles:")
    for group_name, columns in feature_groups.items():
        print(f"{group_name}: {len(columns)} variables")

    # ================================================
    # GENERACIÓN DEL REPORTE DE MONITOREO
    # ================================================

    reporte_monitoreo = generar_reporte_monitoreo(
        X_ref=X_ref,
        X_new=X_new,
        feature_groups=feature_groups
    )

    print("\nReporte de monitoreo generado correctamente:")
    print(reporte_monitoreo.round(4).to_string(index=False))

    print("\nCantidad de variables con alerta de drift:")
    print(reporte_monitoreo["alerta_drift"].sum())

    # Si existen variables con alerta, se muestran aparte para facilitar la lectura.
    variables_con_drift = reporte_monitoreo[
        reporte_monitoreo["alerta_drift"] == True
    ]

    if not variables_con_drift.empty:
        print("\nVariables con posible data drift:")
        print(
            variables_con_drift[[
                "variable",
                "tipo_variable",
                "nivel_drift",
                "motivo_alerta",
                "psi",
                "ks_p_value",
                "jensen_shannon",
                "chi2_p_value"
            ]].round(4).to_string(index=False)
        )
    else:
        print("\nNo se detectaron variables con alerta de drift.")