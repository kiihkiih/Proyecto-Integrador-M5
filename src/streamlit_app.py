# ================================================
# IMPORTACIÓN DE LIBRERÍAS
# ================================================

import pandas as pd
import joblib
import streamlit as st
import plotly.express as px

from pathlib import Path

from model_monitoring import (
    cargar_datos_monitoreo,
    generar_reporte_monitoreo
)


# ================================================
# CONFIGURACIÓN GENERAL
# ================================================

# Se define la ruta raíz del proyecto.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Se define la ruta del modelo entrenado.
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"


# ================================================
# CONFIGURACIÓN DE STREAMLIT
# ================================================

st.set_page_config(
    page_title="Monitoreo del Modelo",
    layout="wide"
)

# ================================================
# CARGA DEL MODELO ENTRENADO
# ================================================

@st.cache_resource
def cargar_modelo():
    """
    Carga el pipeline entrenado guardado en formato pkl.

    El archivo contiene tanto el preprocesador como el modelo final,
    permitiendo reutilizarlo directamente en despliegue.
    """

    modelo = joblib.load(MODEL_PATH)
    return modelo


# ================================================
# CARGA DEL REPORTE DE MONITOREO
# ================================================

@st.cache_data
def cargar_reporte_monitoreo():
    """
    Genera el reporte de monitoreo utilizando la base histórica
    y la base nueva con data drift simulado.
    """

    X_ref, X_new, feature_groups = cargar_datos_monitoreo()

    reporte_monitoreo = generar_reporte_monitoreo(
        X_ref=X_ref,
        X_new=X_new,
        feature_groups=feature_groups
    )

    return X_ref, X_new, reporte_monitoreo

# ================================================
# INTERFAZ PRINCIPAL DE LA APLICACIÓN
# ================================================

st.title("Monitoreo del Modelo de Riesgo Crediticio")

st.markdown(
    """
    Esta aplicación permite monitorear cambios en la distribución de los datos
    utilizados por el modelo de predicción de pago a tiempo.

    Se compara una **base histórica de referencia**, utilizada durante el entrenamiento,
    contra una **base nueva con data drift simulado**.
    """
)

# Se cargan modelo y datos de monitoreo.
modelo = cargar_modelo()
X_ref, X_new, reporte_monitoreo = cargar_reporte_monitoreo()


# ================================================
# RESUMEN GENERAL
# ================================================

st.subheader("Resumen general del monitoreo")

cantidad_variables = reporte_monitoreo.shape[0]
cantidad_alertas = int(reporte_monitoreo["alerta_drift"].sum())
porcentaje_alertas = round((cantidad_alertas / cantidad_variables) * 100, 2)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Variables monitoreadas", cantidad_variables)

with col2:
    st.metric("Variables con drift", cantidad_alertas)

with col3:
    st.metric("% variables con drift", f"{porcentaje_alertas}%")

st.markdown(
    """
    Las alertas se generan a partir de métricas estadísticas como PSI,
    KS Test, Jensen-Shannon y Chi-cuadrado.
    """
)

# ================================================
# INDICADOR VISUAL DE RIESGO
# ================================================

st.subheader("Indicador general de riesgo por data drift")

if porcentaje_alertas >= 50:
    st.error(
        "Riesgo alto: más de la mitad de las variables monitoreadas presentan drift. "
        "Se recomienda revisar el modelo y considerar un proceso de reentrenamiento."
    )
elif porcentaje_alertas >= 20:
    st.warning(
        "Riesgo medio: algunas variables presentan drift. "
        "Se recomienda monitoreo frecuente y revisión de variables críticas."
    )
else:
    st.success(
        "Riesgo bajo: se detectan pocos cambios relevantes en la población monitoreada."
    )

# ================================================
# VISUALIZACIÓN DE MÉTRICAS
# ================================================

st.subheader("Visualización de métricas")

# Se crea una tabla resumen para comparar variables con y sin alerta de drift.
resumen_alertas = reporte_monitoreo["alerta_drift"].value_counts().reset_index()
resumen_alertas.columns = ["alerta_drift", "cantidad"]
resumen_alertas["estado"] = resumen_alertas["alerta_drift"].map({
    True: "Con drift",
    False: "Sin drift"
})

fig_alertas = px.bar(
    resumen_alertas,
    x="estado",
    y="cantidad",
    title="Cantidad de variables con y sin alerta de drift",
    text="cantidad"
)

st.plotly_chart(
    fig_alertas,
    use_container_width=True,
    key="grafico_alertas_drift"
)

# Se grafican los valores de PSI para variables numéricas.
reporte_numerico = reporte_monitoreo[
    reporte_monitoreo["tipo_variable"] == "numerica"
].copy()

if not reporte_numerico.empty:
    fig_psi = px.bar(
        reporte_numerico.sort_values("psi", ascending=False),
        x="variable",
        y="psi",
        title="PSI por variable numérica",
        text="psi"
    )

    fig_psi.add_hline(
        y=0.20,
        line_dash="dash",
        annotation_text="Umbral PSI significativo"
    )

    fig_psi.update_layout(
        xaxis_title="Variable",
        yaxis_title="PSI",
        xaxis_tickangle=-45
    )

    st.plotly_chart(
        fig_psi,
        use_container_width=True,
        key="grafico_psi_variables"
    )


# ================================================
# VARIABLES CON POSIBLE DATA DRIFT
# ================================================

st.subheader("Variables con posible data drift")

variables_con_drift = reporte_monitoreo[
    reporte_monitoreo["alerta_drift"] == True
]

if not variables_con_drift.empty:
    st.dataframe(
        variables_con_drift[[
            "variable",
            "tipo_variable",
            "nivel_drift",
            "motivo_alerta",
            "psi",
            "ks_p_value",
            "jensen_shannon",
            "chi2_p_value"
        ]].round(4),
        use_container_width=True
    )
else:
    st.success("No se detectaron variables con alerta de drift.")

# ================================================
# REPORTE COMPLETO DE MONITOREO
# ================================================

st.subheader("Reporte completo de monitoreo")

st.dataframe(
    reporte_monitoreo.round(4),
    use_container_width=True
)

# ================================================
# ANÁLISIS TEMPORAL
# ================================================

st.subheader("Análisis temporal")

if "anio_prestamo" in X_new.columns and "mes_prestamo" in X_new.columns:

    df_temporal = X_new.copy()

    df_temporal["periodo"] = (
        df_temporal["anio_prestamo"].astype("Int64").astype(str) +
        "-" +
        df_temporal["mes_prestamo"].astype("Int64").astype(str).str.zfill(2)
    )

    evolucion_temporal = (
        df_temporal
        .groupby("periodo")
        .size()
        .reset_index(name="cantidad_registros")
        .sort_values("periodo")
    )

    fig_temporal = px.line(
        evolucion_temporal,
        x="periodo",
        y="cantidad_registros",
        markers=True,
        title="Evolución de registros en la base actual"
    )

    fig_temporal.update_layout(
        xaxis_title="Periodo",
        yaxis_title="Cantidad de registros"
    )

    st.plotly_chart(
        fig_temporal,
        use_container_width=True,
        key="grafico_analisis_temporal"
    )  

    st.markdown(
        """
        Este gráfico permite observar la distribución temporal de los datos actuales.
        Cambios abruptos en la cantidad de registros por período pueden indicar modificaciones
        en la población monitoreada o en el proceso de captura de datos.
        """
    )

else:
    st.info("No se encontraron variables temporales suficientes para realizar análisis temporal.")

# ================================================
# INFORMACIÓN DEL MODELO CARGADO
# ================================================

st.subheader("Modelo cargado")

st.write("El modelo entrenado fue cargado correctamente desde:")

st.code(str(MODEL_PATH))

st.write(
    """
    El archivo `.pkl` contiene el pipeline completo, incluyendo el preprocesamiento
    y el modelo seleccionado durante la etapa de entrenamiento.
    """
)

# ================================================
# RECOMENDACIONES AUTOMÁTICAS
# ================================================

st.subheader("Recomendaciones")

if cantidad_alertas == 0:
    st.success(
        """
        No se detectaron señales relevantes de data drift. 
        Se recomienda mantener el monitoreo periódico del modelo.
        """
    )

elif porcentaje_alertas < 50:
    st.warning(
        """
        Se detectó data drift en algunas variables. 
        Se recomienda revisar las variables con alerta, analizar su impacto en el modelo 
        y aumentar la frecuencia de monitoreo.
        """
    )

else:
    st.error(
        """
        Se detectó data drift significativo en una proporción alta de variables. 
        Se recomienda revisar el desempeño del modelo, validar la calidad de los datos nuevos 
        y considerar un proceso de reentrenamiento o recalibración.
        """
    )

st.markdown(
    """
    **Acciones sugeridas:**

    - Revisar las variables con mayor PSI o Jensen-Shannon.
    - Validar si los cambios corresponden a comportamiento real del negocio o errores de carga.
    - Evaluar nuevamente el desempeño del modelo con datos recientes.
    - Considerar reentrenamiento si el drift persiste o afecta variables críticas.
    """
)
