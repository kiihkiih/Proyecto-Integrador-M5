# Proyecto Integrador M5 - Pipeline MLOps para Riesgo Crediticio

## Descripción general

Este proyecto desarrolla un pipeline completo de Machine Learning orientado a la predicción de riesgo crediticio. El objetivo principal es anticipar si un cliente realizará el pago de su crédito a tiempo, utilizando información histórica relacionada con préstamos, ingresos, comportamiento financiero y características del cliente.

A diferencia de un proyecto tradicional de Machine Learning centrado solamente en entrenar un modelo, este trabajo busca cubrir varias etapas del ciclo de vida de un modelo en producción:

- carga y comprensión de datos
- análisis exploratorio
- ingeniería de características
- entrenamiento y evaluación de modelos
- persistencia del mejor modelo
- monitoreo de Data Drift
- visualización mediante Streamlit
- disponibilización del modelo mediante una API
- contenerización con Docker

De esta forma, el proyecto simula un flujo MLOps completo, donde el modelo no solo se entrena, sino que también puede ser monitoreado, utilizado desde una aplicación y desplegado en un entorno reproducible.

## Objetivo del negocio

El problema de negocio consiste en apoyar la toma de decisiones crediticias mediante un modelo capaz de estimar si un cliente pagará a tiempo o no.

Una predicción de este tipo puede ayudar a una entidad financiera a:

- identificar clientes con mayor riesgo de incumplimiento
- mejorar el análisis previo a la aprobación de créditos
- priorizar acciones de seguimiento
- reducir pérdidas asociadas a mora o impago
- tomar decisiones basadas en datos y no solamente en reglas manuales

El modelo trabaja sobre la variable objetivo `Pago_atiempo`, donde:

- `1` representa que el cliente pagó a tiempo
- `0` representa que el cliente no pagó a tiempo

Durante el desarrollo se detectó que la variable objetivo estaba fuertemente desbalanceada, por lo que se evitó evaluar los modelos únicamente con accuracy. En su lugar, se utilizaron métricas más informativas para problemas de clasificación desbalanceada, como precision, recall, F1-score, F1 macro y ROC-AUC.

## Estructura del proyecto

```text
Proyecto-Integrador-M5/
│
├── models/
│   └── best_model.pkl
│
├── src/
│   ├── Cargar_datos.ipynb
│   ├── comprension_eda.ipynb
│   ├── ft_engineering.py
│   ├── model_training_evaluation.py
│   ├── model_monitoring.py
│   ├── model_deploy.py
│   └── streamlit_app.py
│
├── Base_de_datos.csv
├── Base_de_datos_con_Data_Drift_Simulado.xlsx
│
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── .gitignore
└── README.md
```

### Descripción de los componentes principales

- `src/Cargar_datos.ipynb`: notebook destinado a la carga inicial del dataset y primera inspección de la información.
- `src/comprension_eda.ipynb`: notebook de comprensión del negocio, análisis exploratorio y revisión de variables.
- `src/ft_engineering.py`: script encargado de la limpieza, transformación e ingeniería de características.
- `src/model_training_evaluation.py`: script de entrenamiento, comparación de modelos y guardado del mejor pipeline.
- `src/model_monitoring.py`: script de monitoreo y detección de Data Drift.
- `src/streamlit_app.py`: aplicación visual para consultar resultados del monitoreo.
- `src/model_deploy.py`: API construida con FastAPI para disponibilizar el modelo entrenado.
- `models/best_model.pkl`: pipeline completo del mejor modelo entrenado, serializado con Joblib.
- `Dockerfile`: archivo utilizado para construir una imagen Docker reproducible de la API.

## Carga de datos y análisis exploratorio

En el primer avance se trabajó sobre la carga, comprensión y exploración inicial del dataset.

El objetivo de esta etapa fue entender la estructura de la información antes de entrenar cualquier modelo. Para esto se revisaron:

- dimensiones del dataset
- tipos de datos
- valores faltantes
- variables numéricas y categóricas
- distribución de la variable objetivo
- posibles inconsistencias
- relaciones entre variables
- presencia de outliers
- correlaciones relevantes

Durante esta etapa se identificaron columnas con valores nulos, entre ellas:

- `tendencia_ingresos`
- `promedio_ingresos_datacredito`
- `saldo_mora_codeudor`
- `saldo_principal`
- `saldo_mora`
- `saldo_total`
- `puntaje_datacredito`

También se detectó que la columna `tendencia_ingresos` contenía valores inválidos mezclados con sus categorías correctas. Por ese motivo, se decidió conservar únicamente las categorías válidas:

- `Creciente`
- `Estable`
- `Decreciente`

Los valores que no coincidían con esas categorías fueron tratados como valores faltantes para ser imputados posteriormente dentro del pipeline.

Otro hallazgo importante fue el fuerte desbalance de la variable objetivo `Pago_atiempo`. La gran mayoría de los registros correspondían a clientes que sí pagaron a tiempo. Esto marcó una decisión importante para las siguientes etapas: no evaluar los modelos solamente con accuracy, ya que una accuracy alta podía ocultar un mal desempeño sobre la clase minoritaria.

##  Ingeniería de características

En el segundo avance se construyó un módulo reutilizable de ingeniería de características en el archivo `src/ft_engineering.py`.

El objetivo fue transformar el dataset original en una entrada adecuada para los modelos de Machine Learning, manteniendo el proceso ordenado, reproducible y reutilizable.

Las principales tareas implementadas fueron:

- carga de datos desde archivo
- limpieza de valores nulos o inconsistentes
- corrección de categorías inválidas
- conversión de fechas
- creación de nuevas variables
- selección de columnas predictoras
- separación entre variables numéricas, categóricas nominales y categóricas ordinales
- construcción de un preprocesador con `ColumnTransformer`
- separación en conjuntos de entrenamiento y prueba

### Variables creadas

Se generaron nuevas variables con el objetivo de aportar más información al modelo:

- `ratio_cuota_salario`: relación entre la cuota pactada y el salario del cliente.
- `ratio_capital_salario`: relación entre el capital prestado y el salario del cliente.
- `total_creditos_sectoriales`: suma de créditos registrados en distintos sectores.
- `anio_prestamo`: año de la fecha del préstamo.
- `mes_prestamo`: mes de la fecha del préstamo.

Estas variables buscan capturar relaciones financieras relevantes, por ejemplo cuánto representa una cuota respecto al salario o qué nivel de exposición crediticia tiene el cliente.

### Prevención de Data Leakage

Durante las pruebas iniciales se observaron resultados excesivamente perfectos, con métricas cercanas a 1.00. Esto indicaba una posible presencia de Data Leakage.

El Data Leakage ocurre cuando el modelo recibe información que en un escenario real no estaría disponible al momento de hacer la predicción. Esto puede generar resultados artificialmente altos durante la evaluación, pero poco útiles en producción.

Por este motivo se excluyeron variables que podían contener información posterior al resultado del pago o demasiado directamente relacionada con la mora:

- `saldo_mora`
- `saldo_total`
- `saldo_principal`
- `saldo_mora_codeudor`
- `ratio_mora_saldo_total`
- `flag_tiene_mora`
- `puntaje`
- `puntaje_datacredito`

Luego de retirar estas variables, las métricas bajaron a valores más realistas, permitiendo una evaluación más honesta del desempeño del modelo.

### Preprocesamiento

El preprocesamiento fue organizado según el tipo de variable.

Para variables numéricas se aplicó:

- imputación con mediana
- escalado con `StandardScaler`

Para variables categóricas nominales se aplicó:

- imputación con la categoría más frecuente
- codificación One-Hot Encoding

Para la variable ordinal `tendencia_ingresos` se aplicó:

- imputación con la categoría más frecuente
- codificación ordinal respetando el orden: Decreciente < Estable < Creciente

Toda esta lógica quedó integrada dentro de un `ColumnTransformer`, lo que permite aplicar las mismas transformaciones tanto en entrenamiento como en predicción.

## Entrenamiento y evaluación de modelos

El entrenamiento se implementó en el archivo `src/model_training_evaluation.py`.

El objetivo de esta etapa fue comparar distintos modelos supervisados y seleccionar el de mejor rendimiento de acuerdo con métricas adecuadas para el problema.

Modelos evaluados:

- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier

Cada modelo fue integrado en un pipeline junto con el preprocesador, garantizando que las transformaciones aplicadas a los datos fueran consistentes.

### Métricas utilizadas

Debido al desbalance de clases, se utilizaron varias métricas:

- Accuracy
- Precision
- Recall
- F1-score
- F1 macro
- ROC-AUC
- matriz de confusión

La métrica principal para ordenar los modelos fue `F1 macro`, ya que permite evaluar el desempeño promedio entre ambas clases sin favorecer excesivamente a la clase mayoritaria.

El mejor modelo obtenido fue guardado como: models/best_model.pkl

Este archivo contiene el pipeline completo, incluyendo tanto el preprocesamiento como el modelo entrenado.

## Monitoreo y detección de Data Drift

En esta etapa se implementó un sistema de monitoreo para detectar cambios en la distribución de los datos.

El objetivo fue simular una situación real donde, después de entrenar un modelo con datos históricos, llegan nuevos datos que pueden tener una distribución distinta. Si los datos nuevos cambian demasiado respecto a los datos de entrenamiento, el modelo puede perder rendimiento.

Para esta etapa se utilizaron dos fuentes:

- `Base_de_datos.csv`: datos históricos de referencia.
- `Base_de_datos_con_Data_Drift_Simulado.xlsx`: datos nuevos con drift simulado.

El monitoreo fue implementado en `src/model_monitoring.py`.

### Métricas de drift implementadas

Para variables numéricas se utilizaron:

- Population Stability Index (PSI)
- Kolmogorov-Smirnov Test
- Jensen-Shannon Divergence

Para variables categóricas se utilizó:

- Chi-cuadrado

Cada variable fue evaluada y clasificada según el nivel de alerta detectado:

- `Sin drift`
- `Drift leve`
- `Drift significativo`

El reporte final permite identificar qué variables cambiaron su comportamiento respecto a la base original.

### Objetivo del monitoreo

El monitoreo busca responder preguntas como:

- ¿Los datos nuevos se parecen a los datos con los que entrenamos el modelo?
- ¿Cambió la distribución de las variables?
- ¿Hay variables que podrían afectar el rendimiento del modelo?
- ¿Es necesario revisar, recalibrar o reentrenar el modelo?

En el análisis realizado se detectaron múltiples variables con señales de drift. Esto demuestra la importancia de monitorear un modelo luego de su entrenamiento, especialmente en contextos financieros donde el comportamiento de los clientes puede cambiar con el tiempo.

## Aplicación Streamlit

Además del script de monitoreo, se desarrolló una aplicación visual utilizando Streamlit en el archivo `src/streamlit_app.py`.

La aplicación permite consultar de forma más clara los resultados del monitoreo.

Incluye:

- resumen general del estado del monitoreo
- cantidad de variables analizadas
- cantidad de variables con drift
- porcentaje de variables afectadas
- semáforo visual de riesgo
- gráficos de PSI
- tablas con variables alertadas
- reporte completo de métricas

Esta aplicación permite que un usuario no técnico pueda interpretar rápidamente si el modelo está recibiendo datos estables o si existen señales de alerta.

##  Despliegue del modelo mediante API

En el cuarto avance se disponibilizó el modelo entrenado mediante una API construida con FastAPI.

El objetivo fue permitir que otros sistemas puedan consumir el modelo enviando datos en formato JSON y recibiendo una predicción como respuesta.

La API fue implementada en: src/model_deploy.py


Al iniciar, la API carga el modelo almacenado en: models/best_model.pkl

Luego expone endpoints que permiten verificar el estado del servicio y realizar predicciones.

### Endpoints disponibles

#### GET /

Endpoint raíz para verificar que la API está activa.

Respuesta esperada:

```json
{
  "mensaje": "API de Riesgo Crediticio activa",
  "estado": "ok"
}
```

---

#### GET /health

Endpoint de salud del servicio. Permite verificar si el modelo fue cargado correctamente.

Respuesta esperada:

```json
{
  "status": "ok",
  "modelo_cargado": true,
  "ruta_modelo": "/app/models/best_model.pkl"
}
```

---

#### POST /predict

Endpoint principal de predicción.

Recibe una lista de registros y devuelve para cada uno:

- índice del registro
- clase predicha
- interpretación de la predicción
- probabilidad estimada de pago a tiempo

Ejemplo de entrada:

```json
{
  "registros": [
    {
      "fecha_prestamo": "2025-03-15",
      "tipo_credito": 4,
      "tipo_laboral": "Empleado",
      "tendencia_ingresos": "Creciente",
      "capital_prestado": 2500000,
      "plazo_meses": 12,
      "edad_cliente": 35,
      "salario_cliente": 3200000,
      "total_otros_prestamos": 450000,
      "cuota_pactada": 280000,
      "cant_creditosvigentes": 2,
      "huella_consulta": 1,
      "creditos_sectorFinanciero": 1,
      "creditos_sectorCooperativo": 0,
      "creditos_sectorReal": 0,
      "promedio_ingresos_datacredito": 3000000
    }
  ]
}
```

Ejemplo de respuesta:

```json
{
  "cantidad_registros": 1,
  "predicciones": [
    {
      "indice": 0,
      "prediccion": 1,
      "interpretacion": "Pagará a tiempo",
      "probabilidad_pago_a_tiempo": 0.94
    }
  ]
}
```

## Docker

Para asegurar que la API pueda ejecutarse en un entorno reproducible, se creó una imagen Docker del proyecto.

El contenedor incluye:

- imagen base de Python
- dependencias instaladas desde `requirements.txt`
- código fuente del proyecto
- modelo entrenado
- configuración para ejecutar Uvicorn
- API FastAPI disponible en el puerto 8000


### Documentación interactiva

Una vez levantado el contenedor, la documentación de Swagger se encuentra disponible en:  http://127.0.0.1:8000/docs

Desde allí se pueden probar los endpoints `/`, `/health` y `/predict`.

## ====================================================================================================

## Cómo ejecutar el proyecto

## Ejecución local
### Crear entorno virtual

```bash
python -m venv .venv
```

### Activar entorno virtual en Windows

```bash
.venv\Scripts\activate
```

### Instalar dependencias

```bash
pip install -r requirements.txt
```

---

## Ejecutar entrenamiento

```bash
python src/model_training_evaluation.py
```

Este comando entrena los modelos definidos, compara sus métricas y guarda el mejor pipeline en `models/best_model.pkl`.

---

## Ejecutar monitoreo con Streamlit

```bash
python -m streamlit run src/streamlit_app.py
```

Este comando abre la aplicación visual para revisar el reporte de Data Drift.

---

## Ejecutar API localmente

```bash
python src/model_deploy.py
```

Luego abrir:

```text
http://127.0.0.1:8000/docs
```

## Versionado del proyecto

El proyecto fue versionado mediante Git y tags para registrar el avance progresivo de cada etapa.

| Versión | Descripción |
|---|---|
| v1.0.1 | Carga de datos y comprensión inicial |
| v1.1.0 | Ingeniería de características |
| v1.1.1 | Entrenamiento y evaluación de modelos |
| v1.1.2 | Guardado del mejor modelo entrenado |
| v1.2.0 | Monitoreo y detección de Data Drift |
| v1.2.1 | Aplicación Streamlit |
| v1.3.0 | API FastAPI y Docker |


## Conclusión orientada al negocio

El proyecto desarrollado responde a una necesidad concreta de una entidad financiera: mejorar la capacidad de anticipar qué clientes tienen mayor probabilidad de pagar sus créditos a tiempo y cuáles podrían representar un riesgo de incumplimiento.

A partir del análisis de datos históricos de préstamos, ingresos, comportamiento financiero y características de los clientes, se construyó una solución que permite apoyar el proceso de evaluación crediticia con información objetiva basada en datos.

El modelo predictivo permite estimar la probabilidad de pago a tiempo para nuevos solicitantes o clientes existentes. Esto puede ayudar a la empresa a tomar mejores decisiones al momento de aprobar créditos, definir condiciones comerciales, priorizar controles o diseñar estrategias preventivas de seguimiento.

Durante el desarrollo también se identificó un aspecto clave para el negocio: la base presentaba un fuerte desbalance, ya que la mayoría de los clientes pagaban a tiempo. Por este motivo, no se evaluó el modelo únicamente con accuracy, sino también con métricas que permiten observar mejor el comportamiento sobre la clase de mayor riesgo. Esto es importante porque, para una empresa financiera, detectar correctamente los posibles casos de incumplimiento puede tener más impacto que simplemente acertar sobre los casos más frecuentes.

Además, se incorporó un sistema de monitoreo de Data Drift para controlar si los nuevos datos mantienen un comportamiento similar al de los datos históricos. Esto es especialmente relevante en el contexto crediticio, donde las condiciones económicas, laborales y financieras de los clientes pueden cambiar con el tiempo. Si los datos nuevos se alejan demasiado de los datos usados para entrenar el modelo, la empresa puede detectar la alerta y evaluar si necesita revisar, ajustar o reentrenar el modelo.

Finalmente, la solución fue disponibilizada mediante una API y contenida con Docker, lo que facilita su integración con otros sistemas internos de la empresa. De esta manera, el modelo no queda limitado a un entorno de desarrollo, sino que puede ser consumido por aplicaciones, tableros o procesos automatizados de evaluación crediticia.

En conjunto, este proyecto aporta una herramienta práctica para reducir la incertidumbre en la toma de decisiones crediticias, mejorar el control del riesgo y avanzar hacia una gestión más automatizada, monitoreable y basada en datos.