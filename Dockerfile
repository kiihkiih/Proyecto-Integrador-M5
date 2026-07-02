# ================================================
# IMAGEN BASE DE PYTHON
# ================================================

FROM python:3.14-slim


# ================================================
# CONFIGURACIÓN DEL DIRECTORIO DE TRABAJO
# ================================================

WORKDIR /app

ENV PYTHONPATH=/app/src


# ================================================
# COPIA DE DEPENDENCIAS
# ================================================

COPY requirements.txt .


# ================================================
# INSTALACIÓN DE DEPENDENCIAS
# ================================================

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt


# ================================================
# COPIA DEL CÓDIGO Y ARCHIVOS DEL PROYECTO
# ================================================

COPY . .


# ================================================
# EXPOSICIÓN DEL PUERTO DE LA API
# ================================================

EXPOSE 8000


# ================================================
# COMANDO DE EJECUCIÓN
# ================================================

CMD ["uvicorn", "src.model_deploy:app", "--host", "0.0.0.0", "--port", "8000"]