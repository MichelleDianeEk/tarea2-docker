# Imagen base ligera con Python 3.12 (la misma versión usada para entrenar).
FROM python:3.12-slim

# Directorio de trabajo dentro de la imagen.
WORKDIR /app

# Copiar primero requirements.txt para aprovechar la caché de Docker:
# si solo cambia el código, no se reinstalan las librerías.
COPY requirements.txt .

# Instalar las dependencias sin guardar la caché de pip (imagen más pequeña).
RUN pip install --no-cache-dir -r requirements.txt

# Copiar la API y el modelo ya entrenado (no se entrena dentro de Docker).
COPY app.py .
COPY model/ ./model/

# Documentar que la API escucha en el puerto 8000 del contenedor.
EXPOSE 8000

# Iniciar el servidor uvicorn con la API; 0.0.0.0 permite conexiones desde fuera del contenedor.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
