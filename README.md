# Tarea 2 – Portafolio de prácticas y proyecto individual de Docker

Ek Poot Pulido Michelle Diane – 211597866 – Programación Bioinformática – D01

| Carpeta | Contenido |
| --- | --- |
| practica1-nginx | Servidor web con Nginx y mapeo de puertos |
| practica2-python-pandas | Análisis con Pandas, docker save / load |
| practica3-flask-mysql | Flask + MySQL con Docker Compose y volumen |
| practica4-multietapa | Dockerfile multietapa con R y Python |
| practica5-cmd-entrypoint | Diferencia entre CMD y ENTRYPOINT |
| practica6-tensorflow | Clasificación de imágenes con TensorFlow |
| practica7-optimizacion | Comparación de tamaño: normal, Alpine y multietapa |
| proyecto-ml-docker | Proyecto individual: clasificador de tumores de mama con FastAPI |

## Ejecutar el proyecto individual

    cd proyecto-ml-docker
    docker build -t cancer-api:v1 .
    docker run -d -p 8000:8000 --name cancer-api cancer-api:v1

Documentación de la API: http://localhost:8000/docs
