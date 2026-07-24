# Imagen ÚNICA para las dos fases: la de CPU y la de GPU comparten casi todo el
# código (nucleo, politica, vectorizado), y mantener dos imágenes sincronizadas
# da más problemas de los que ahorra.
#
# Construir desde la RAÍZ del proyecto, no desde esta carpeta:
#   docker build -f "Apr Superv nube/Dockerfile" -t tdr-nube .
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    NUMBA_CACHE_DIR=/tmp/numba \
    TDR_DATOS=/datos \
    CLOUDSDK_PYTHON=/usr/local/bin/python3

# La CLI de gcloud va dentro porque las tareas suben su avance al bucket cada
# pocos minutos por su cuenta (ver comun.arrancar_sincronizacion). En una VM de
# GCP se autentica sola con la cuenta de servicio de la máquina.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl gnupg ca-certificates \
    && echo "deb https://packages.cloud.google.com/apt cloud-sdk main" \
        > /etc/apt/sources.list.d/google-cloud-sdk.list \
    && curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
        | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg \
    && sed -i 's|deb |deb [signed-by=/usr/share/keyrings/cloud.google.gpg] |' \
        /etc/apt/sources.list.d/google-cloud-sdk.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        google-cloud-cli \
    && rm -rf /var/lib/apt/lists/* \
    # La gcloud de apt busca /usr/bin/python3, pero en python:3.12-slim el
    # intérprete está en /usr/local/bin. Sin este enlace, gcloud falla con
    # "exec: /usr/bin/python3: not found" y NINGUNA subida al bucket funciona.
    && ln -sf /usr/local/bin/python3 /usr/bin/python3

WORKDIR /app
# Sintaxis JSON en los COPY: es la ÚNICA forma que Docker parsea correctamente
# cuando las rutas llevan espacios ("Apr Superv nube", "Apr Superv local").
COPY ["Apr Superv nube/requirements_nube.txt", "./"]
RUN pip install --no-cache-dir -r requirements_nube.txt

COPY ["mapas/", "./mapas/"]
COPY ["Apr Superv local/nucleo.py", "Apr Superv local/politica.py", \
      "Apr Superv local/generador.py", "Apr Superv local/entrenar.py", \
      "./Apr Superv local/"]
COPY ["Apr Superv nube/", "./Apr Superv nube/"]

WORKDIR "/app/Apr Superv nube"

# Compila los kernels de numba en la construcción, para que las 100 máquinas de
# la fase 1 no paguen cada una el minuto de compilación al arrancar.
RUN python -c "import comun, nucleo; nucleo.warmup()"

CMD ["python", "generador_nube.py", "--help"]
