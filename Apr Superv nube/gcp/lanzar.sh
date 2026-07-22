#!/usr/bin/env bash
# Lanzamiento completo en Google Cloud. Ejecutar desde la RAÍZ del proyecto.
#
#   bash "Apr Superv nube/gcp/lanzar.sh" preparar     # una sola vez
#   bash "Apr Superv nube/gcp/lanzar.sh" generar      # fase 1 (CPU, spot)
#   bash "Apr Superv nube/gcp/lanzar.sh" muestras     # fase 1.5 (CPU, 1 máquina)
#   bash "Apr Superv nube/gcp/lanzar.sh" amplia       # fase 2a (GPU, 35 % datos)
#   bash "Apr Superv nube/gcp/lanzar.sh" fina         # fase 2b (GPU, 65 % datos)
#   bash "Apr Superv nube/gcp/lanzar.sh" final        # fase 3 (100 % datos)
#
# Las tres últimas van escalonadas a propósito: comparar miles de
# configuraciones entre sí no necesita todo el dataset, afinar la rejilla sí
# necesita bastante, y el modelo que se queda se entrena con todo.
set -euo pipefail

PROYECTO="${TDR_PROYECTO:?exporta TDR_PROYECTO con el id del proyecto de GCP}"
REGION="${TDR_REGION:-europe-southwest1}"          # Madrid: barato y cerca
BUCKET="${TDR_BUCKET:-gs://${PROYECTO}-tdr}"
IMAGEN="${REGION}-docker.pkg.dev/${PROYECTO}/tdr/nube:latest"

# Tamaño del trabajo. 40 tareas × 32 vCPU = 1 280 núcleos en paralelo.
ESCENARIOS="${TDR_ESCENARIOS:-10000}"
TAREAS_CPU="${TDR_TAREAS_CPU:-40}"
TAREAS_GPU="${TDR_TAREAS_GPU:-8}"
CONFIGS="${TDR_CONFIGS:-4000}"

plantilla() {  # plantilla TAREAS FRACCION CONFIGS ESPACIO fichero
  sed -e "s|{{IMAGEN}}|${IMAGEN}|g" -e "s|{{BUCKET}}|${BUCKET}|g" \
      -e "s|{{ESCENARIOS}}|${ESCENARIOS}|g" -e "s|{{TAREAS}}|$1|g" \
      -e "s|{{FRACCION}}|$2|g" -e "s|{{CONFIGS}}|$3|g" \
      -e "s|{{ESPACIO}}|$4|g" "$5"
}

case "${1:-}" in
preparar)
  gcloud config set project "${PROYECTO}"
  gcloud services enable batch.googleapis.com compute.googleapis.com \
      artifactregistry.googleapis.com storage.googleapis.com
  gcloud artifacts repositories create tdr --repository-format=docker \
      --location="${REGION}" 2>/dev/null || true
  gcloud storage buckets create "${BUCKET}" --location="${REGION}" 2>/dev/null || true
  gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet
  docker build -f "Apr Superv nube/Dockerfile" -t "${IMAGEN}" .
  docker push "${IMAGEN}"
  # Los escenarios de evaluación se generan aquí mismo (tardan segundos) y se
  # suben, para que todas las tareas del barrido puntúen con LOS MISMOS casos.
  TDR_BUCKET="${BUCKET}" python "Apr Superv nube/escenarios.py" --n 400
  echo "listo · imagen ${IMAGEN} · bucket ${BUCKET}"
  ;;

generar)
  plantilla "${TAREAS_CPU}" 1 0 "" \
      "Apr Superv nube/gcp/job_generacion.json" > /tmp/job.json
  gcloud batch jobs submit "tdr-gen-$(date +%s)" --location="${REGION}" \
      --config /tmp/job.json
  ;;

muestras)
  # Un solo bicho grande: es un paso corto y conviene que el superset salga en
  # un único sitio.
  plantilla 1 1 0 "" "Apr Superv nube/gcp/job_muestras.json" > /tmp/job.json
  gcloud batch jobs submit "tdr-prep-$(date +%s)" --location="${REGION}" \
      --config /tmp/job.json
  ;;

amplia)
  # Búsqueda aleatoria por el espacio entero, con el 35 % de los datos.
  plantilla "${TAREAS_GPU}" 0.35 "${CONFIGS}" "" \
      "Apr Superv nube/gcp/job_barrido.json" > /tmp/job.json
  gcloud batch jobs submit "tdr-amplia-$(date +%s)" --location="${REGION}" \
      --config /tmp/job.json
  ;;

fina)
  # Rejilla fina alrededor de lo que ganó, con el 65 % de los datos y sin criba
  # (aquí todas las candidatas son buenas y no interesa descartar por prisa).
  plantilla "${TAREAS_GPU}" 0.65 "${CONFIGS}" "espacio_fino.json" \
      "Apr Superv nube/gcp/job_barrido.json" > /tmp/job.json
  gcloud batch jobs submit "tdr-fina-$(date +%s)" --location="${REGION}" \
      --config /tmp/job.json
  ;;

final)
  plantilla 1 1 0 "" "Apr Superv nube/gcp/job_final.json" > /tmp/job.json
  gcloud batch jobs submit "tdr-final-$(date +%s)" --location="${REGION}" \
      --config /tmp/job.json
  ;;

*)
  echo "uso: lanzar.sh {preparar|generar|muestras|barrido|final}" >&2
  exit 1
  ;;
esac
