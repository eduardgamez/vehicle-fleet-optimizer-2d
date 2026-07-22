#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilidades compartidas por los scripts de nube.

Esta carpeta NO duplica el pipeline: reutiliza tal cual `nucleo.py`,
`politica.py` y `entrenar.py` de «Apr Superv local» (se añaden al sys.path aquí),
y solo reimplementa las partes que hacen falta para escalar:

  · reparto del trabajo en TAREAS independientes (una por máquina/contenedor),
  · reanudación tras una interrupción (las máquinas spot se pueden cortar),
  · construcción VECTORIZADA de las muestras y del rollout.

Las variables de entrada/salida de la red, el espacio de hiperparámetros y el
criterio de puntuación son EXACTAMENTE los del pipeline local.
"""

import os
import sys

# La consola de Windows suele ser cp1252 y no puede imprimir algunos símbolos;
# forzar UTF-8 evita que un print tumbe una ejecución de horas (igual que en
# entrenar.py del pipeline local).
for _flujo in (sys.stdout, sys.stderr):
    try:
        _flujo.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

RAIZ_NUBE = os.path.dirname(os.path.abspath(__file__))
RAIZ_PROYECTO = os.path.dirname(RAIZ_NUBE)
RAIZ_LOCAL = os.path.join(RAIZ_PROYECTO, "Apr Superv local")

# `nucleo`, `politica` y `entrenar` se importan del pipeline local: una sola
# definición de la física, del vector de entrada y del espacio de búsqueda.
if RAIZ_LOCAL not in sys.path:
    sys.path.insert(0, RAIZ_LOCAL)

# Carpetas de trabajo de la nube. Se pueden reapuntar con variables de entorno
# para que cada máquina escriba en su disco local y suba al bucket al final.
DATOS_DIR = os.environ.get("TDR_DATOS", os.path.join(RAIZ_NUBE, "datos"))
RUTAS_DIR = os.environ.get("TDR_RUTAS", os.path.join(DATOS_DIR, "rutas"))
MUESTRAS_DIR = os.environ.get("TDR_MUESTRAS", os.path.join(DATOS_DIR, "muestras"))
MODELOS_DIR = os.environ.get("TDR_MODELOS", os.path.join(DATOS_DIR, "modelos"))

MAPA_ENTRENAMIENTO = os.path.join(RAIZ_PROYECTO, "mapas",
                                  "mapa_entrenamiento.json")


def asegurar(*carpetas):
    for c in carpetas:
        os.makedirs(c, exist_ok=True)


# --------------------------------------------------------------------------- #
# Identidad de la tarea dentro de un trabajo de N tareas paralelas
# --------------------------------------------------------------------------- #
def indice_tarea(por_defecto=0):
    """Índice de esta tarea (0..N-1). Lo publican los orquestadores como
    variable de entorno; si no hay ninguna, es una ejecución local suelta."""
    for var in ("TDR_TAREA", "BATCH_TASK_INDEX", "AWS_BATCH_JOB_ARRAY_INDEX",
                "JOB_COMPLETION_INDEX"):
        if os.environ.get(var):
            return int(os.environ[var])
    return por_defecto


def total_tareas(por_defecto=1):
    for var in ("TDR_TAREAS", "BATCH_TASK_COUNT", "AWS_BATCH_JOB_ARRAY_SIZE"):
        if os.environ.get(var):
            return int(os.environ[var])
    return por_defecto


def reparto(elementos, indice=None, total=None):
    """Trozo de 'elementos' que le toca a esta tarea (reparto por módulo, para
    que trozos consecutivos queden equilibrados aunque el coste crezca con el
    índice)."""
    i = indice_tarea() if indice is None else indice
    n = total_tareas() if total is None else total
    return elementos[i::max(1, n)]


# --------------------------------------------------------------------------- #
# Sincronización con el almacenamiento de objetos (opcional)
# --------------------------------------------------------------------------- #
def sincronizar(origen, destino):
    """Copia una carpeta a/desde un bucket con la CLI que corresponda
    (gs:// → gcloud storage, s3:// → aws s3). Sin prefijo remoto no hace nada:
    así los mismos scripts corren en un portátil sin tocar nada."""
    import subprocess
    remoto = origen.startswith(("gs://", "s3://")) or \
        destino.startswith(("gs://", "s3://"))
    if not remoto:
        return False
    if "s3://" in (origen + destino):
        cmd = ["aws", "s3", "sync", origen, destino]
    else:
        cmd = ["gcloud", "storage", "rsync", "-r", origen, destino]
    print(f"[sync] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    return True


def arrancar_sincronizacion(local, remoto, minutos=10):
    """Sube 'local' al bucket cada 'minutos' en segundo plano.

    Es lo que hace baratas las máquinas spot: si cortan la máquina se pierde,
    como mucho, el trabajo de esos minutos, y al reiniciar la tarea se baja lo
    ya subido y se reanuda desde ahí. Escribir directamente en el bucket no
    sirve: son miles de añadidos pequeños y cada uno reescribiría el objeto
    entero."""
    import threading
    import time
    if not remoto:
        return None

    def bucle():
        while True:
            time.sleep(minutos * 60)
            try:
                sincronizar(local, remoto)
            except Exception as e:                       # noqa: BLE001
                print(f"[sync] fallo al subir (se reintenta): {e}", flush=True)

    hilo = threading.Thread(target=bucle, daemon=True)
    hilo.start()
    return hilo


def recuperar(local, remoto):
    """Baja lo que ya hubiera en el bucket antes de empezar (reanudación)."""
    if not remoto:
        return
    try:
        sincronizar(remoto, local)
    except Exception as e:                               # noqa: BLE001
        print(f"[sync] no se ha podido bajar el estado previo: {e}", flush=True)


def bucket_env(sufijo):
    """URL del bucket para 'sufijo' (rutas/, muestras/, modelos/) si se ha
    configurado TDR_BUCKET; si no, None."""
    base = os.environ.get("TDR_BUCKET")
    return f"{base.rstrip('/')}/{sufijo}" if base else None
