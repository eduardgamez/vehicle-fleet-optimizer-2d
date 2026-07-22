#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FASE 1.5 (CPU) · De los CSV de rutas al SUPERSET de muestras en binario.

Es el paso que hace viable el barrido grande. En el pipeline local, cada
configuración que cambiaba la representación (nº de vecinos, historia,
horizonte) obligaba a releer todos los CSV y reconstruir las muestras fila a
fila en Python; con millones de filas eso cuesta más que entrenar. Aquí se hace
UNA sola vez:

  · se leen todos los CSV en paralelo (un proceso por fichero),
  · se construyen las muestras VECTORIZADAS y en la representación MÁXIMA
    (el «superset»), junto al tiempo de cierre de cada bloque de vecino,
  · se guardan en .npy, que el entrenamiento abre por mapeo de memoria.

Después, cada punto del barrido se obtiene recortando columnas de ese superset
(ver `vectorizado.py`), sin volver a tocar los CSV ni la CPU.

Uso:
    python preparar_datos.py --rutas datos/rutas --salida datos/muestras
    python preparar_datos.py --paso 2        (se queda 1 de cada 2 instantes)
"""

import argparse
import ast
import glob
import json
import math
import multiprocessing
import os
import time
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import comun
from comun import MUESTRAS_DIR, RUTAS_DIR, asegurar
import politica as pol
import vectorizado as vec

# Representación MÁXIMA del superset. Tiene que cubrir el mayor valor de cada eje
# del espacio de búsqueda, y NO se puede subir después sin rehacer los datos.
# Con flotas de hasta 8 vehículos, un vehículo nunca ve más de 7 vecinos.
N_VEC_MAX = 7
H_MAX = 10
HOR_MAX = 20


# --------------------------------------------------------------------------- #
# Lectura de un CSV (misma semántica que entrenar.leer_runs, por fichero)
# --------------------------------------------------------------------------- #
def leer_archivo(arch):
    """[(run_id, condiciones, modo_opt, {vid: array (n, 6)})] de un CSV.

    Descarta runs repetidos (una semilla reintentada tras un corte) y filas mal
    formadas (la última línea de un fichero que se quedó a medias)."""
    from entrenar import _parsear_condiciones
    runs, vistos = [], set()
    conds, filas, run_actual, opt_actual = None, {}, None, "secuencial"

    def cerrar():
        if conds and filas and run_actual not in vistos:
            vistos.add(run_actual)
            runs.append((run_actual, conds, opt_actual,
                         {vid: np.array(v, dtype=np.float64)
                          for vid, v in filas.items()}))

    with open(arch, encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if not linea or linea.startswith("run,"):
                continue
            if linea.startswith("#"):
                cerrar()
                conds, filas = None, {}
                if "condiciones_iniciales=" in linea:
                    try:
                        conds = _parsear_condiciones(linea)
                        run_actual = linea.split("run=", 1)[1].split()[0]
                        opt_actual = ("secuencial" if "opt=" not in linea
                                      else linea.split("opt=", 1)[1].split()[0])
                    except (ValueError, SyntaxError, KeyError, IndexError):
                        conds = None
                continue
            if conds is None:
                continue
            try:
                c = linea.split(",")
                vid = int(c[1])
                x1, y1, x2, y2 = map(float, c[3:7])
                v, th, a, giro = map(float, c[7:11])
            except (ValueError, IndexError):
                continue                      # línea truncada por un corte
            filas.setdefault(vid, []).append(
                ((x1 + x2) / 2.0, (y1 + y2) / 2.0, th, v, a, giro))
    cerrar()
    return runs


def es_validacion(run_id):
    """~10 % de los runs a validación, estable entre ejecuciones (mismo criterio
    que el pipeline local). Es la validación por MSE, que solo sirve para elegir
    la mejor época; la nota que decide el modelo sale de escenarios nuevos."""
    return zlib.crc32(str(run_id).encode()) % 10 == 0


# --------------------------------------------------------------------------- #
# Worker: un CSV → un lote de .npy
# --------------------------------------------------------------------------- #
def procesar(tarea):
    arch, salida, paso, etiqueta = tarea
    runs = leer_archivo(arch)
    Xs, TCs, Ys, Vs = [], [], [], []
    for run_id, conds, opt, datos in runs:
        X, TC, Y = vec.superset_run(conds, opt, datos, N_VEC_MAX, H_MAX, HOR_MAX)
        if paso > 1:
            X, TC, Y = X[::paso], TC[::paso], Y[::paso]
        Xs.append(X)
        TCs.append(TC)
        Ys.append(Y)
        Vs.append(np.full(len(X), es_validacion(run_id), dtype=bool))
    if not Xs:
        return etiqueta, 0, len(runs)
    X = np.concatenate(Xs)
    np.save(os.path.join(salida, f"{etiqueta}_X.npy"), X)
    np.save(os.path.join(salida, f"{etiqueta}_TC.npy"), np.concatenate(TCs))
    np.save(os.path.join(salida, f"{etiqueta}_Y.npy"), np.concatenate(Ys))
    np.save(os.path.join(salida, f"{etiqueta}_val.npy"), np.concatenate(Vs))
    return etiqueta, len(X), len(runs)


def main():
    ap = argparse.ArgumentParser(
        description="Convierte los CSV de rutas en el superset de muestras.")
    ap.add_argument("--rutas", default=RUTAS_DIR)
    ap.add_argument("--salida", default=MUESTRAS_DIR)
    ap.add_argument("--paso", type=int, default=1,
                    help="se queda con 1 instante de cada N. A DT=0,1 s los "
                         "instantes seguidos son casi idénticos, así que 2 o 3 "
                         "recorta el dataset sin perder casi información (def. 1)")
    ap.add_argument("--procesos", type=int,
                    default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    archivos = sorted(glob.glob(os.path.join(args.rutas, "**", "*.csv"),
                                recursive=True))
    if not archivos:
        raise SystemExit(f"No hay CSV en {args.rutas}.")
    asegurar(args.salida)
    print(f"[prep] {len(archivos)} ficheros · superset n_vecinos={N_VEC_MAX} "
          f"h_pasado={H_MAX} horizonte={HOR_MAX} "
          f"(dim {vec.dim_super(N_VEC_MAX, H_MAX)}) · paso {args.paso}",
          flush=True)

    tareas = [(a, args.salida, args.paso, f"lote{i:04d}")
              for i, a in enumerate(archivos)]
    t0 = time.perf_counter()
    procesos = max(1, min(args.procesos, len(tareas)))
    if procesos == 1:
        resultados = [procesar(t) for t in tareas]
    else:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=procesos, mp_context=ctx) as ex:
            resultados = list(ex.map(procesar, tareas))

    lotes = [(e, n) for e, n, _ in resultados if n]
    filas = sum(n for _, n in lotes)
    runs = sum(r for _, _, r in resultados)
    meta = {"n_vec_max": N_VEC_MAX, "h_max": H_MAX, "hor_max": HOR_MAX,
            "dim_super": vec.dim_super(N_VEC_MAX, H_MAX), "n_pred": pol.N_PRED,
            "paso": args.paso, "filas": filas, "runs": runs,
            "lotes": [e for e, _ in lotes]}
    with open(os.path.join(args.salida, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[prep] {runs:,} runs · {filas:,} muestras · "
          f"{time.perf_counter() - t0:.1f} s → {args.salida}", flush=True)

    destino = comun.bucket_env("muestras")
    if destino:
        comun.sincronizar(args.salida, destino)


if __name__ == "__main__":
    main()
