#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generador HEADLESS de rutas para el dataset de aprendizaje supervisado.

Sin interfaz gráfica: genera escenarios aleatorios (flotas con todo al azar,
igual que el modo "Aleatorios" de la GUI) sobre un MAPA FIJO, los resuelve con
el planificador clásico y acumula las rutas definitivas en CSVs CLASIFICADOS
por parámetros iniciales (una carpeta por nº de vehículos, un fichero por
proceso worker). El mapa es siempre el mismo entre ejecuciones y workers
(entorno controlado): si no existe, se crea una única vez de forma determinista
y se guarda en mapas/mapa_entrenamiento.json (la GUI puede cargarlo también).

NUNCA se descartan escenarios por tardar: cada uno se busca hasta los topes de
nodos del planificador (el mismo criterio que la GUI en el orden definitivo).

Ejemplos:
    python generador.py --escenarios 200
    python generador.py --escenarios 5000 --workers 12 --veh 2-8 --calidad 3
    python generador.py --escenarios 100 --opt global --max-ordenes 12
"""

import argparse
import json
import math
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

from nucleo import (
    W, H, DT, CAP_COMPLETO, PLAZO_VEH_CAND, MAPAS_DIR, RUTAS_DIR, PALETA,
    Entorno, Planificador, Reservas,
    espec_a_vehiculo, generar_ordenes, planificar_orden, bloqueos_metas,
    guardar_dataset_supervisado, cargar_mapa_en, warmup,
)

MAPA_ENTRENAMIENTO = os.path.join(MAPAS_DIR, "mapa_entrenamiento.json")
SEMILLA_MAPA = 20260718        # el mapa fijo se genera SIEMPRE con esta semilla


# --------------------------------------------------------------------------- #
# Mapa fijo de entrenamiento
# --------------------------------------------------------------------------- #
def asegurar_mapa(ruta, densidad):
    """Devuelve la ruta de un mapa fijo, creándolo (determinista) si no existe."""
    if os.path.exists(ruta):
        return ruta
    estado = random.getstate()
    random.seed(SEMILLA_MAPA)
    env = Entorno()
    env.generar(densidad)
    random.setstate(estado)
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    datos = {"version": 1, "mundo": [W, H],
             "obstaculos": [[[float(px), float(py)] for px, py in pol]
                            for pol in env.obstaculos]}
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(datos, f)
    print(f"[mapa] creado el mapa fijo de entrenamiento → {ruta} "
          f"({len(datos['obstaculos'])} obstáculos)")
    return ruta


# --------------------------------------------------------------------------- #
# Escenarios aleatorios (misma lógica que el modo "Aleatorios" de la GUI)
# --------------------------------------------------------------------------- #
def _muestrear(env, length, width, otros, sep):
    for _ in range(4000):
        x = random.uniform(length, W - length)
        y = random.uniform(length, H - length)
        th = random.uniform(0, 2 * math.pi)
        if not env.libre(x, y, th, length, width, margen=0.3):
            continue
        if all(math.hypot(x - ox, y - oy) >= sep for ox, oy in otros):
            return (x, y, th)
    return None


def _angulo_llegada(env, mx, my, length, width):
    for _ in range(25):
        c = random.uniform(0, 2 * math.pi)
        if env.libre(mx, my, c, length, width, margen=0.25):
            return c
    return None


def aleatorios_spec(env, n):
    """n especificaciones al azar (diccionarios en grados, como los del texto de
    la GUI) o None si no hay sitio libre para todos."""
    especs, inicios, metas = [], [], []
    for i in range(n):
        largo = random.uniform(0.7, 1.2)
        ancho = largo * random.uniform(0.50, 0.62)
        sep = largo * 1.6
        ini = _muestrear(env, largo, ancho, inicios, sep)
        met = _muestrear(env, largo, ancho, metas, sep)
        if ini is None or met is None:
            return None
        ix, iy, ith = ini
        mx, my, _ = met
        thm = _angulo_llegada(env, mx, my, largo, ancho)
        especs.append({
            "id": i + 1,
            "inicio": (round(ix, 2), round(iy, 2)),
            "giro_inicial": round(math.degrees(ith), 1),
            "meta": (round(mx, 2), round(my, 2)),
            "angulo_llegada": (round(math.degrees(thm), 1)
                               if thm is not None else "libre"),
            "largo": round(largo, 2),
            "ancho": round(ancho, 2),
            "v_max": round(random.uniform(1.8, 3), 2),
            "a_max": round(random.uniform(0.6, 1), 2),
            "giro_max": round(random.uniform(28.0, 40.0), 1),
            "grupo": random.randint(1, 2),
            "prioridad": random.randint(1, 3),
        })
        inicios.append((ix, iy))
        metas.append((mx, my))
    return especs


def _reubicar(env, veh, vehiculos):
    """Si un vehículo no tiene ruta, prueba extremos nuevos (como la GUI)."""
    sep = veh.length * 1.6
    otros_ini = [v.inicio[:2] for v in vehiculos if v.idx != veh.idx]
    otros_meta = [v.meta for v in vehiculos if v.idx != veh.idx]
    ini = _muestrear(env, veh.length, veh.width, otros_ini, sep)
    met = _muestrear(env, veh.length, veh.width, otros_meta, sep)
    if ini is None or met is None:
        return False
    ix, iy, ith = ini
    thm = _angulo_llegada(env, met[0], met[1], veh.length, veh.width)
    veh.inicio = (ix, iy, ith)
    veh.meta = (met[0], met[1])
    veh.meta_th = thm
    return True


# --------------------------------------------------------------------------- #
# Planificación de la flota (calco del flujo de la GUI, sin UI ni paralelo
# interno: el paralelismo de este generador es ENTRE escenarios)
# --------------------------------------------------------------------------- #
def planificar_flota(env, pl, vehiculos, opt, max_cand, reubicable=True):
    indices = list(range(len(vehiculos)))
    ordenes = generar_ordenes(vehiculos, indices, opt, max_cand)
    cap_prev = pl.max_exp
    base = Reservas()
    mejor = None
    try:
        explorar = len(ordenes) > 1
        for orden in ordenes:
            if explorar:
                # Presupuesto por candidato (solo COMPARA órdenes; el mejor se
                # rescata después sin límite de tiempo, así que no se descarta
                # ningún escenario por lento).
                pl.max_exp = cap_prev
                dur_veh = PLAZO_VEH_CAND
            else:
                pl.max_exp = CAP_COMPLETO
                pl.deadline = None
                dur_veh = None
            trays, motivos, coste = planificar_orden(
                pl, vehiculos, orden, base, deadline_dur=dur_veh)
            fallos = sum(1 for i in orden if trays[i] is None)
            if mejor is None or (fallos, coste) < (mejor[0], mejor[1]):
                mejor = (fallos, coste, trays, motivos, orden)
            if fallos == 0 and not explorar:
                break

        fallos, _, trays, motivos, orden = mejor
        if fallos:
            # Rescate exhaustivo de los vehículos sin ruta (sin plazos de reloj).
            pl.max_exp = CAP_COMPLETO
            pl.deadline = None
            reservas = base.copia()
            for i in orden:
                if trays[i] is not None:
                    v = vehiculos[i]
                    reservas.add(trays[i], v.length, v.width)
            for i in orden:
                if trays[i] is not None:
                    continue
                veh = vehiculos[i]
                pl.bloqueos = bloqueos_metas(vehiculos, orden, excepto=i,
                                             pose0=veh.inicio)
                pl.deadline = None
                traj = pl.planificar(veh, reservas)
                motivos[i] = pl.motivo
                intentos = 0
                while pl.motivo == "sin_ruta" and reubicable and intentos < 12:
                    if not _reubicar(env, veh, vehiculos):
                        break
                    pl.bloqueos = bloqueos_metas(vehiculos, orden, excepto=i,
                                                 pose0=veh.inicio)
                    pl.deadline = None
                    traj = pl.planificar(veh, reservas)
                    motivos[i] = pl.motivo
                    intentos += 1
                if traj is not None and len(traj) >= 2:
                    trays[i] = traj
                    motivos[i] = "ok"
                    reservas.add(traj, veh.length, veh.width)
                elif traj:
                    trays[i] = [traj[0], traj[0]]
                    motivos[i] = "ok"
        return trays, motivos
    finally:
        pl.deadline = None
        pl.max_exp = cap_prev


# --------------------------------------------------------------------------- #
# Worker: resuelve una tanda de escenarios y los acumula en su propio CSV
# --------------------------------------------------------------------------- #
def trabajar(args):
    (wid, semillas, ruta_mapa, veh_min, veh_max, calidad, opt, max_cand,
     salida) = args
    env = Entorno()
    cargar_mapa_en(env, ruta_mapa)
    warmup()
    pl = Planificador(env)
    pl.configurar_calidad(calidad)

    filas_tot, veh_ok, veh_tot, saltados = 0, 0, 0, 0
    for k, sem in enumerate(semillas):
        random.seed(sem)
        n = random.randint(veh_min, veh_max)
        especs = aleatorios_spec(env, n)
        if especs is None:
            saltados += 1
            continue
        vehiculos = [espec_a_vehiculo(e, i, env) for i, e in enumerate(especs)]
        t0 = time.perf_counter()
        trays, motivos = planificar_flota(env, pl, vehiculos, opt, max_cand)
        for i, veh in enumerate(vehiculos):
            traj = trays.get(i)
            veh.traj = traj if traj is not None else []
            veh.mision_ok = traj is not None
            veh.dt_plan = DT
        # Clasificación por parámetros iniciales: carpeta por nº de vehículos,
        # fichero por (calidad, modo de optimización, worker).
        ruta_csv = os.path.join(
            salida, f"nveh_{n:02d}",
            f"dataset_c{calidad}_{opt}_w{os.getpid()}.csv")
        filas = guardar_dataset_supervisado(vehiculos, ruta_csv)
        filas_tot += filas
        ok = sum(1 for v in vehiculos if v.mision_ok)
        veh_ok += ok
        veh_tot += n
        print(f"[w{wid}] escenario {k + 1}/{len(semillas)} (semilla {sem}): "
              f"{ok}/{n} rutas, +{filas} filas, "
              f"{time.perf_counter() - t0:.1f} s", flush=True)
    return wid, filas_tot, veh_ok, veh_tot, saltados


def repartir(semillas, n_workers):
    return [semillas[i::n_workers] for i in range(n_workers)]


def main():
    ap = argparse.ArgumentParser(
        description="Genera rutas de entrenamiento sin interfaz gráfica.")
    ap.add_argument("--escenarios", type=int, default=100,
                    help="nº de escenarios aleatorios a resolver (def. 100)")
    ap.add_argument("--veh", default="2-6",
                    help="rango del nº de vehículos por escenario (def. 2-6)")
    ap.add_argument("--calidad", type=int, default=3, choices=range(1, 6),
                    help="calidad de ruta 1-5 (def. 3)")
    ap.add_argument("--opt", default="secuencial",
                    choices=("secuencial", "global", "prioridades"),
                    help="modo de optimización de flota (def. secuencial)")
    ap.add_argument("--max-ordenes", type=int, default=12,
                    help="órdenes candidatos máx. en global/prioridades (def. 12)")
    ap.add_argument("--workers", type=int,
                    default=max(1, (os.cpu_count() or 2) - 1),
                    help="procesos en paralelo (def. núcleos-1)")
    ap.add_argument("--semilla", type=int, default=0,
                    help="semilla base; escenario i usa semilla+i (def. 0)")
    ap.add_argument("--mapa", default=MAPA_ENTRENAMIENTO,
                    help="mapa JSON fijo (def. mapas/mapa_entrenamiento.json, "
                         "se crea determinista si no existe)")
    ap.add_argument("--dens", type=float, default=0.0,
                    help="densidad de obstáculos extra si hay que crear el mapa")
    ap.add_argument("--salida", default=RUTAS_DIR,
                    help="carpeta raíz de los CSV (def. rutas/)")
    args = ap.parse_args()

    veh_min, _, veh_max = args.veh.partition("-")
    veh_min = max(1, int(veh_min))
    veh_max = min(len(PALETA), int(veh_max or veh_min))

    ruta_mapa = asegurar_mapa(args.mapa, args.dens)
    semillas = list(range(args.semilla, args.semilla + args.escenarios))
    n_workers = max(1, min(args.workers, len(semillas)))

    print(f"[gen] {args.escenarios} escenarios · {veh_min}-{veh_max} vehículos "
          f"· calidad {args.calidad} · opt {args.opt} · {n_workers} workers "
          f"· mapa {os.path.basename(ruta_mapa)}")
    t0 = time.perf_counter()
    tandas = [(i, s, ruta_mapa, veh_min, veh_max, args.calidad, args.opt,
               args.max_ordenes, args.salida)
              for i, s in enumerate(repartir(semillas, n_workers)) if s]

    if n_workers == 1:
        resultados = [trabajar(tandas[0])]
    else:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
            resultados = list(ex.map(trabajar, tandas))

    filas = sum(r[1] for r in resultados)
    ok = sum(r[2] for r in resultados)
    tot = sum(r[3] for r in resultados)
    saltados = sum(r[4] for r in resultados)
    dur = time.perf_counter() - t0
    print(f"[gen] FIN: {filas:,} filas · {ok}/{tot} vehículos con ruta · "
          f"{saltados} escenarios sin sitio · {dur / 60:.1f} min "
          f"({dur / max(1, args.escenarios):.1f} s/escenario)")


if __name__ == "__main__":
    main()
