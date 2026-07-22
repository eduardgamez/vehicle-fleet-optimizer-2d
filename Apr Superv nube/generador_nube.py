#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FASE 1 (CPU) · Generación masiva de rutas de ENTRENAMIENTO.

Es el `generador.py` local con lo que hace falta para llenar de trabajo un
montón de máquinas alquiladas y sobrevivir a que corten alguna a media faena:

  · MISMO mapa fijo (mapas/mapa_entrenamiento.json) y CALIDAD 5.
  · Sin plazos de RELOJ: el único límite es el de nodos que ya define la calidad.
    Además de dar la mejor ruta que el planificador sabe encontrar, hace el
    dataset reproducible (con plazos, la ruta dependía de lo cargada que
    estuviera la máquina).
  · La unidad de trabajo es (escenario, orden candidato), no el escenario: así un
    caso de 12 vehículos con cientos de órdenes se reparte entre todos los
    núcleos en vez de ocupar uno solo durante horas.
  · SHARDING entre máquinas: cada tarea resuelve las semillas que le tocan y
    escribe sus propios CSV, sin coordinarse con las demás.
  · REANUDABLE: se anotan las semillas ya resueltas y lo hecho se sube al bucket
    cada pocos minutos, así que perder una máquina spot cuesta minutos, no horas.
    El identificador de cada run es la semilla, así que un escenario a medio
    escribir se descarta al preparar los datos en vez de colarse duplicado.

Aquí SOLO se generan rutas de entrenamiento: los escenarios con los que se
evalúa la red no necesitan que el planificador clásico los resuelva, porque lo
que se mide es si la red llega a la meta, no si copia una ruta concreta. Esos
salen (en segundos) de `escenarios.py`.

Ejemplos:
    python generador_nube.py --escenarios 10000 --workers 60
    python generador_nube.py --escenarios 10000 --tarea 3 --tareas 20
"""

import argparse
import math
import multiprocessing
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor

import comun
from comun import MAPA_ENTRENAMIENTO, RUTAS_DIR, asegurar, reparto

from nucleo import (
    CAP_COMPLETO, DT, Entorno, Planificador, Reservas, bloqueos_metas,
    cargar_mapa_en, warmup, espec_a_vehiculo, generar_ordenes,
    muestras_supervisadas, parsear_especificaciones, planificar_orden,
    total_combinaciones, veh_a_dict,
)
from politica import MODOS_OPT
from generador import aleatorios_spec, _reubicar

# Reparto de escenarios entre los tres modos de optimización: a partes iguales.
# La red recibe el modo como entrada, así que necesita ver los tres por igual
# para aprender a distinguirlos.
MEZCLA = (1 / 3, 1 / 3, 1 / 3)      # secuencial, global, prioridades

# Las semillas de entrenamiento arrancan en 0; las de evaluación viven en un
# rango altísimo (ver escenarios.py), así que no se pueden solapar por mucho que
# crezca el dataset.
SEMILLA_BASE = 0


# --------------------------------------------------------------------------- #
# Escritura del CSV (solo la hace el proceso padre)
# --------------------------------------------------------------------------- #
def escribir_run(ruta, run_id, opt, condiciones, filas):
    """Añade un escenario al CSV, con el mismo formato que
    `nucleo.guardar_dataset_supervisado` salvo en el identificador: aquí es la
    semilla y no la marca de tiempo, así que el mismo escenario produce siempre
    el mismo run y se puede deduplicar al reanudar."""
    if not filas:
        return 0
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    nuevo = not os.path.exists(ruta) or os.path.getsize(ruta) == 0
    with open(ruta, "a", encoding="utf-8") as f:
        if nuevo:
            f.write("run,vid,instante,x1,y1,x2,y2,v,theta,a,giro\n")
        f.write(f"# run={run_id} opt={opt} condiciones_iniciales={condiciones}\n")
        for vid, i, x1, y1, x2, y2, v, th, a, giro in filas:
            f.write(f"{run_id},{vid},{i},{x1:.6f},{y1:.6f},{x2:.6f},{y2:.6f},"
                    f"{v:.6f},{th:.6f},{a:.6f},{giro:.6f}\n")
        f.flush()
        os.fsync(f.fileno())
    return len(filas)


def _leer_progreso(ruta):
    if not os.path.exists(ruta):
        return set()
    with open(ruta, encoding="utf-8") as f:
        return {l.strip() for l in f if l.strip()}


# --------------------------------------------------------------------------- #
# Escenarios: de una semilla a una flota, y cuántos órdenes explorarle
# --------------------------------------------------------------------------- #
def elegir_modo(semilla, opt, mezcla=MEZCLA):
    """Modo de optimización de un escenario. Usa un generador de números PROPIO,
    sembrado aparte, para que cambiar el reparto entre modos no altere las flotas
    que salen de cada semilla."""
    if opt != "mixto":
        return opt
    return random.Random(semilla * 2 + 1).choices(MODOS_OPT, weights=mezcla)[0]


def sortear_preferencias(n):
    """Grupos y prioridades de una flota de n vehículos.

    El modo aleatorio de la aplicación sortea grupo en 1-2 y prioridad en 1-3,
    que es una estructura muy pobre: siempre pocos grupos y de tamaño parecido.
    Como los modos 'prioridades' y 'global' se juegan precisamente en cómo esté
    repartida la flota, aquí se sortea antes la ESTRUCTURA y luego el reparto:

      · cuántos grupos distintos hay (de uno solo a uno por vehículo) y cuántos
        niveles de prioridad,
      · con qué proporciones se reparten los vehículos entre ellos: los pesos van
        al cuadrado, así que salen tanto repartos equilibrados como flotas con un
        grupo enorme y varios de un vehículo.

    Así la red ve desde flotas sin preferencias hasta flotas totalmente
    jerarquizadas, y todo lo de en medio."""
    n_g = random.randint(1, n)
    n_p = random.randint(1, max(1, min(n, 5)))
    pesos_g = [random.random() ** 2 + 0.05 for _ in range(n_g)]
    pesos_p = [random.random() ** 2 + 0.05 for _ in range(n_p)]
    return (random.choices(range(1, n_g + 1), weights=pesos_g, k=n),
            random.choices(range(1, n_p + 1), weights=pesos_p, k=n))


def construir_escenario(semilla, tamanos, opt, mezcla, env):
    """(modo, vehículos) de una semilla, o (modo, None) si la flota no cabe en el
    mapa. Es determinista, así que cualquier proceso reconstruye el mismo
    escenario a partir del número: por eso las tareas viajan como una semilla y
    un orden, y no como flotas enteras."""
    random.seed(semilla)
    n = random.choice(tamanos)
    modo = elegir_modo(semilla, opt, mezcla)
    especs = aleatorios_spec(env, n)
    if especs is None:
        return modo, None
    grupos, prioridades = sortear_preferencias(n)
    for e, g, p in zip(especs, grupos, prioridades):
        e["grupo"], e["prioridad"] = g, p
    normal = parsear_especificaciones(repr(especs))
    return modo, [espec_a_vehiculo(e, i, env) for i, e in enumerate(normal)]


def ordenes_a_explorar(vehiculos, modo, fraccion, curvatura, tope):
    """Cuántos órdenes de planificación se comparan en este escenario:

        órdenes = fraccion · (1 + ln(posibles) / k) ^ k,   acotado por 'tope'

    Un número FIJO (los 12 del pipeline local) es absurdo en los dos extremos:
    con 3 vehículos son el doble de los que existen y con 6 son el 1,7 % de
    ellos. La referencia tiene que ser una PROPORCIÓN de los posibles.

    Pero esa proporción tampoco puede ser constante, porque los posibles crecen
    como un factorial (n! en 'global'; el producto de los factoriales de cada
    grupo en 'prioridades') y el 50 % de 15! no lo calcula nadie. Lo que se
    quiere es un porcentaje que RECORTE POCO al principio y que, cuando el
    problema ya sea inabarcable, decaiga casi al mismo ritmo al que crece el
    factorial (es decir, que el número de órdenes deje de crecer).

    Eso lo da esta curva: en el logaritmo de los posibles, el nº de órdenes
    empieza siguiéndolos casi de tú a tú y se va aplanando. La 'curvatura' k es
    lo que se tarda en aplanar: cuanto más baja, antes se estabiliza. Con k=1,5:

        vehículos:     3     4     5     6     8    10
        posibles:      6    24   120   720  40320  3,6M
        se exploran:   3     6     9    13    23    37
        porcentaje:  54%   23%    7%  1,8%  0,06%    —

    Comparado con una potencia fija, esto explora bastante más en las flotas
    pequeñas (donde mirarlo casi todo es barato) y bastante menos en las grandes
    (donde es lo que dispara el coste). 'secuencial' no entra: tiene un único
    orden determinista.

    El exponente se aplica a los DOS modos que exploran. Cuando los grupos son
    variados (ver `sortear_preferencias`), 'prioridades' puede tener tantos
    órdenes posibles como 'global' —basta con que casi toda la flota caiga en el
    mismo grupo—, así que tratarlo aparte no tendría sentido: lo que decide es
    cuántos posibles hay, no cómo se llame el modo.

    El tope es la última red de seguridad."""
    total = total_combinaciones(vehiculos, modo)
    k = max(0.1, curvatura)
    n = math.ceil(fraccion * (1.0 + math.log(max(1, total)) / k) ** k)
    # Con dos órdenes posibles conviene mirar los dos: son gratis.
    return max(min(total, 2), min(int(tope), n))


# --------------------------------------------------------------------------- #
# Planificación definitiva de un orden ya elegido
# --------------------------------------------------------------------------- #
def planificar_definitivo(env, pl, vehiculos, orden, cap_calidad):
    """Planifica el orden GANADOR con el presupuesto completo y rescata a los que
    se queden sin ruta.

    Es la segunda mitad de `generador.planificar_flota`, con una diferencia: allí
    solo se replanificaban los vehículos que habían fallado, así que los demás se
    guardaban con la ruta que hubiera salido de la fase de comparar órdenes. Como
    lo que se guarda aquí es el ejemplo que la red va a copiar, se replanifica
    entero."""
    base = Reservas()
    try:
        pl.deadline = None
        pl.max_exp = CAP_COMPLETO
        trays, motivos, _ = planificar_orden(pl, vehiculos, orden, base,
                                             deadline_dur=None)
        if any(trays[i] is None for i in orden):
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
                traj = pl.planificar(veh, reservas)
                motivos[i] = pl.motivo
                intentos = 0
                while pl.motivo == "sin_ruta" and intentos < 12:
                    if not _reubicar(env, veh, vehiculos):
                        break
                    pl.bloqueos = bloqueos_metas(vehiculos, orden, excepto=i,
                                                 pose0=veh.inicio)
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
        pl.max_exp = cap_calidad


# --------------------------------------------------------------------------- #
# Reparto del trabajo: la unidad NO es el escenario, es (escenario, orden)
# --------------------------------------------------------------------------- #
# Los órdenes candidatos de un escenario son independientes entre sí (cada uno
# planifica la flota entera desde cero contra las mismas reservas iniciales), así
# que se pueden repartir entre todos los núcleos. Con un proceso por escenario,
# un caso de 12 vehículos con cientos de órdenes ocupa UN núcleo durante horas y
# retrasa el final del trabajo aunque el resto de la máquina esté libre; troceado
# por órdenes, esos cientos de órdenes los devoran todos los núcleos a la vez.
#
# El trabajo va por TANDAS de escenarios, y cada tanda son dos pasadas:
#   1. todos los órdenes candidatos de todos los escenarios de la tanda, en un
#      único reparto plano (ahí está el grueso del cómputo);
#   2. el orden ganador de cada escenario, planificado a tope, uno por escenario.
# El proceso padre es el único que escribe; los workers solo devuelven filas.

_ENV = None            # estado por proceso, creado una sola vez al arrancar
_PL = None
_CAP_CALIDAD = None


def _iniciar(ruta_mapa, calidad):
    """Prepara el mapa y el planificador de un proceso del grupo. Se ejecuta una
    vez por proceso, no una por tarea: crear el entorno y compilar los kernels en
    cada orden candidato costaría más que planificarlo."""
    global _ENV, _PL, _CAP_CALIDAD
    _ENV = Entorno()
    cargar_mapa_en(_ENV, ruta_mapa)
    warmup()
    _PL = Planificador(_ENV)
    _PL.configurar_calidad(calidad)
    _CAP_CALIDAD = _PL.max_exp


def _tarea_candidato(t):
    """Evalúa UN orden candidato. Su ruta se descarta: solo interesa cómo de
    bueno sale el orden, para poder elegir."""
    semilla, tamanos, opt, mezcla, orden = t
    _, vehiculos = construir_escenario(semilla, tamanos, opt, mezcla, _ENV)
    _PL.deadline = None
    _PL.max_exp = _CAP_CALIDAD
    trays, _, coste = planificar_orden(_PL, vehiculos, orden, Reservas(),
                                       deadline_dur=None)
    fallos = sum(1 for i in orden if trays[i] is None)
    return semilla, orden, fallos, coste


def _tarea_definitiva(t):
    """Planifica a tope el orden ganador y devuelve ya las filas del CSV."""
    semilla, tamanos, opt, mezcla, orden = t
    t0 = time.perf_counter()
    modo, vehiculos = construir_escenario(semilla, tamanos, opt, mezcla, _ENV)
    trays, _ = planificar_definitivo(_ENV, _PL, vehiculos, orden, _CAP_CALIDAD)
    for i, veh in enumerate(vehiculos):
        traj = trays.get(i)
        veh.traj = traj if traj is not None else []
        veh.mision_ok = traj is not None
        veh.dt_plan = DT
    ok = [v for v in vehiculos if v.mision_ok and v.traj]
    filas = []
    for veh in ok:
        filas.extend(muestras_supervisadas(veh))
    return (semilla, modo, len(vehiculos), len(ok),
            [veh_a_dict(v) for v in ok], filas, time.perf_counter() - t0)


def _tandas(pendientes, tamanos, args, mezcla, env, n_workers):
    """Agrupa escenarios en tandas con órdenes de sobra para llenar todos los
    núcleos, sin que una tanda se alargue tanto que perder la máquina duela."""
    tanda, ordenes_tanda = [], 0
    for sem in pendientes:
        modo, vehiculos = construir_escenario(sem, tamanos, args.opt, mezcla, env)
        if vehiculos is None:                    # no cabía la flota en el mapa
            tanda.append((sem, modo, None, 0))
            continue
        max_cand = ordenes_a_explorar(vehiculos, modo, args.fraccion_ordenes,
                                      args.curvatura_ordenes, args.tope_ordenes)
        ordenes = generar_ordenes(vehiculos, list(range(len(vehiculos))), modo,
                                  max_cand)
        tanda.append((sem, modo, ordenes, len(vehiculos)))
        ordenes_tanda += len(ordenes)
        if ordenes_tanda >= 8 * n_workers or len(tanda) >= 4 * n_workers:
            yield tanda
            tanda, ordenes_tanda = [], 0
    if tanda:
        yield tanda


def generar(mias, args, tamanos, mezcla, idt):
    """Bucle principal de esta máquina. Devuelve (filas, veh_ok, veh, saltados)."""
    env = Entorno()
    cargar_mapa_en(env, args.mapa)
    asegurar(args.salida)
    prog_path = os.path.join(args.salida, f"progreso_t{idt:03d}.txt")
    hechas = _leer_progreso(prog_path)
    pendientes = [s for s in mias if str(s) not in hechas]
    if hechas:
        print(f"[gen] reanudando: {len(mias) - len(pendientes)} hechas, "
              f"{len(pendientes)} pendientes", flush=True)
    if not pendientes:
        return 0, 0, 0, 0

    n_workers = max(1, min(args.workers, os.cpu_count() or 1))
    ctx = multiprocessing.get_context("spawn")
    filas_tot, veh_ok, veh_tot, saltados, hechos = 0, 0, 0, 0, 0
    t_ini = time.perf_counter()

    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx,
                             initializer=_iniciar,
                             initargs=(args.mapa, args.calidad)) as ex, \
            open(prog_path, "a", encoding="utf-8") as fprog:
        for tanda in _tandas(pendientes, tamanos, args, mezcla, env, n_workers):
            # -- pasada 1: todos los órdenes candidatos de la tanda, en plano --
            tareas = [(sem, tamanos, args.opt, mezcla, orden)
                      for sem, _, ordenes, _ in tanda
                      if ordenes and len(ordenes) > 1
                      for orden in ordenes]
            mejor = {}
            for sem, orden, fallos, coste in ex.map(_tarea_candidato, tareas,
                                                    chunksize=1):
                # Se desempata por el propio orden: los resultados llegan en
                # cualquier secuencia y la elección tiene que salir igual
                # ejecute donde ejecute.
                clave = (fallos, coste, orden)
                if sem not in mejor or clave < mejor[sem]:
                    mejor[sem] = clave

            # -- pasada 2: el orden ganador de cada escenario, a tope ----------
            definitivas = []
            for sem, modo, ordenes, _ in tanda:
                if ordenes is None:
                    saltados += 1
                    fprog.write(f"{sem}\n")
                    continue
                orden = mejor[sem][2] if sem in mejor else ordenes[0]
                definitivas.append((sem, tamanos, args.opt, mezcla, orden))
            fprog.flush()

            for res in ex.map(_tarea_definitiva, definitivas, chunksize=1):
                sem, modo, n, n_ok, condiciones, filas, dur = res
                if filas:
                    ruta_csv = os.path.join(
                        args.salida, f"nveh_{n:02d}",
                        f"rutas_c{args.calidad}_{modo}_t{idt:03d}.csv")
                    filas_tot += escribir_run(ruta_csv, sem, modo, condiciones,
                                              filas)
                # El progreso se anota DESPUÉS de que el escenario esté en disco.
                fprog.write(f"{sem}\n")
                fprog.flush()
                veh_ok += n_ok
                veh_tot += n
                hechos += 1
                if hechos % args.cada == 0 or hechos == len(pendientes):
                    t = time.perf_counter() - t_ini
                    print(f"[gen] {hechos}/{len(pendientes)} · último: {n} veh. "
                          f"{modo} {dur:.0f} s · {filas_tot:,} filas · "
                          f"{veh_ok}/{veh_tot} con ruta · media "
                          f"{t / hechos:.0f} s/escenario · faltan "
                          f"~{t / hechos * (len(pendientes) - hechos) / 60:.0f} "
                          f"min", flush=True)
    return filas_tot, veh_ok, veh_tot, saltados


# --------------------------------------------------------------------------- #
def parsear_veh(spec):
    """Tamaños de flota permitidos, a partir de algo como «1-6,8,10,12,15».
    Se sortea uno por escenario, con igual probabilidad cada tamaño de la lista
    (repetir un valor le da más peso)."""
    tam = []
    for trozo in str(spec).split(","):
        trozo = trozo.strip()
        if not trozo:
            continue
        if "-" in trozo:
            a, _, b = trozo.partition("-")
            tam.extend(range(int(a), int(b) + 1))
        else:
            tam.append(int(trozo))
    tam = [t for t in tam if t >= 1]
    if not tam:
        raise SystemExit(f"--veh no válido: {spec!r}")
    return tam


def main():
    ap = argparse.ArgumentParser(
        description="Genera rutas de entrenamiento en paralelo (fase 1, CPU).")
    ap.add_argument("--escenarios", type=int, default=1000,
                    help="escenarios de TODO el trabajo (se reparten entre tareas)")
    ap.add_argument("--veh", default="1-6,8",
                    help="tamaños de flota, como «1-6,8,10»: se sortea uno por "
                         "escenario y repetir un valor le da más peso "
                         "(def. 1-8,10)")
    ap.add_argument("--calidad", type=int, default=5, choices=range(1, 6),
                    help="calidad de ruta (def. 5)")
    ap.add_argument("--opt", default="mixto",
                    choices=("secuencial", "global", "prioridades", "mixto"),
                    help="modo de optimización; 'mixto' mezcla los tres, que es "
                         "lo que necesita la red para aprender a distinguirlos "
                         "(def. mixto)")
    ap.add_argument("--fraccion-ordenes", dest="fraccion_ordenes", type=float,
                    default=1.0,
                    help="factor de los órdenes explorados (def. 0.5)")
    ap.add_argument("--curvatura-ordenes", dest="curvatura_ordenes", type=float,
                    default=4.0,
                    help="cuánto tarda en aplanarse el nº de órdenes explorados "
                         "según crecen los posibles. Alta = recorta poco durante "
                         "más tiempo pero se dispara en flotas grandes; baja = "
                         "se estabiliza antes (def. 1.5)")
    ap.add_argument("--tope-ordenes", dest="tope_ordenes", type=int, default=400,
                    help="tope duro de órdenes por escenario (def. 400)")
    ap.add_argument("--mezcla", default=",".join(str(p) for p in MEZCLA),
                    help="reparto de escenarios entre secuencial, global y "
                         f"prioridades (def. {MEZCLA})")
    ap.add_argument("--workers", type=int,
                    default=max(1, (os.cpu_count() or 2) - 1),
                    help="procesos por máquina (def. núcleos-1)")
    ap.add_argument("--tarea", type=int, default=None,
                    help="índice de esta máquina (def. el del orquestador)")
    ap.add_argument("--tareas", type=int, default=None,
                    help="nº total de máquinas (def. el del orquestador)")
    ap.add_argument("--semilla", type=int, default=SEMILLA_BASE)
    ap.add_argument("--mapa", default=MAPA_ENTRENAMIENTO)
    ap.add_argument("--salida", default=RUTAS_DIR)
    ap.add_argument("--cada", type=int, default=10,
                    help="cada cuántos escenarios se escribe una línea de log")
    ap.add_argument("--sync-min", dest="sync_min", type=int, default=10,
                    help="cada cuántos minutos se sube lo hecho al bucket "
                         "(solo si hay TDR_BUCKET; def. 10)")
    args = ap.parse_args()

    tamanos = parsear_veh(args.veh)
    if not os.path.exists(args.mapa):
        raise SystemExit(f"No existe el mapa {args.mapa}.")

    todas = list(range(args.semilla, args.semilla + args.escenarios))
    mias = reparto(todas, args.tarea, args.tareas)
    asegurar(args.salida)

    # Con máquinas spot esto es lo que evita repetir horas de cómputo: al
    # arrancar se recupera lo ya subido y durante la ejecución se sube cada pocos
    # minutos, así una interrupción cuesta minutos y no la tarea entera.
    remoto = comun.bucket_env("rutas")
    comun.recuperar(args.salida, remoto)
    comun.arrancar_sincronizacion(args.salida, remoto, args.sync_min)

    mezcla = tuple(float(p) for p in args.mezcla.split(","))
    idt = args.tarea if args.tarea is not None else comun.indice_tarea()
    print(f"[gen] tarea {idt}/{args.tareas or comun.total_tareas()} · "
          f"{len(mias)} escenarios de {args.escenarios} · flotas de "
          f"{min(tamanos)}-{max(tamanos)} veh. · calidad {args.calidad} · opt "
          f"{args.opt} (mezcla {mezcla}) · órdenes = "
          f"{args.fraccion_ordenes}·(1+ln(posibles)/{args.curvatura_ordenes})"
          f"^{args.curvatura_ordenes} "
          f"(tope {args.tope_ordenes}) · {args.workers} procesos", flush=True)

    t0 = time.perf_counter()
    filas, ok, tot, saltados = generar(mias, args, tamanos, mezcla, idt)
    dur = time.perf_counter() - t0
    print(f"[gen] FIN: {filas:,} filas · {ok}/{tot} vehículos con ruta · "
          f"{saltados} escenarios sin sitio · {dur / 60:.1f} min", flush=True)
    if remoto:
        comun.sincronizar(args.salida, remoto)


if __name__ == "__main__":
    main()
