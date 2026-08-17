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
import collections
import itertools
import json
import math
import multiprocessing
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

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


def _leer_candidatos(ruta):
    """Candidatos ya evaluados de la tanda EN CURSO (checkpoint anti-corte de
    spot). Cada línea es [semilla, modo, orden, fallos, coste]. Devuelve
    (hechos, mejor): 'hechos' = conjunto de (semilla, modo, tuple(orden)) ya
    calculados para no repetirlos, y 'mejor' = mejor orden por (semilla, modo) ya
    encontrado. Así, si la máquina cae a media exploración de un escenario, al
    reanudar solo se recalcula lo que faltaba (minutos), no las 400 ordenaciones.
    """
    hechos, mejor = set(), {}
    if not os.path.exists(ruta):
        return hechos, mejor
    with open(ruta, encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if not linea:
                continue
            try:
                sem, modo, orden, fallos, coste = json.loads(linea)
            except (ValueError, TypeError):
                continue                       # línea a medias por un corte
            hechos.add((sem, modo, tuple(orden)))
            # tuple() OBLIGATORIO: json devuelve listas, pero los órdenes vivos
            # (generar_ordenes) son tuplas. Si no se normaliza aquí, en cuanto
            # una candidata viva empata en (fallos, coste) con una del
            # checkpoint, el desempate compara tupla con lista y revienta:
            # TypeError: '<' not supported between 'tuple' and 'list'.
            clave = (fallos, coste, tuple(orden))
            k = (sem, modo)
            if k not in mejor or clave < mejor[k]:
                mejor[k] = clave
    return hechos, mejor


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


# Fracción de las semillas SECUENCIALES que NO se generan: su plaza la cubre un
# orden rescatado de un caso de global/grupos, que sale gratis porque ya estaba
# calculado. Aquí está el ahorro de la técnica: no se paga por generar un
# secuencial que ya tenemos de otra parte.
#
# Se descarta sorteando SOLO entre las que ya salieron secuenciales, en vez de
# bajar el peso del secuencial en la mezcla. Es deliberado: la mezcla decide el
# modo de CADA semilla, así que cambiarla reasignaría el modo de semillas ya
# generadas y tiraría a la basura los casos de global/grupos ya pagados, que son
# los caros. Así el modo de cada semilla no se mueve y lo hecho sigue valiendo.
FRAC_SEC_DESCARTADAS = 0.4


def modos_de_semilla(semilla, opt, mezcla):
    """Modo a GENERAR para una semilla: un run por semilla, o NINGUNO si es una
    secuencial de las descartadas (ver FRAC_SEC_DESCARTADAS).

    Con opt != 'mixto': ese modo. Con 'mixto', la mezcla objetivo (secuencial,
    global, prioridades) se reparte por semilla con un RNG propio (no altera la
    flota, igual que `elegir_modo`).

    Devuelve pares (modo, etiqueta) por compatibilidad con quien lo consume;
    ahora ambos coinciden siempre.

    HUBO un sistema de GEMELOS ('secuencial_par'): una fracción de los
    global/prioridades se generaba también en secuencial con las mismas
    condiciones iniciales, para que la red viera la misma escena resuelta de las
    dos formas. Se ha desmantelado: los órdenes descartados que ahora se rescatan
    de cada caso de prioridades (ver _tarea_candidato) ya aportan de sobra
    situaciones secuenciales con condiciones iniciales de escenario
    global/prioridades, y encima salen gratis, mientras que el gemelo costaba un
    run entero. Los CSV de gemelos ya generados siguen siendo válidos y se
    siguen leyendo al entrenar."""
    if opt != "mixto":
        return [(opt, opt)]
    modos = ("secuencial", "global", "prioridades")
    m = random.Random(semilla * 2 + 1).choices(modos, weights=mezcla)[0]
    if m == "secuencial" and random.Random(semilla * 5 + 11).random() < FRAC_SEC_DESCARTADAS:
        return []            # esa plaza la cubre un orden rescatado, no se genera
    return [(m, m)]


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


def construir_escenario(semilla, tamanos, modo, env):
    """(modo, vehículos) de una semilla, o (modo, None) si la flota no cabe en el
    mapa. Es determinista, así que cualquier proceso reconstruye el mismo
    escenario a partir del número: por eso las tareas viajan como una semilla y
    un orden, y no como flotas enteras.

    El MODO entra como parámetro y NO interviene en el sorteo de la situación (la
    elección de modo va con un RNG propio, ver `modos_de_semilla`). Por eso la
    MISMA semilla da la MISMA flota en cualquier modo, lo que permite emparejar
    un escenario resuelto en 'secuencial' con el mismo en 'global'/'prioridades'.
    """
    random.seed(semilla)
    n = random.choice(tamanos)
    especs = aleatorios_spec(env, n)
    if especs is None:
        return modo, None
    grupos, prioridades = sortear_preferencias(n)
    for e, g, p in zip(especs, grupos, prioridades):
        e["grupo"], e["prioridad"] = g, p
    normal = parsear_especificaciones(repr(especs))
    return modo, [espec_a_vehiculo(e, i, env) for i, e in enumerate(normal)]


# Cuántos órdenes descartados se rescatan como secuenciales POR ESCENARIO.
# Varios salen a cuenta: ya están calculados y son gratis. El precio es que dos
# órdenes rescatados del mismo escenario comparten condiciones iniciales, así
# que en el PRIMER instante presentan a la red las mismas entradas con objetivos
# distintos. A partir de ahí las trayectorias divergen y el conflicto
# desaparece, o sea que afecta a una fracción pequeña de las muestras de cada
# run; y las alternativas en conflicto son simétricas entre sí (solo cambia el
# desempate entre vehículos de igual grupo y prioridad). Con todo, no conviene
# subirlo mucho: cuantos más se saquen del mismo sitio, menos variedad de
# situaciones por euro.
RESCATES_POR_ESC = 3

# Base de semillas para los GEMELOS (ver --gemelos). Muy por encima de cualquier
# rango de trabajo real, para que un gemelo no coincida nunca con un escenario
# del dataset ni se pise con otra tanda.
BASE_GEMELOS = 50_000_000


def reetiquetar_para_orden(vehiculos, orden):
    """Reescribe grupo y prioridad de la flota para que ordenar por (grupo,
    prioridad) dé EXACTAMENTE 'orden'. Modifica los vehículos en el sitio.

    Es lo que permite reaprovechar CUALQUIER orden descartado como run
    secuencial, incluidos los de modo 'global'. El problema que resuelve: la red
    recibe, por cada vecino, un signo de preferencia deducido de (grupo,
    prioridad). Un orden global puede adelantar a un vehículo de prioridad baja;
    guardado tal cual como secuencial, le estaría enseñando a la red a ignorar
    la preferencia justo en el modo que existe para respetarla. En vez de tirar
    ese orden, se cambian las etiquetas para que la situación SEA de verdad la
    que produce ese orden. Las trayectorias ya calculadas siguen siendo las
    correctas, porque planificar en secuencial esta flota reetiquetada da ese
    mismo orden.

    Se REPARTEN las etiquetas que ya tenía la flota (ordenadas) a lo largo de
    'orden', en vez de inventar una prioridad distinta por vehículo. Así la
    forma de la jerarquía —cuántos niveles hay y de qué tamaño— sigue siendo la
    que sortea `sortear_preferencias`, y los runs rescatados no se sesgan hacia
    flotas totalmente jerarquizadas. El precio es que, cuando hay empates, los
    vehículos empatados quedan interchangeables: ahí el orden guardado y el que
    daría una planificación secuencial pueden diferir. Es el caso benigno (son
    equivalentes por definición) y es el mismo que ya se acepta al rescatar
    varios órdenes del mismo escenario."""
    claves = sorted((v.grupo, v.prioridad) for v in vehiculos)
    for pos, i in enumerate(orden):
        vehiculos[i].grupo, vehiculos[i].prioridad = claves[pos]


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
    """Evalúa UN orden candidato. Normalmente su ruta se descarta: solo interesa
    cómo de bueno sale el orden, para poder elegir.

    Si 'rescatar' viene a True, la ruta NO se tira: se devuelve además como un
    run de entrenamiento etiquetado 'secuencial_rec'. La idea es que un orden
    descartado ya es, de hecho, una flota resuelta planificando de uno en uno —
    justo lo que la red ve como "secuencial"— y ya está calculada, así que
    tirarla es regalar datos. Sirve CUALQUIER orden y de cualquier modo: los
    grupos y prioridades que se guardan se reescriben para que encajen con el
    orden planificado (ver reetiquetar_para_orden)."""
    semilla, tamanos, modo, orden, rescatar = t
    t0 = time.perf_counter()
    _, vehiculos = construir_escenario(semilla, tamanos, modo, _ENV)
    _PL.deadline = None
    _PL.max_exp = _CAP_CALIDAD
    trays, _, coste = planificar_orden(_PL, vehiculos, orden, Reservas(),
                                       deadline_dur=None)
    fallos = sum(1 for i in orden if trays[i] is None)

    # Los que se quedaron sin ruta se reintentan con el techo alto, igual que
    # haría la pasada definitiva, para que lo que se guarde no esté por debajo
    # del estándar del resto. Solo hace falta si estas rutas se van a guardar.
    if fallos and rescatar:
        _PL.max_exp = CAP_COMPLETO
        reservas = Reservas()
        for i in orden:
            if trays[i] is not None:
                v = vehiculos[i]
                reservas.add(trays[i], v.length, v.width)
        for i in orden:
            if trays[i] is not None:
                continue
            veh = vehiculos[i]
            _PL.bloqueos = bloqueos_metas(vehiculos, orden, excepto=i,
                                          pose0=veh.inicio)
            traj = _PL.planificar(veh, reservas)
            if traj is not None and len(traj) >= 2:
                trays[i] = traj
                reservas.add(traj, veh.length, veh.width)
        _PL.max_exp = _CAP_CALIDAD

    # Las FILAS se construyen UNA sola vez y se comparten entre los dos usos: son
    # las mismas trayectorias, lo único que cambia entre un uso y otro son las
    # condiciones iniciales que las acompañan (el rescatado las lleva
    # reetiquetadas). Construirlas dos veces era trabajo tirado.
    definitiva = rescate = None
    if not fallos or rescatar:
        for i, veh in enumerate(vehiculos):
            traj = trays.get(i)
            veh.traj = traj if traj is not None else []
            veh.mision_ok = traj is not None
            veh.dt_plan = DT
        ok = [v for v in vehiculos if v.mision_ok and v.traj]
        filas = []
        for veh in ok:
            filas.extend(muestras_supervisadas(veh))
        if filas:
            # Si NINGÚN vehículo se quedó sin ruta, estas trayectorias son ya
            # las definitivas: replanificar el orden con `planificar_definitivo`
            # daría exactamente lo mismo. El techo más alto de esa pasada no
            # cambia nada porque la búsqueda termina mucho antes, por
            # estancamiento; medido, tarda lo mismo y devuelve rutas idénticas.
            # Así que si este orden acaba ganando, el padre las guarda sin
            # volver a planificar. Con fallos no se devuelven: ahí la pasada
            # definitiva sí aporta (reubica y reintenta a los que no llegaron).
            if not fallos:
                definitiva = (len(vehiculos), len(ok),
                              [veh_a_dict(v) for v in ok], filas,
                              time.perf_counter() - t0)
            if rescatar:
                # Se reetiqueta ANTES de volcar las condiciones del rescatado:
                # lo que se guarda tiene que ser la flota cuyo orden secuencial
                # es justo el que se planificó. Las de 'definitiva' ya están
                # tomadas arriba, con las etiquetas originales.
                reetiquetar_para_orden(vehiculos, orden)
                rescate = (len(vehiculos), len(ok),
                           [veh_a_dict(v) for v in ok], filas)
    return semilla, modo, orden, fallos, coste, rescate, definitiva


def _tarea_definitiva(t):
    """Planifica a tope el orden ganador y devuelve ya las filas del CSV."""
    semilla, tamanos, modo, orden = t
    t0 = time.perf_counter()
    _, vehiculos = construir_escenario(semilla, tamanos, modo, _ENV)
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


def generar(mias, args, tamanos, mezcla, idt):
    """Bucle principal de esta máquina. Devuelve (filas, veh_ok, veh, saltados)."""
    env = Entorno()
    cargar_mapa_en(env, args.mapa)
    asegurar(args.salida)
    prog_path = os.path.join(args.salida, f"progreso_t{idt:03d}.txt")
    cand_path = os.path.join(args.salida, f"candidatos_t{idt:03d}.jsonl")
    hechas = _leer_progreso(prog_path)          # claves "semilla:modo"
    # Cada semilla produce un run (ver modos_de_semilla); la unidad de
    # trabajo y de reanudación es (semilla, modo). 'etiqueta_de' es la etiqueta
    # con la que se GUARDA (hoy siempre igual al modo; se mantiene el mecanismo
    # porque los CSV ya generados con gemelos 'secuencial_par' siguen leyéndose).
    etiqueta_de = {}
    pendientes = []
    for sem in mias:
        for modo, etiqueta in modos_de_semilla(sem, args.opt, mezcla):
            etiqueta_de[(sem, modo)] = etiqueta
            if f"{sem}:{modo}" not in hechas:
                pendientes.append((sem, modo))
    if hechas:
        print(f"[gen] reanudando: {len(hechas)} hechas, "
              f"{len(pendientes)} pendientes", flush=True)
    if not pendientes:
        return 0, 0, 0, 0

    # GEMELOS: por cada unidad pendiente se preparan N escenarios EQUIVALENTES
    # (mismo nº de vehículos, mismo modo, misma calidad) pero con condiciones
    # iniciales distintas, sacados de semillas fuera del rango del trabajo. Los
    # dos compiten por la misma PLAZA y se queda la que termine primero; el resto
    # se descarta al vuelo.
    #
    # Para qué: al final quedan unos pocos escenarios patológicamente lentos que
    # acaparan un núcleo cada uno durante horas mientras el resto de la máquina
    # está parada. Como las condiciones iniciales son aleatorias de todos modos,
    # cambiar una situación imposible por otra equivalente no cambia en nada la
    # composición del dataset — mismo tamaño de flota y mismo modo— y evita
    # esperar indefinidamente. Los gemelos se exploran igual que cualquier otro
    # escenario (sus candidatas completas), así que no se rebaja la calidad.
    slot_de = {k: k for k in pendientes}       # unidad -> plaza que ocupa
    plazas_llenas = set()                      # plazas que ya tienen ganador
    gemelos_de = collections.Counter()         # intentos lanzados por plaza
    usadas_g = {s for s, _ in pendientes} | {int(k.split(":")[0])
                                             for k in hechas if ":" in k}
    if args.gemelos:
        for sem, modo in list(pendientes):
            _, vehs = construir_escenario(sem, tamanos, modo, env)
            if vehs is None:
                continue
            n_obj, s_alt, puestos = len(vehs), BASE_GEMELOS + sem * 1000, 0
            while puestos < args.gemelos and s_alt < BASE_GEMELOS + sem * 1000 + 5000:
                if s_alt not in usadas_g:
                    _, v2 = construir_escenario(s_alt, tamanos, modo, env)
                    if v2 is not None and len(v2) == n_obj:
                        pendientes.append((s_alt, modo))
                        etiqueta_de[(s_alt, modo)] = modo
                        slot_de[(s_alt, modo)] = (sem, modo)
                        usadas_g.add(s_alt)
                        gemelos_de[(sem, modo)] += 1
                        puestos += 1
                s_alt += 1
        print(f"[gen] {len(pendientes)} unidades en total "
              f"({args.gemelos} gemelos iniciales por plaza; se reponen solos "
              f"según se liberen núcleos)", flush=True)

    n_workers = max(1, min(args.workers, os.cpu_count() or 1))
    ctx = multiprocessing.get_context("spawn")
    filas_tot, veh_ok, veh_tot, saltados, hechos = 0, 0, 0, 0, 0
    t_ini = time.perf_counter()

    # Planificador CONTINUO: en vez de tandas con barrera (todas las órdenes
    # candidatas y LUEGO las definitivas), se mantiene una ventana de trabajo
    # siempre llena. Así, mientras unos pocos planes lentos (p. ej. un global de
    # 8 vehículos) siguen corriendo, ningún núcleo queda parado esperándolos: coge
    # ya la siguiente candidata o definitiva de cualquier otro escenario. Nada se
    # descarta ni se abarata: los lentos terminan igual, solo que no bloquean.
    hechos_cand, mejor = _leer_candidatos(cand_path)
    info = {}                              # (sem,modo) -> candidatas que faltan
    listos_def = collections.deque()       # unidades listas para planificar a tope
    mejor_def = {}                         # (sem,modo) -> rutas del mejor hasta ahora
    ahorradas = 0                          # planificaciones definitivas evitadas

    # Varios escenarios se mantienen ABIERTOS a la vez (hasta PROFUNDIDAD) y sus
    # candidatas se intercalan por turnos (round-robin): así un escenario caro
    # (p. ej. un global de 8 vehículos, con cientos de órdenes) NUNCA acapara la
    # ventana entera él solo, que era lo que volvía a bloquear a los baratos que
    # iban detrás pese al planificador continuo.
    PROFUNDIDAD = max(8, n_workers)
    # Tope de definitivas encoladas por adelantado. Sin él, 'asegurar_abiertos'
    # vaciaba el preparador ENTERO en cada arranque: la mayoría de unidades son
    # secuenciales (van a 'listos_def' y NO cuentan para PROFUNDIDAD), así que
    # el bucle seguía pidiendo hasta encontrar 32 escenarios con candidatas,
    # construyendo de golpe miles de escenarios que no tocaban aún.
    TOPE_DEF = 2 * n_workers
    abiertos = []                          # [(sem, modo, cola_de_órdenes_restantes)]
    # Candidatas ya evaluadas de cada escenario, arrastrando las del checkpoint:
    # al reanudar, el que ya iba adelantado NO debe volver a ir primero.
    hechas_de = collections.Counter((s, m) for s, m, _ in hechos_cand)
    prep_agotada = False
    rescatada = {}                         # (sem,modo) -> órdenes a guardar como sec.

    def mas_gemelos():
        """Va soltando gemelos NUEVOS para las plazas que sigan sin ganador,
        empezando por la que menos intentos lleve.

        Sin esto, los gemelos se creaban todos de golpe al principio: en cuanto
        una plaza se resolvía, sus competidores se descartaban y esos núcleos se
        quedaban parados hasta el final. Ahora, cada hueco que se libera se
        rellena con un intento nuevo de las plazas que aún resisten, así que la
        máquina no deja de empujar donde hace falta."""
        while True:
            libres = [p for p in slot_de.values() if p not in plazas_llenas]
            if not libres:
                return
            plaza = min(set(libres), key=lambda p: gemelos_de[p])
            sem0, modo = plaza
            _, vehs = construir_escenario(sem0, tamanos, modo, env)
            if vehs is None:
                plazas_llenas.add(plaza)          # no hay flota que valga: fuera
                continue
            n_obj = len(vehs)
            s_alt = BASE_GEMELOS + sem0 * 1000 + gemelos_de[plaza] + 1
            tope = BASE_GEMELOS + sem0 * 1000 + 5000
            puesto = None
            while s_alt < tope:
                if s_alt not in usadas_g:
                    _, v2 = construir_escenario(s_alt, tamanos, modo, env)
                    if v2 is not None and len(v2) == n_obj:
                        puesto = s_alt
                        break
                s_alt += 1
            gemelos_de[plaza] += 1
            if puesto is None:
                plazas_llenas.add(plaza)          # agotadas las semillas: fuera
                continue
            usadas_g.add(puesto)
            slot_de[(puesto, modo)] = plaza
            etiqueta_de[(puesto, modo)] = modo
            print(f"[gen] gemelo nuevo sem={puesto} para la plaza "
                  f"{sem0}:{modo} (intento {gemelos_de[plaza]})", flush=True)
            yield (puesto, modo)

    def preparador():
        unidades = itertools.chain(pendientes,
                                   mas_gemelos() if args.gemelos else ())
        for i_prep, (sem, modo) in enumerate(unidades):
            if slot_de.get((sem, modo), (sem, modo)) in plazas_llenas:
                continue                     # su plaza ya tiene ganador
            print(f"[gen] preparando #{i_prep} sem={sem} modo={modo}",
                  flush=True)
            _, vehiculos = construir_escenario(sem, tamanos, modo, env)
            if vehiculos is None:                     # la flota no cabía
                yield ("saltado", sem, modo, None)
                continue
            n = len(vehiculos)
            max_cand = ordenes_a_explorar(vehiculos, modo, args.fraccion_ordenes,
                                          args.curvatura_ordenes, args.tope_ordenes)
            ordenes = generar_ordenes(vehiculos, list(range(n)), modo, max_cand)
            print(f"[gen]   #{i_prep} {n} veh. · {len(ordenes) if ordenes else 0} "
                  f"órdenes", flush=True)
            k = (sem, modo)
            if ordenes and len(ordenes) > 1:
                # Una sola orden RESCATADA por escenario (ver _tarea_candidato),
                # elegida entre las que respetan grupo y prioridad. Se sortea de
                # forma determinista a partir de la semilla, no se coge la
                # primera ni la mejor: así los runs recuperados vienen de sitios
                # distintos y uniformes en vez de sesgarse hacia un tipo de
                # orden. Una sola por escenario evita además guardar dos
                # versiones del MISMO escenario, que se contradirían entre sí.
                rnd = random.Random(sem * 3 + 7)
                rescatada[k] = {tuple(o) for o in
                                rnd.sample(ordenes,
                                           min(RESCATES_POR_ESC, len(ordenes)))}
                rest = [o for o in ordenes
                        if (sem, modo, tuple(o)) not in hechos_cand]
                if rest and not args.cerrar_parciales:
                    info[k] = {"faltan": len(rest), "por_defecto": ordenes[0]}
                    yield ("abrir", sem, modo, collections.deque(rest))
                else:
                    # Ya está todo explorado, o se está CERRANDO EN FALSO: en ese
                    # segundo caso el escenario se cierra con la mejor candidata
                    # que haya en el checkpoint, sin evaluar las que faltaban.
                    # Sirve para cuando se acaba el presupuesto: sin esto, un
                    # escenario a medio explorar no escribe NADA y todo su
                    # cómputo se tira (el 2026-08-05 eran 714 candidatas ya
                    # pagadas repartidas en 27 escenarios).
                    orden = mejor[k][2] if k in mejor else ordenes[0]
                    yield ("def", sem, modo, orden)
            else:                                     # secuencial / 1 orden: directo
                yield ("def", sem, modo,
                       ordenes[0] if ordenes else list(range(n)))

    prep = preparador()

    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx,
                             initializer=_iniciar,
                             initargs=(args.mapa, args.calidad)) as ex, \
            open(prog_path, "a", encoding="utf-8") as fprog, \
            open(cand_path, "a", encoding="utf-8") as fcand:

        def asegurar_abiertos():
            """Abre escenarios nuevos hasta tener PROFUNDIDAD a la vez (o hasta
            agotar el trabajo), para que siempre haya varios entre los que
            rotar."""
            nonlocal prep_agotada, saltados
            while (len(abiertos) < PROFUNDIDAD and len(listos_def) < TOPE_DEF
                   and not prep_agotada):
                try:
                    tipo, sem, modo, extra = next(prep)
                except StopIteration:
                    prep_agotada = True
                    break
                if tipo == "saltado":
                    saltados += 1
                    fprog.write(f"{sem}:{modo}\n")
                    fprog.flush()
                elif tipo == "abrir":
                    abiertos.append((sem, modo, extra))
                else:                                  # "def"
                    listos_def.append((sem, modo, extra))

        def siguiente_candidata():
            """Una candidata del escenario abierto MENOS avanzado, o None si no
            queda ninguno.

            Se elige por mínimo de candidatas evaluadas, no por rotación simple.
            La rotación reparte los turnos por igual a partir del momento en que
            cada escenario se abre, pero los que entran tarde (al reponer hueco,
            o tras un reinicio) arrastran la desventaja para siempre. Medido el
            2026-08-05: entre los pendientes había uno con 120 candidatas hechas
            y siete con 1. Repartir por el que menos lleva los iguala, que es lo
            que hace falta si el presupuesto puede cortar a media exploración:
            así todos quedan explorados a una profundidad parecida y ninguno sale
            favorecido."""
            asegurar_abiertos()
            if not abiertos:
                return None
            i_min = min(range(len(abiertos)), key=lambda i: hechas_de[abiertos[i][0],
                                                                     abiertos[i][1]])
            sem, modo, cola = abiertos[i_min]
            orden = cola.popleft()
            hechas_de[(sem, modo)] += 1
            if not cola:
                del abiertos[i_min]                    # agotado: sale de la rueda
            return (sem, modo, orden)

        def anotar_definitiva(s, m, n, n_ok, condiciones, filas, dur):
            """Guarda un escenario resuelto y lleva la cuenta. Se llama tanto con
            el resultado de `_tarea_definitiva` como con las rutas que ya venían
            de la candidata ganadora (ver `mejor_def`), porque en ambos casos lo
            que hay que hacer es exactamente lo mismo."""
            nonlocal filas_tot, veh_ok, veh_tot, hechos
            plaza = slot_de.get((s, m), (s, m))
            if plaza in plazas_llenas:
                return                       # otro gemelo llegó antes: se tira
            plazas_llenas.add(plaza)
            if filas:
                # La etiqueta (no el modo de planificación) es lo que se guarda:
                # distingue 'secuencial_par' de la secuencial pura, para poder
                # darles peso distinto al entrenar.
                etq = etiqueta_de.get((s, m), m)
                ruta_csv = os.path.join(
                    args.salida, f"nveh_{n:02d}",
                    f"rutas_c{args.calidad}_{etq}_t{idt:03d}.csv")
                filas_tot += escribir_run(ruta_csv, s, etq, condiciones, filas)
            # El progreso se anota DESPUÉS de que el escenario esté en disco. Se
            # anota la PLAZA, no la unidad: si ganó un gemelo, lo que queda por
            # hecho es el escenario original, que es el que no hay que repetir.
            fprog.write(f"{plaza[0]}:{plaza[1]}\n")
            if plaza != (s, m):
                fprog.write(f"{s}:{m}\n")
            fprog.flush()
            veh_ok += n_ok
            veh_tot += n
            hechos += 1
            if hechos % args.cada == 0:
                t = time.perf_counter() - t_ini
                print(f"[gen] {hechos}/{len(pendientes)} · último: {n} "
                      f"veh. {m} {dur:.0f} s · {filas_tot:,} filas · "
                      f"{veh_ok}/{veh_tot} con ruta · media "
                      f"{t / hechos:.0f} s/run", flush=True)

        ventana = 2 * n_workers
        futs = {}
        while True:
            # 1) rellenar la ventana: en CADA vuelta se intentan las DOS cosas
            #    (una definitiva lista, si hay, y una candidata intercalada), en
            #    vez de agotar antes todas las definitivas. Con solo definitivas
            #    primero, mientras hubiera trabajo rápido en cola (lo habitual)
            #    nunca se llegaba a pedir una candidata nueva, y las lentas se
            #    quedaban paradas ratos largos en vez de avanzar poco a poco.
            while len(futs) < ventana:
                avanzo = False
                if listos_def:
                    sem, modo, orden = listos_def.popleft()
                    if slot_de.get((sem, modo), (sem, modo)) in plazas_llenas:
                        continue             # su plaza ya la ganó otro gemelo
                    f = ex.submit(_tarea_definitiva, (sem, tamanos, modo, orden))
                    futs[f] = ("def", sem, modo)
                    avanzo = True
                if len(futs) < ventana:
                    item = siguiente_candidata()
                    if item is not None:
                        sem, modo, orden = item
                        resc = tuple(orden) in rescatada.get((sem, modo), ())
                        f = ex.submit(_tarea_candidato,
                                      (sem, tamanos, modo, orden, resc))
                        futs[f] = ("cand", sem, modo)
                        avanzo = True
                if not avanzo:
                    if listos_def:
                        # 'siguiente_candidata' llama a 'asegurar_abiertos', que
                        # puede haber llenado la cola de definitivas DESPUÉS de
                        # que la mirásemos arriba. Sin este continue se salía del
                        # bucle con trabajo recién encolado y, al estar 'futs'
                        # vacío, se daba por terminado todo. Pasaba siempre con
                        # --cerrar-parciales, donde NINGÚN escenario se abre para
                        # explorar y por tanto todo va a parar a esta cola.
                        continue
                    break                              # nada más por ahora

            if not futs:
                break                                 # no queda nada por hacer

            # 2) esperar a que se libere algún núcleo y procesar lo terminado.
            for f in wait(futs, return_when=FIRST_COMPLETED).done:
                tipo, sem, modo = futs.pop(f)
                if tipo == "cand":
                    s, m, orden, fallos, coste, rescate, definitiva = f.result()
                    if rescate is not None:
                        # Orden descartado que se aprovecha como run secuencial:
                        # su ruta ya estaba calculada, así que sale gratis. Va
                        # con etiqueta propia para poder capar su proporción al
                        # entrenar, pero la red lo verá como 'secuencial'.
                        # Los varios rescates de un mismo escenario comparten
                        # run_id (la semilla) A PROPÓSITO: al entrenar, el
                        # reparto train/validación es por clave de run, así que
                        # así caen todos del mismo lado y no se valida con una
                        # variante del escenario que ya se ha entrenado.
                        n_r, ok_r, cond_r, filas_r = rescate
                        ruta_csv = os.path.join(
                            args.salida, f"nveh_{n_r:02d}",
                            f"rutas_c{args.calidad}_secuencial_rec_t{idt:03d}.csv")
                        filas_tot += escribir_run(ruta_csv, s, "secuencial_rec",
                                                  cond_r, filas_r)
                    # Se anota en el acto: si cae la máquina, no se repite.
                    fcand.write(json.dumps([s, m, orden, fallos, coste]) + "\n")
                    fcand.flush()
                    k = (s, m)
                    clave = (fallos, coste, orden)     # desempate por el orden
                    if k not in mejor or clave < mejor[k]:
                        mejor[k] = clave
                        # Se guardan las rutas del mejor HASTA AHORA y se suelta
                        # las del anterior: así, como mucho, se retiene un
                        # escenario por cada uno de los abiertos.
                        mejor_def[k] = definitiva
                    info[k]["faltan"] -= 1
                    if info[k]["faltan"] == 0:          # ya se puede elegir ganador
                        gana = mejor[k][2] if k in mejor else info[k]["por_defecto"]
                        lista = mejor_def.pop(k, None)
                        if lista is not None:
                            # El ganador ya venía planificado sin fallos: sus
                            # rutas son las definitivas y no hay que repetir la
                            # planificación entera (ver _tarea_candidato).
                            ahorradas += 1
                            anotar_definitiva(s, m, *lista)
                        else:
                            listos_def.append((s, m, gana))
                        del info[k]
                else:
                    anotar_definitiva(*f.result())
    if ahorradas:
        print(f"[gen] {ahorradas} planificaciones definitivas evitadas "
              f"(el ganador ya venía resuelto de la comparación)", flush=True)
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
    ap.add_argument("--gemelos", type=int, default=0,
                    help="escenarios EQUIVALENTES (mismo nº de vehículos y "
                         "modo, condiciones distintas) que compiten por cada "
                         "plaza pendiente; se queda el primero que termine. "
                         "Sirve para no dejar núcleos parados esperando a un "
                         "escenario patológicamente lento (def. 0)")
    ap.add_argument("--cerrar-parciales", dest="cerrar_parciales",
                    action="store_true",
                    help="no evalúa candidatas nuevas: cierra cada escenario a "
                         "medio explorar con la mejor que ya tenga en el "
                         "checkpoint. Para rematar cuando se acaba el "
                         "presupuesto, en vez de perder lo ya calculado")
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
