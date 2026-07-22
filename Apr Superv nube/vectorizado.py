#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Construcción VECTORIZADA (numpy) del vector de entrada de la red.

Es la misma definición de `politica.vector_entrada`, calculada para miles de
filas a la vez en lugar de una por una en Python. `verificar.py` comprueba que
ambas coinciden fila a fila; si se toca `politica.vector_entrada`, hay que
reflejar el cambio aquí y volver a pasar esa verificación.

Además introduce el SUPERSET de representación, que es la optimización clave de
la fase 2 en la nube:

  El barrido explora tres ejes que cambian la ENTRADA (nº de vecinos, historia
  de controles y horizonte del filtro de vecinos), y hasta ahora cada valor
  obligaba a reconstruir todas las muestras desde el CSV. Pero, por cómo está
  definida la entrada, la representación pequeña es un RECORTE EXACTO de la
  grande:

    · los vecinos se ordenan por tiempo de cierre y se rellenan por orden, así
      que quedarse con los n primeros bloques ES tener n_vecinos = n;
    · la historia va alineada a la derecha, así que quedarse con las 2·h últimas
      columnas del bloque ES tener h_pasado = h;
    · bajar el horizonte solo descarta vecinos con tiempo de cierre mayor, que
      por el orden anterior son siempre los ÚLTIMOS bloques ocupados: basta
      poner a cero los que superen el umbral (por eso se guarda el tiempo de
      cierre de cada bloque junto a las muestras).

  Así se construyen las muestras UNA sola vez, con los valores máximos, y cada
  una de las combinaciones del barrido se obtiene con un recorte de columnas y
  una máscara, sin volver a tocar el disco ni la CPU.
"""

import math

import numpy as np

import politica as pol
from nucleo import DT

# Dimensiones de los bloques del vector: son las del pipeline local y no cambian
# con la representación (lo que cambia es cuántos bloques hay).
DIM_EGO = pol.DIM_EGO
DIM_META = pol.DIM_META
DIM_VECINO = pol.DIM_VECINO
DIM_CTX = pol.DIM_CTX
MODOS_OPT = pol.MODOS_OPT


def ang_norm(a):
    """Versión vectorizada de nucleo.ang_norm (rango (-π, π])."""
    return np.arctan2(np.sin(a), np.cos(a))


# --------------------------------------------------------------------------- #
# Trazado del superset: qué columnas ocupa cada bloque y cuáles usa cada config
# --------------------------------------------------------------------------- #
def dim_super(n_vec_max, h_max):
    return (DIM_EGO + DIM_META + 2 * h_max + DIM_VECINO * n_vec_max + DIM_CTX)


def columnas_vista(n_vec, h_pasado, n_vec_max, h_max):
    """Índices de las columnas del superset que forman la entrada de una
    representación (n_vec, h_pasado). El resultado tiene exactamente la longitud
    y el orden que espera `politica.crear_red` para esa configuración."""
    if n_vec > n_vec_max or h_pasado > h_max:
        raise ValueError(f"la vista ({n_vec}, {h_pasado}) no cabe en el superset "
                         f"({n_vec_max}, {h_max})")
    cols = list(range(DIM_EGO + DIM_META))                # ego + meta
    base_pas = DIM_EGO + DIM_META
    # Historia alineada a la DERECHA: las 2·h_pasado últimas columnas del bloque.
    cols += list(range(base_pas + 2 * (h_max - h_pasado), base_pas + 2 * h_max))
    base_vec = base_pas + 2 * h_max
    cols += list(range(base_vec, base_vec + DIM_VECINO * n_vec))
    base_ctx = base_vec + DIM_VECINO * n_vec_max
    cols += list(range(base_ctx, base_ctx + DIM_CTX))
    return np.asarray(cols, dtype=np.int64)


def mascara_horizonte(tc, horizonte):
    """Bloques de vecino que SOBREVIVEN a un horizonte dado, a partir del tiempo
    de cierre guardado por bloque. tc: (M, n_vec_max) → (M, n_vec_max) booleana."""
    return tc <= horizonte * DT


# --------------------------------------------------------------------------- #
# Bloques del vector de entrada
# --------------------------------------------------------------------------- #
def bloque_ego_meta(x, y, th, v, largo, ancho, v_max, a_max, giro_max,
                    meta_x, meta_y, meta_th):
    """(M, DIM_EGO + DIM_META). 'meta_th' con NaN donde el ángulo es libre."""
    M = len(x)
    c, s = np.cos(th), np.sin(th)
    out = np.empty((M, DIM_EGO + DIM_META), dtype=np.float64)
    out[:, 0] = x
    out[:, 1] = y
    out[:, 2] = s
    out[:, 3] = c
    out[:, 4] = v
    out[:, 5] = largo
    out[:, 6] = ancho
    out[:, 7] = v_max
    out[:, 8] = a_max
    out[:, 9] = giro_max

    dx, dy = meta_x - x, meta_y - y
    libre = ~np.isfinite(meta_th)
    dth = np.where(libre, 0.0, ang_norm(np.where(libre, 0.0, meta_th) - th))
    out[:, DIM_EGO + 0] = dx * c + dy * s
    out[:, DIM_EGO + 1] = -dx * s + dy * c
    out[:, DIM_EGO + 2] = np.hypot(dx, dy)
    out[:, DIM_EGO + 3] = np.sin(dth)
    out[:, DIM_EGO + 4] = np.cos(dth)
    out[:, DIM_EGO + 5] = libre.astype(np.float64)
    return out


def bloque_vecinos(x, y, th, largo_e, ancho_e, v_max_e, grupo_e, prio_e,
                   ox, oy, oth, ov, ovmax, olargo, oancho, ogrupo, oprio,
                   valido, n_vec, horizonte):
    """(M, DIM_VECINO·n_vec) y (M, n_vec) con el tiempo de cierre de cada bloque
    (inf donde no hay vecino). Los arrays 'o*' son (M, K) y deben ser FINITOS
    también en las posiciones inválidas (basta rellenarlas con ceros)."""
    M = len(x)
    if valido.shape[1] == 0:
        return (np.zeros((M, DIM_VECINO * n_vec)),
                np.full((M, n_vec), np.inf))

    dx, dy = ox - x[:, None], oy - y[:, None]
    dist = np.hypot(dx, dy)
    radio_e = 0.5 * np.hypot(largo_e, ancho_e)
    radio_o = 0.5 * np.hypot(olargo, oancho)
    holgura = np.maximum(0.0, dist - radio_e[:, None] - radio_o)
    v_cierre = v_max_e[:, None] + ovmax
    t_cierre = np.where(v_cierre > 1e-6, holgura / np.where(v_cierre > 1e-6,
                                                           v_cierre, 1.0), 0.0)
    # Clave de orden: los no relevantes (o inexistentes) van al final con inf.
    clave = np.where(valido & (t_cierre <= horizonte * DT), t_cierre, np.inf)
    # Orden ESTABLE: con tiempos de cierre iguales conserva el orden de la lista
    # de compañeros, igual que el sort de Python del pipeline local.
    K = clave.shape[1]
    orden = np.argsort(clave, axis=1, kind="stable")[:, :n_vec]
    if orden.shape[1] < n_vec:      # menos compañeros que ranuras
        relleno = np.zeros((M, n_vec - orden.shape[1]), dtype=orden.dtype)
        orden = np.concatenate([orden, relleno], axis=1)
    # Ranuras que ni siquiera tienen un compañero al que apuntar (flota pequeña).
    hay_ranura = np.arange(n_vec)[None, :] < K

    def tomar(a):
        return np.take_along_axis(np.ascontiguousarray(a), orden, axis=1)

    tc = np.where(hay_ranura, tomar(clave), np.inf)
    presente = np.isfinite(tc)

    sdx, sdy, sdist = tomar(dx), tomar(dy), tomar(dist)
    sth, sv = tomar(oth), tomar(ov)
    slargo, sancho = tomar(olargo), tomar(oancho)
    sgrupo, sprio = tomar(ogrupo), tomar(oprio)

    c, s = np.cos(th)[:, None], np.sin(th)[:, None]
    co, so = np.cos(sth), np.sin(sth)
    hx = 0.5 * slargo * co - 0.5 * sancho * so
    hy = 0.5 * slargo * so + 0.5 * sancho * co
    p1x, p1y = sdx + hx, sdy + hy
    p2x, p2y = sdx - hx, sdy - hy
    dth = ang_norm(sth - th[:, None])
    # Preferencia: menor (grupo, prioridad) → planifica antes → tiene preferencia.
    dg = sgrupo - grupo_e[:, None]
    dp = sprio - prio_e[:, None]
    pref = -np.where(dg != 0, np.sign(dg), np.sign(dp))

    campos = np.stack([p1x * c + p1y * s, -p1x * s + p1y * c,
                       p2x * c + p2y * s, -p2x * s + p2y * c,
                       sdist, np.sin(dth), np.cos(dth), sv,
                       pref, np.ones_like(sdist)], axis=2)
    campos *= presente[:, :, None]
    return campos.reshape(M, DIM_VECINO * n_vec), tc


def bloque_ctx(opt, n_veh, M):
    """(M, DIM_CTX): one-hot del modo de optimización + nº de vehículos saturado.
    'n_veh' puede ser escalar o array (M,)."""
    out = np.zeros((M, DIM_CTX))
    if opt in MODOS_OPT:
        out[:, MODOS_OPT.index(opt)] = 1.0
    out[:, len(MODOS_OPT)] = 1.0 - 1.0 / np.maximum(1, n_veh)
    return out


# --------------------------------------------------------------------------- #
# Muestras de un run completo (todas las filas del CSV de un escenario)
# --------------------------------------------------------------------------- #
def superset_run(conds, opt, datos, n_vec_max, h_max, hor_max):
    """Muestras de un escenario en la representación MÁXIMA.

    'conds' y 'datos' son los que devuelve `entrenar.leer_runs`. Devuelve
    (X (M, dim_super), TC (M, n_vec_max), Y (M, 2·N_PRED)), con una fila por
    vehículo e instante, en el mismo orden que `entrenar.construir_muestras`."""
    vids = [vid for vid in datos if vid in conds]
    V = len(vids)
    if V == 0:
        d = dim_super(n_vec_max, h_max)
        return (np.zeros((0, d), np.float32), np.zeros((0, n_vec_max), np.float32),
                np.zeros((0, 2 * pol.N_PRED), np.float32))

    largos = [len(datos[v]) for v in vids]
    T = max(largos)
    # Estado (x, y, θ, v) y control (a, giro) de cada vehículo, alargados hasta T:
    # más allá de su última fila un vehículo se queda APARCADO en su pose final
    # con velocidad 0, que es lo que ven sus compañeros que aún circulan.
    pose = np.zeros((V, T, 4))
    ctrl = np.zeros((V, T, 2))
    for k, vid in enumerate(vids):
        arr = datos[vid]
        n = len(arr)
        pose[k, :n] = arr[:, :4]
        ctrl[k, :n] = arr[:, 4:6]
        if n < T:
            pose[k, n:] = arr[-1, :4]
            pose[k, n:, 3] = 0.0

    par = {c: np.array([conds[v][c] for v in vids], dtype=np.float64)
           for c in ("largo", "ancho", "v_max", "a_max", "giro_max",
                     "grupo", "prioridad")}
    meta = np.array([conds[v]["meta"] for v in vids], dtype=np.float64)
    meta_th = np.array([np.nan if conds[v]["meta_th"] is None
                        else conds[v]["meta_th"] for v in vids])

    Xs, TCs, Ys = [], [], []
    for k, vid in enumerate(vids):
        n = largos[k]
        t = np.arange(n)
        x, y, th, v = pose[k, :n, 0], pose[k, :n, 1], pose[k, :n, 2], pose[k, :n, 3]
        uno = np.ones(n)

        em = bloque_ego_meta(
            x, y, th, v, par["largo"][k] * uno, par["ancho"][k] * uno,
            par["v_max"][k] * uno, par["a_max"][k] * uno, par["giro_max"][k] * uno,
            meta[k, 0] * uno, meta[k, 1] * uno, meta_th[k] * uno)

        # Historia: la ranura m guarda el control del instante t − h_max + m
        # (ceros donde aún no había historia).
        idx = t[:, None] - h_max + np.arange(h_max)[None, :]
        hay = idx >= 0
        idxc = np.clip(idx, 0, n - 1)
        pas = np.zeros((n, h_max, 2))
        pas[:, :, 0] = np.where(hay, ctrl[k, idxc, 0] / max(par["a_max"][k], 1e-6), 0.0)
        pas[:, :, 1] = np.where(hay, ctrl[k, idxc, 1] / max(par["giro_max"][k], 1e-6), 0.0)
        pas = pas.reshape(n, 2 * h_max)

        otros = [j for j in range(V) if j != k]
        if otros:
            oi = np.array(otros)
            vec, tc = bloque_vecinos(
                x, y, th, par["largo"][k] * uno, par["ancho"][k] * uno,
                par["v_max"][k] * uno, par["grupo"][k] * uno, par["prioridad"][k] * uno,
                pose[oi, :n, 0].T, pose[oi, :n, 1].T, pose[oi, :n, 2].T,
                pose[oi, :n, 3].T,
                np.broadcast_to(par["v_max"][oi], (n, len(oi))),
                np.broadcast_to(par["largo"][oi], (n, len(oi))),
                np.broadcast_to(par["ancho"][oi], (n, len(oi))),
                np.broadcast_to(par["grupo"][oi], (n, len(oi))),
                np.broadcast_to(par["prioridad"][oi], (n, len(oi))),
                np.ones((n, len(oi)), dtype=bool), n_vec_max, hor_max)
        else:
            vec = np.zeros((n, DIM_VECINO * n_vec_max))
            tc = np.full((n, n_vec_max), np.inf)

        ctx = bloque_ctx(opt, V, n)
        Xs.append(np.concatenate([em, pas, vec, ctx], axis=1))
        TCs.append(tc)

        # Objetivo: los N_PRED controles siguientes, normalizados; pasado el final
        # de la trayectoria el vehículo está aparcado y el objetivo es (0, 0).
        Y = np.zeros((n, 2 * pol.N_PRED))
        for j in range(pol.N_PRED):
            fut = t + j
            dentro = fut < n
            f = np.clip(fut, 0, n - 1)
            Y[:, 2 * j] = np.where(dentro, ctrl[k, f, 0] / max(par["a_max"][k], 1e-6), 0.0)
            Y[:, 2 * j + 1] = np.where(dentro, ctrl[k, f, 1] / max(par["giro_max"][k], 1e-6), 0.0)
        Ys.append(Y)

    X = np.concatenate(Xs).astype(np.float32)
    TC = np.concatenate(TCs)
    # El tiempo de cierre solo se usa para comparar con umbrales pequeños: el
    # infinito de las ranuras vacías se satura para poder guardarlo en float32.
    TC = np.minimum(TC, 1e6).astype(np.float32)
    Y = np.concatenate(Ys).astype(np.float32)
    return X, TC, Y


# --------------------------------------------------------------------------- #
# Entradas de un INSTANTE para varias flotas a la vez (despliegue / rollout)
# --------------------------------------------------------------------------- #
def entradas_instante(est, par, idx_otros, valido, ctx, n_vec_max, hor_max,
                      pasado):
    """Vector de entrada de TODOS los vehículos (M) en un instante.

    'est' (M, 4) es (x, y, θ, v); 'par' un dict de arrays (M,) con los
    parámetros; 'idx_otros' (M, K) los índices de los compañeros de flota y
    'valido' (M, K) qué posiciones lo son; 'ctx' (M, DIM_CTX) el contexto de
    escenario, ya calculado (es constante durante todo el rollout); 'pasado'
    (M, h_max, 2) la historia de controles ya normalizada."""
    x, y, th, v = est[:, 0], est[:, 1], est[:, 2], est[:, 3]
    em = bloque_ego_meta(x, y, th, v, par["largo"], par["ancho"], par["v_max"],
                         par["a_max"], par["giro_max"], par["meta_x"],
                         par["meta_y"], par["meta_th"])
    pas = pasado.reshape(len(x), -1)

    def de_otros(a):
        return np.where(valido, a[idx_otros], 0.0)

    vec, tc = bloque_vecinos(
        x, y, th, par["largo"], par["ancho"], par["v_max"], par["grupo"],
        par["prioridad"],
        de_otros(x), de_otros(y), de_otros(th), de_otros(v),
        de_otros(par["v_max"]), de_otros(par["largo"]), de_otros(par["ancho"]),
        de_otros(par["grupo"]), de_otros(par["prioridad"]),
        valido, n_vec_max, hor_max)
    return np.concatenate([em, pas, vec, ctx], axis=1).astype(np.float32), tc


# --------------------------------------------------------------------------- #
# Rollout en bucle cerrado VECTORIZADO
# --------------------------------------------------------------------------- #
# Es la misma simulacion que politica.rollout_multiflota, con todas las flotas y
# vehiculos avanzando como arrays en lugar de con bucles de Python.
#
# Por que importa: la nota de rollout es lo que decide que red gana el barrido, y
# se calcula UNA VEZ POR CONFIGURACION. La version local recorre 1 800 pasos x
# todos los vehiculos en Python puro, asi que evaluar es tan caro como entrenar y
# la GPU se pasa la mayor parte del tiempo esperando. Aqui cada paso son unas
# pocas operaciones de numpy sobre todos los vehiculos a la vez.
#
# La fisica, el criterio de llegada y el orden de las operaciones son identicos a
# los del pipeline local; verificar.py lo comprueba comparando las poses finales.

def _preparar(flotas, opts, h_pasado):
    """Aplana las flotas en arrays (M, …) y precalcula lo que no cambia."""
    vehs = [v for f in flotas for v in f]
    M = len(vehs)
    tam = [len(f) for f in flotas]
    K = max(tam) - 1 if tam else 0

    par = {}
    par["largo"] = np.array([v.length for v in vehs])
    par["ancho"] = np.array([v.width for v in vehs])
    par["v_max"] = np.array([v.v_max for v in vehs])
    par["a_max"] = np.array([v.a_max for v in vehs])
    par["giro_max"] = np.array([v.delta_max for v in vehs])
    par["grupo"] = np.array([float(v.grupo) for v in vehs])
    par["prioridad"] = np.array([float(v.prioridad) for v in vehs])
    par["meta_x"] = np.array([v.meta[0] for v in vehs])
    par["meta_y"] = np.array([v.meta[1] for v in vehs])
    par["meta_th"] = np.array([np.nan if v.meta_th is None else v.meta_th
                               for v in vehs])
    par["batalla"] = np.array([v.wheelbase for v in vehs])

    # Compañeros de flota: índices globales y máscara de validez (las flotas
    # pequeñas dejan ranuras sobrantes, que no se usan).
    idx_otros = np.zeros((M, K), dtype=np.int64)
    valido = np.zeros((M, K), dtype=bool)
    ctx = np.zeros((M, pol.DIM_CTX))
    base = 0
    for f, flota in enumerate(flotas):
        n = len(flota)
        ctx[base:base + n] = bloque_ctx(opts[f], n, n)
        for i in range(n):
            otros = [base + j for j in range(n) if j != i]
            idx_otros[base + i, :len(otros)] = otros
            valido[base + i, :len(otros)] = True
        base += n

    est = np.zeros((M, 4))
    for m, v in enumerate(vehs):
        est[m, 0], est[m, 1], est[m, 2] = v.inicio
        est[m, 3] = v.v_inicial
    pasado = np.zeros((M, h_pasado, 2))
    return vehs, est, par, idx_otros, valido, ctx, pasado


def rollout(flotas, politica, t_max=180.0, opts=None, n_vec=None,
            horizonte=None, h_pasado=None, guardar_traj=False):
    """Simula todas las flotas con la política. Escribe veh.traj y veh.mision_ok
    y devuelve el nº de llegados por flota, igual que rollout_multiflota. Por
    defecto veh.traj guarda solo la pose inicial y la final (es lo único que usa
    la puntuación y ahorra gigabytes); con guardar_traj=True guarda el camino."""
    if opts is None:
        opts = ["secuencial"] * len(flotas)
    n_vec = pol.N_VECINOS if n_vec is None else n_vec
    horizonte = pol.HORIZONTE_VECINO if horizonte is None else horizonte
    h_pasado = pol.H_PASADO if h_pasado is None else h_pasado

    vehs, est, par, idx_otros, valido, ctx, pasado = _preparar(
        flotas, opts, h_pasado)
    M = len(vehs)
    if M == 0:
        return [0] * len(flotas)
    llegado = np.zeros(M, dtype=bool)
    planes = np.zeros((M, pol.N_PRED, 2))
    camino = [est[:, :3].copy()] if guardar_traj else None

    for paso in range(int(round(t_max / DT))):
        activo = ~llegado
        if not activo.any():
            break
        if paso % pol.N_PRED == 0:
            obs, _ = entradas_instante(est, par, idx_otros, valido, ctx,
                                       n_vec, horizonte, pasado)
            planes[activo] = politica.predecir_lote(obs[activo])

        a_n, g_n = planes[:, paso % pol.N_PRED, 0], planes[:, paso % pol.N_PRED, 1]
        a = np.clip(a_n, -1.0, 1.0) * par["a_max"]
        giro = np.clip(g_n, -1.0, 1.0) * par["giro_max"]
        x, y, th, v = est[:, 0], est[:, 1], est[:, 2], est[:, 3]
        nx = x + v * DT * np.cos(th)
        ny = y + v * DT * np.sin(th)
        nth = ang_norm(th + v * DT * np.tan(giro) / par["batalla"])
        nv = np.clip(v + a * DT, -par["v_max"], par["v_max"])
        est[:, 0] = np.where(activo, nx, x)
        est[:, 1] = np.where(activo, ny, y)
        est[:, 2] = np.where(activo, nth, th)
        est[:, 3] = np.where(activo, nv, v)

        # Historia de controles, ya normalizada y alineada a la derecha.
        nuevo = np.stack([a / np.maximum(par["a_max"], 1e-6),
                          giro / np.maximum(par["giro_max"], 1e-6)], axis=1)
        desplazado = np.concatenate([pasado[:, 1:], nuevo[:, None, :]], axis=1)
        pasado = np.where(activo[:, None, None], desplazado, pasado)

        cerca = np.hypot(est[:, 0] - par["meta_x"], est[:, 1] - par["meta_y"])
        recien = (activo & (cerca < pol.DIST_LLEGADA)
                  & (np.abs(est[:, 3]) < pol.V_LLEGADA))
        est[recien, 3] = 0.0
        llegado |= recien
        if guardar_traj:
            camino.append(est[:, :3].copy())

    for m, veh in enumerate(vehs):
        if guardar_traj:
            veh.traj = [tuple(p[m]) for p in camino]
        else:
            veh.traj = [tuple(camino[0][m]) if camino else veh.inicio,
                        tuple(est[m, :3])]
        veh.dt_plan = DT
        veh.mision_ok = bool(llegado[m])

    llegados, base = [], 0
    for flota in flotas:
        n = len(flota)
        llegados.append(int(llegado[base:base + n].sum()))
        base += n
    return llegados
