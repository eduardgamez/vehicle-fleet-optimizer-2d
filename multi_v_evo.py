#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulador físico de enrutamiento de FLOTAS en ESPACIO CONTINUO.

Modelo físico:

  1. Espacio continuo: las posiciones son coordenadas reales (x, y) en metros.

  2. Cinemática realista (modelo de bicicleta):
         x'  = v · cos(θ)
         y'  = v · sin(θ)
         θ'  = (v / L) · tan(δ)        L = batalla (wheelbase)
         v'  = a                       a = aceleración (acotada)

  3. Giro no holonómico: el ángulo de dirección δ está acotado (|δ| ≤ δ_max),
     lo que impone un radio de giro mínimo  R_min = L / tan(δ_max).

  4. Colisiones reales: los vehículos son rectángulos ORIENTADOS (OBB) y la
     detección usa el Teorema de los Ejes Separadores (SAT), tanto contra los
     obstáculos fijos como entre los propios vehículos.

Capacidades:

  · OPTIMIZACIÓN DE FLOTA — además de la planificación cooperativa clásica
    (vehículo a vehículo en orden fijo), el programa puede optimizar GLOBALMENTE:
    explora muchos órdenes de planificación distintos y se queda con la solución
    de menor tiempo total de la flota. También admite PRIORIDADES personalizadas:
    grupos de vehículos con la misma prioridad (optimizados entre sí) o un orden
    estricto impuesto por el usuario.

  · VEHÍCULOS PERSONALIZABLES — cada vehículo puede definirse desde una entrada
    de texto (lista de diccionarios): tamaño, velocidad, aceleración, capacidad
    de giro, punto y orientación iniciales, destino, prioridad… y el ÁNGULO DE
    LLEGADA exigido en el destino.

  · ÁNGULO DE LLEGADA — el vehículo no solo debe alcanzar el punto final, sino
    hacerlo orientado con un ángulo concreto (aparcamiento direccional). El
    conector analítico persigue una "zanahoria" situada sobre el eje de llegada,
    de modo que el coche se alinea con dicho eje antes de rematar.

  · REUTILIZACIÓN — un vehículo que ya llegó a su destino puede recibir una
    misión nueva escribiendo en la entrada de texto una lista de diccionarios
    con su identificador; solo los vehículos mencionados reciben ruta nueva y
    el resto permanece aparcado (y es respetado como obstáculo).

  · MAPAS DESDE IMAGEN — una imagen puede convertirse en mapa: los píxeles
    cercanos al negro se vuelven obstáculo y los cercanos al blanco, espacio
    libre. El mapa resultante puede GUARDARSE en la carpeta `mapas/` y volver a
    cargarse en cualquier otra ejecución del programa.

Arquitectura:

  · PLANIFICACIÓN — Hybrid A* cooperativo sobre el estado continuo (x, y, θ, v)
    con primitivas de arco del modelo de bicicleta. Cada vehículo evita los
    obstáculos fijos, las metas ajenas y las trayectorias temporales de los
    vehículos ya planificados (obstáculos móviles en el tiempo). El coste es el
    TIEMPO, con una heurística Eikonal-con-obstáculos y una conexión analítica
    (pure-pursuit) para rematar la llegada.

  · EJECUCIÓN — se reproducen las trayectorias resultantes, ya libres de colisión.

Rendimiento:

  El núcleo numérico (geometría OBB/SAT, integración del modelo de bicicleta y
  los chequeos de colisión) domina el cálculo, así que se compila a CÓDIGO
  NATIVO en el arranque con Numba, operando sobre arrays NumPy empaquetados.
  Numba genera instrucciones para la microarquitectura concreta del equipo
  donde se ejecuta (Apple Silicon, x86, …).

  Requiere:   numpy, numba   (pip install numpy numba)
  Opcional:   pillow         (para importar imágenes JPG/BMP/…; sin Pillow se
                              admiten PNG/GIF/PPM a través del propio Tk)
  Ejecutar:   python3 flota_global.py
"""

import ast
import json
import math
import os
import time
import heapq
import queue
import random
import itertools
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

from math import cos, sin, tan, hypot, sqrt, atan2

import numpy as np
from numba import njit


# Acoplamiento giro–velocidad: el ángulo de dirección efectivo se reduce
# linealmente con la rapidez. A velocidad máxima el volante llega a
# (1 - STEER_SPEED_C) del tope; a coche parado, al tope entero. Es un efecto
# LIGERO (como en la vida real: a más velocidad, menos giro para no derrapar),
# no un límite agresivo.
STEER_SPEED_C = 0.35


# --------------------------------------------------------------------------- #
# Mundo (en metros) y dibujo
# --------------------------------------------------------------------------- #
W = 40.0            # ancho del mundo (m)
H = 24.0            # alto del mundo (m)
SCALE = 22          # píxeles por metro

COL_FONDO = "#f4f6fb"
COL_OBST = "#3a4252"
COL_BORDE = "#1f2530"

PALETA = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#118ab2", "#f032e6", "#9a6324", "#469990", "#bfa100",
    "#800000", "#000075", "#e07a5f", "#2a9d8f", "#6a4c93",
]

DT = 0.1            # paso de integración temporal (s)

VEH_LEN = 1.3       # largo por defecto del vehículo (m)
VEH_WID = 0.7      # ancho por defecto del vehículo (m)
VEH_VMAX = 2.5      # velocidad máxima por defecto (m/s)
VEH_AMAX = 1.0      # aceleración máxima por defecto (m/s²)
VEH_GIRO = 35.0     # ángulo de dirección máximo por defecto (grados)

# Carpeta persistente donde se guardan los mapas importados desde imagen.
MAPAS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mapas")


# --------------------------------------------------------------------------- #
# Geometría de dibujo (Python; fuera del camino crítico de cálculo)
# --------------------------------------------------------------------------- #
def obb_corners(x, y, theta, length, width):
    """Esquinas (en orden) de un rectángulo centrado en (x,y) orientado θ."""
    hl, hw = length / 2.0, width / 2.0
    c, s = math.cos(theta), math.sin(theta)
    pts = []
    for lx, ly in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw)):
        pts.append((x + lx * c - ly * s, y + lx * s + ly * c))
    return pts


def rect_poly(x, y, w, h):
    return obb_corners(x + w / 2.0, y + h / 2.0, 0.0, w, h)


def ang_norm(a):
    """Normaliza un ángulo a (-π, π]."""
    return math.atan2(math.sin(a), math.cos(a))


# =========================================================================== #
# NÚCLEOS NUMÉRICOS (compilados a código nativo por Numba)
#
# Toda la geometría de colisión y la integración del modelo de bicicleta operan
# sobre arrays NumPy empaquetados.
#
# Detalle numérico: sin fastmath, la aritmética es la estándar IEEE. Un eje SAT
# degenerado se codifica (0,0): al proyectar da 0 en ambos polígonos y nunca
# reporta separación, es decir, se ignora (como corresponde).
# =========================================================================== #
@njit(cache=True)
def _k_corners(x, y, th, L, Wd):
    """4 esquinas del OBB centrado en (x,y), orientación th, tamaño L×Wd."""
    c = cos(th); s = sin(th)
    hl = 0.5 * L; hw = 0.5 * Wd
    out = np.empty((4, 2))
    out[0, 0] = x + hl * c - hw * s; out[0, 1] = y + hl * s + hw * c
    out[1, 0] = x + hl * c + hw * s; out[1, 1] = y + hl * s - hw * c
    out[2, 0] = x - hl * c + hw * s; out[2, 1] = y - hl * s - hw * c
    out[3, 0] = x - hl * c - hw * s; out[3, 1] = y - hl * s + hw * c
    return out


@njit(cache=True)
def _k_axes(c):
    """Normales unitarias a las 4 aristas (degenerada → (0,0), inocua)."""
    out = np.empty((4, 2))
    for i in range(4):
        x1 = c[i, 0]; y1 = c[i, 1]
        x2 = c[(i + 1) % 4, 0]; y2 = c[(i + 1) % 4, 1]
        ex = x2 - x1; ey = y2 - y1
        nx = -ey; ny = ex
        d = hypot(nx, ny)
        if d > 1e-12:
            out[i, 0] = nx / d; out[i, 1] = ny / d
        else:
            out[i, 0] = 0.0; out[i, 1] = 0.0
    return out


@njit(cache=True)
def _k_sat(cA, axA, cB, axB):
    """SAT entre dos cuadriláteros convexos usando sus 8 ejes candidatos."""
    for src in range(2):
        for i in range(4):
            if src == 0:
                ax = axA[i, 0]; ay = axA[i, 1]
            else:
                ax = axB[i, 0]; ay = axB[i, 1]
            a0 = 1e18; a1 = -1e18; b0 = 1e18; b1 = -1e18
            for j in range(4):
                pa = cA[j, 0] * ax + cA[j, 1] * ay
                if pa < a0: a0 = pa
                if pa > a1: a1 = pa
                pb = cB[j, 0] * ax + cB[j, 1] * ay
                if pb < b0: b0 = pb
                if pb > b1: b1 = pb
            if a1 < b0 or b1 < a0:
                return False
    return True


@njit(cache=True)
def _k_free_sb(x, y, th, Lm, Wm, worldW, worldH,
               oc, oax, obb, K, blc, blax, blbb, B, diag, margin):
    """¿Cabe el OBB (ya inflado a Lm×Wm) sin salirse del mundo ni tocar los
    obstáculos fijos ('oc') ni las metas ajenas bloqueadas ('blc')?"""
    cA = _k_corners(x, y, th, Lm, Wm)
    for j in range(4):
        if cA[j, 0] < 0.0 or cA[j, 0] > worldW or cA[j, 1] < 0.0 or cA[j, 1] > worldH:
            return False
    aA = _k_axes(cA)
    vr = 0.5 * hypot(Lm, Wm)
    for i in range(K):
        if hypot(obb[i, 0] - x, obb[i, 1] - y) > obb[i, 2] + vr:
            continue
        if _k_sat(cA, aA, oc[i], oax[i]):
            return False
    if B > 0:
        vrb = 0.5 * diag + margin
        for i in range(B):
            if hypot(blbb[i, 0] - x, blbb[i, 1] - y) > blbb[i, 2] + vrb:
                continue
            if _k_sat(cA, aA, blc[i], blax[i]):
                return False
    return True


@njit(cache=True)
def _k_hits_dyn(x, y, th, kk, Ld, Wd, rx, roff, rlen, rlw, R, diag, mdin):
    """¿El OBB (inflado a Ld×Wd) choca con algún vehículo RESERVADO en el paso
    fino kk? Para kk más allá del final de una reserva, esta sigue aparcada."""
    if R == 0:
        return False
    cA = _k_corners(x, y, th, Ld, Wd)
    aA = _k_axes(cA)
    vr = 0.5 * diag + mdin
    for j in range(R):
        lj = rlen[j]
        idx = roff[j] + (kk if kk < lj else lj - 1)
        ox = rx[idx, 0]; oy = rx[idx, 1]; oth = rx[idx, 2]
        rl = rlw[j, 0]; rw = rlw[j, 1]
        if hypot(x - ox, y - oy) > vr + 0.5 * hypot(rl, rw):
            continue
        cB = _k_corners(ox, oy, oth, rl, rw)
        aB = _k_axes(cB)
        if _k_sat(cA, aA, cB, aB):
            return True
    return False


@njit(cache=True)
def _k_expand(x0, y0, th0, v0, k, acc, dl, subpasos, h, L,
              vmaxc, vrev, veh_len, veh_wid, margin, mdin, diag,
              worldW, worldH, oc, oax, obb, K, blc, blax, blbb, B,
              rx, roff, rlen, rlw, R, feas, osub, ov):
    """Evalúa TODAS las acciones (accel, delta) de un nodo. Para cada acción integra
    los 'subpasos' del modelo de bicicleta y valida en cada subpaso los obstáculos
    fijos, las metas bloqueadas y las reservas dinámicas. Rellena, por acción a:
    feas[a] = 1/0,  osub[a] = subposes,  ov[a] = velocidad final.

    'dl' son los ÁNGULOS de dirección (rad) de cada acción; el ángulo efectivo
    en cada subpaso se reduce con la rapidez (acoplamiento giro–velocidad)."""
    A = acc.shape[0]
    Lm = veh_len + 2.0 * margin
    Wm = veh_wid + 2.0 * margin
    Ld = veh_len + 2.0 * mdin
    Wd = veh_wid + 2.0 * mdin
    for a in range(A):
        x = x0; y = y0; th = th0; v = v0
        ok = 1
        ac = acc[a]; dl_a = dl[a]
        for sp in range(subpasos):
            v = min(vmaxc, max(-vrev, v + ac * h))
            av = v if v >= 0.0 else -v
            dtn = tan(dl_a * (1.0 - STEER_SPEED_C * (av / vmaxc)))
            th = th + (1.0 / L) * dtn * (v * h)
            x = x + cos(th) * v * h
            y = y + sin(th) * v * h
            if not _k_free_sb(x, y, th, Lm, Wm, worldW, worldH,
                              oc, oax, obb, K, blc, blax, blbb, B, diag, margin):
                ok = 0
                break
            if _k_hits_dyn(x, y, th, k + sp + 1, Ld, Wd,
                           rx, roff, rlen, rlw, R, diag, mdin):
                ok = 0
                break
            osub[a, sp, 0] = x; osub[a, sp, 1] = y; osub[a, sp, 2] = th
        feas[a] = ok
        ov[a] = v


@njit(cache=True)
def _k_tiro(x, y, th, v, k, gx, gy, gth, con_ang, ang_tol,
            L, dmax, ld, vmaxc, amax, goal_tol, v_tol, h,
            veh_len, veh_wid, margin, mdin, diag, worldW, worldH,
            oc, oax, obb, K, blc, blax, blbb, B, rx, roff, rlen, rlw, R):
    """Conexión analítica (pure-pursuit con frenado hasta detenerse en la meta).
    Devuelve (n, poses): n>=0 → éxito con n poses en poses[:n]; n<0 → sin remate.
    Un tope de longitud descarta los rizos que se enroscan sin encarar la meta.

    Valida cada paso igual que la búsqueda: muros y metas ajenas (_k_free_sb),
    los demás vehículos en el instante correcto (_k_hits_dyn) y, al final, que la
    plaza quede libre para siempre (se comprueba fuera con _k_aparca). Es decir,
    el remate es tan seguro frente a colisiones como el propio A*.

    'ld' es la distancia de anticipación (lookahead) del pure-pursuit, acotada:
    corta → corrige el rumbo con firmeza al inicio del remate y luego avanza en
    recta hasta la meta (en vez de un arco amplio y poco pronunciado).

    ÁNGULO DE LLEGADA (con_ang == 1): en vez de perseguir la meta directamente,
    se persigue una "zanahoria" situada sobre el eje de llegada, a una fracción
    de la distancia restante por detrás de la meta en la dirección gth. Al
    acercarse, la zanahoria converge a la meta y el vehículo queda alineado con
    el eje; el remate solo se acepta si la orientación final difiere de gth
    menos que ang_tol."""
    out = np.empty((900, 3))
    n = 0
    Lm = veh_len + 2.0 * margin; Wm = veh_wid + 2.0 * margin
    Ld = veh_len + 2.0 * mdin; Wd = veh_wid + 2.0 * mdin
    d0 = hypot(gx - x, gy - y)
    largo = 0.0
    if con_ang == 1:
        largo_max = 2.4 * d0 + 4.0 * goal_tol + 6.0
    else:
        largo_max = 1.5 * d0 + 2.0 * goal_tol
    kk = k
    for _ in range(900):
        d = hypot(gx - x, gy - y)
        if d <= goal_tol and abs(v) <= v_tol:
            if con_ang == 0:
                return n, out
            da = atan2(sin(th - gth), cos(th - gth))
            if abs(da) <= ang_tol:
                return n, out
            return -1, out
        # Punto perseguido: la meta, o la zanahoria sobre el eje de llegada.
        if con_ang == 1:
            s = 0.62 * d
            smax = 4.0 * L
            if s > smax:
                s = smax
            tx = gx - s * cos(gth); ty = gy - s * sin(gth)
        else:
            tx = gx; ty = gy
        dd = d if d > 0.0 else 0.0
        v_des = min(vmaxc, sqrt(2.0 * amax * dd))
        dv = max(-amax * h, min(amax * h, v_des - v))
        v = min(vmaxc, max(0.0, v + dv))
        alpha = atan2(ty - y, tx - x) - th
        alpha = atan2(sin(alpha), cos(alpha))
        # Lookahead acotado: nunca mayor que 'ld', para que la corrección de rumbo
        # sea firme al principio y el tramo restante quede recto. Cerca del punto
        # perseguido se contrae a la distancia real para homing preciso.
        dt_ = hypot(tx - x, ty - y)
        den = dt_ if dt_ < ld else ld
        if den < 1e-3:
            den = 1e-3
        # Tope de dirección reducido con la rapidez (giro–velocidad).
        dmax_v = dmax * (1.0 - STEER_SPEED_C * (v / vmaxc))
        delta = max(-dmax_v, min(dmax_v, atan2(2.0 * L * sin(alpha), den)))
        th = th + (1.0 / L) * tan(delta) * (v * h)
        x = x + cos(th) * v * h
        y = y + sin(th) * v * h
        largo += abs(v) * h
        if largo > largo_max:
            return -1, out
        if not _k_free_sb(x, y, th, Lm, Wm, worldW, worldH,
                          oc, oax, obb, K, blc, blax, blbb, B, diag, margin):
            return -1, out
        if _k_hits_dyn(x, y, th, kk + 1, Ld, Wd,
                       rx, roff, rlen, rlw, R, diag, mdin):
            return -1, out
        out[n, 0] = x; out[n, 1] = y; out[n, 2] = th
        n += 1
        kk += 1
        if v < 1e-3 and d > goal_tol:
            return -1, out
    return -1, out


@njit(cache=True)
def _k_aparca(px, py, pth, k0, kfin, Ld, Wd,
              rx, roff, rlen, rlw, R, diag, mdin):
    """La plaza final debe quedar libre para siempre: comprueba la pose de
    aparcamiento contra todas las reservas desde k0 hasta que todas paran."""
    for kk in range(k0, kfin + 1):
        if _k_hits_dyn(px, py, pth, kk, Ld, Wd, rx, roff, rlen, rlw, R, diag, mdin):
            return False
    return True


@njit(cache=True)
def _k_maniobra(x, y, th, k, gx, gy, gth, ang_tol,
                L, dmax, ld, vmaxc, amax, goal_tol, v_tol, h,
                veh_len, veh_wid, margin, mdin, diag, worldW, worldH,
                oc, oax, obb, K, blc, blax, blbb, B,
                rx, roff, rlen, rlw, R, res_maxlen):
    """Conector REVERSIBLE para asentar el ÁNGULO DE LLEGADA en sitio apretado.

    Cuando la búsqueda se atasca intentando encarar la meta con la orientación
    exigida, este conector prueba una primera MANIOBRA —adelante o marcha atrás,
    girando a un lado u otro y con distintas longitudes— que puede ALEJARSE de la
    meta, y desde la pose resultante remata hacia adelante con _k_tiro. Es la
    maniobra clásica de 'tirar un poco y rectificar' (o su versión en retroceso).

    Al evaluarse como un TODO (segmento + remate, con validación de colisiones y
    de aparcamiento definitivo), la heurística de distancia de la búsqueda no la
    penaliza aunque el primer tramo se aleje. Devuelve (n, poses) con la maniobra
    completa en poses[:n], o (-1, poses) si ninguna combinación cuadra. Se queda
    con la maniobra total más CORTA entre las que funcionan (no sacrifica calidad)."""
    out = np.empty((1400, 3))
    Ld = veh_len + 2.0 * mdin
    Wd = veh_wid + 2.0 * mdin
    Lm = veh_len + 2.0 * margin
    Wm = veh_wid + 2.0 * margin
    dirs = (1.0, -1.0)                        # adelante / marcha atrás
    steers = (-1.0, -0.5, 0.0, 0.5, 1.0)     # fracción del giro máximo
    seg_max = 30                             # subpasos máximos del tramo de ida
    tail_max = 70                            # subpasos máximos del frenado
    v_man = 0.5 * vmaxc
    if v_man < 0.5:
        v_man = 0.5
    best_n = -1
    best_total = 1e18
    # Segmento = rampa de aceleración (prefijo compartido) + cola de frenado.
    seg = np.empty((seg_max + tail_max, 3))
    for di in range(2):
        dirf = dirs[di]
        for si in range(5):
            frac = steers[si]
            x1 = x; y1 = y; th1 = th
            kk = k
            vmag = 0.0                       # arranca PARADO: sin salto de velocidad
            for step in range(seg_max):
                # Acelera desde 0 respetando a_max (hasta el tope de maniobra).
                vmag = min(v_man, vmag + amax * h)
                vv = dirf * vmag
                fdel = tan(frac * dmax * (1.0 - STEER_SPEED_C * (vmag / vmaxc)))
                th1 = th1 + (1.0 / L) * fdel * (vv * h)
                x1 = x1 + cos(th1) * vv * h
                y1 = y1 + sin(th1) * vv * h
                kk += 1
                if not _k_free_sb(x1, y1, th1, Lm, Wm, worldW, worldH,
                                  oc, oax, obb, K, blc, blax, blbb, B, diag, margin):
                    break                    # este tramo choca: se abandona
                if _k_hits_dyn(x1, y1, th1, kk, Ld, Wd,
                               rx, roff, rlen, rlw, R, diag, mdin):
                    break
                seg[step, 0] = x1; seg[step, 1] = y1; seg[step, 2] = th1
                ncore = step + 1
                if (step & 1) != 0:          # prueba a rematar cada 2 subpasos
                    continue
                # COLA DE FRENADO: decelera hasta v=0 (respetando a_max) antes de
                # cambiar de sentido, para que la inversión pase por velocidad
                # nula y no haya un salto brusco. La pose final queda DETENIDA.
                xt = x1; yt = y1; tht = th1; kt = kk; vt = vmag
                ntail = 0
                tail_ok = True
                while vt > 1e-3 and ntail < tail_max:
                    vt = vt - amax * h
                    if vt < 0.0:
                        vt = 0.0
                    vvt = dirf * vt
                    fdt = tan(frac * dmax * (1.0 - STEER_SPEED_C * (vt / vmaxc)))
                    tht = tht + (1.0 / L) * fdt * (vvt * h)
                    xt = xt + cos(tht) * vvt * h
                    yt = yt + sin(tht) * vvt * h
                    kt += 1
                    if not _k_free_sb(xt, yt, tht, Lm, Wm, worldW, worldH,
                                      oc, oax, obb, K, blc, blax, blbb, B, diag, margin):
                        tail_ok = False; break
                    if _k_hits_dyn(xt, yt, tht, kt, Ld, Wd,
                                   rx, roff, rlen, rlw, R, diag, mdin):
                        tail_ok = False; break
                    seg[ncore + ntail, 0] = xt
                    seg[ncore + ntail, 1] = yt
                    seg[ncore + ntail, 2] = tht
                    ntail += 1
                if not tail_ok:
                    continue
                nseg = ncore + ntail
                # Desde el fin del tramo (coche DETENIDO) intenta el remate normal.
                nt, pt = _k_tiro(xt, yt, tht, 0.0, kt, gx, gy, gth, 1, ang_tol,
                                 L, dmax, ld, vmaxc, amax, goal_tol, v_tol, h,
                                 veh_len, veh_wid, margin, mdin, diag, worldW, worldH,
                                 oc, oax, obb, K, blc, blax, blbb, B,
                                 rx, roff, rlen, rlw, R)
                if nt <= 0:
                    continue
                kfin = kt + nt
                kmax = kfin if kfin > res_maxlen else res_maxlen
                if not _k_aparca(pt[nt - 1, 0], pt[nt - 1, 1], pt[nt - 1, 2],
                                 kfin, kmax, Ld, Wd,
                                 rx, roff, rlen, rlw, R, diag, mdin):
                    continue
                total = nseg + nt
                if total < best_total:
                    best_total = total
                    m = 0
                    for q in range(nseg):
                        out[m, 0] = seg[q, 0]; out[m, 1] = seg[q, 1]
                        out[m, 2] = seg[q, 2]; m += 1
                    for q in range(nt):
                        out[m, 0] = pt[q, 0]; out[m, 1] = pt[q, 1]
                        out[m, 2] = pt[q, 2]; m += 1
                    best_n = m
    return best_n, out


@njit(cache=True)
def _k_occ(nx, ny, res, infl, worldW, worldH, oc, oax, obb, K):
    """Rejilla de ocupación inflada por 'infl': cada celda es libre si el punto
    (inflado) no toca bordes ni obstáculos. Se calcula una vez por mapa."""
    out = np.empty((nx, ny), np.int8)
    side = 2.0 * infl
    _eb = np.empty((0, 4, 2))
    _ebb = np.empty((0, 3))
    for ix in range(nx):
        for iy in range(ny):
            x = ix * res; y = iy * res
            free = _k_free_sb(x, y, 0.0, side, side, worldW, worldH,
                              oc, oax, obb, K, _eb, _eb, _ebb, 0, 1.0, 0.0)
            out[ix, iy] = 1 if free else 0
    return out


# Arrays vacíos reutilizables (evitan reasignar en cada consulta).
_EMPTY_C = np.empty((0, 4, 2))
_EMPTY_BB = np.empty((0, 3))
_EMPTY_XY = np.empty((0, 3))
_EMPTY_OFF = np.empty(0, np.int64)
_EMPTY_LW = np.empty((0, 2))


def _pack_polys(polys):
    """Empaqueta polígonos (4 esquinas) en arrays: (K,4,2) esquinas, (K,4,2) ejes
    precalculados y (K,3) círculo envolvente (centro + radio) para descarte rápido."""
    K = len(polys)
    c = np.zeros((K, 4, 2)); ax = np.zeros((K, 4, 2)); bb = np.zeros((K, 3))
    for i, p in enumerate(polys):
        for j in range(4):
            c[i, j, 0] = p[j][0]; c[i, j, 1] = p[j][1]
        for j in range(4):
            x1, y1 = p[j]; x2, y2 = p[(j + 1) % 4]
            ex = x2 - x1; ey = y2 - y1
            nx = -ey; ny = ex
            d = hypot(nx, ny)
            if d > 1e-12:
                ax[i, j, 0] = nx / d; ax[i, j, 1] = ny / d
        cx = sum(q[0] for q in p) / 4.0
        cy = sum(q[1] for q in p) / 4.0
        rr = 0.0
        for q in p:
            r = hypot(q[0] - cx, q[1] - cy)
            if r > rr:
                rr = r
        bb[i, 0] = cx; bb[i, 1] = cy; bb[i, 2] = rr
    return c, ax, bb


# --------------------------------------------------------------------------- #
# Obstáculos (polígonos convexos) y entorno
# --------------------------------------------------------------------------- #
class Entorno:
    def __init__(self):
        self.obstaculos = []        # lista de polígonos (para dibujar)
        self.np_c = _EMPTY_C        # (K,4,2) esquinas
        self.np_ax = _EMPTY_C       # (K,4,2) ejes precalculados
        self.np_bb = _EMPTY_BB      # (K,3) centro + radio
        self.origen = "aleatorio"   # "aleatorio" | nombre del mapa importado
        self.rects = None           # rectángulos [x,y,w,h] si el mapa vino de imagen

    def _empaquetar(self, obs):
        self.obstaculos = obs
        if obs:
            self.np_c, self.np_ax, self.np_bb = _pack_polys(obs)
        else:
            self.np_c, self.np_ax, self.np_bb = _EMPTY_C, _EMPTY_C, _EMPTY_BB

    def generar(self, densidad=0.0):
        """Genera un mapa tipo CIUDAD: manzanas rectangulares separadas por calles
        de anchura variable —algunas estrechas—, con un anillo perimetral libre.
        'densidad' añade tabiques extra. Cada llamada produce un mapa distinto."""
        borde = 2.0
        calles = [2.2, 2.4, 2.8, 3.4, 4.0]
        obs = []
        x = borde
        while x < W - borde - 2.5:
            bw = random.uniform(3.5, 7.0)
            bw = min(bw, W - borde - x)
            if bw < 2.5:
                break
            y = borde
            while y < H - borde - 2.0:
                bh = random.uniform(2.5, 6.0)
                bh = min(bh, H - borde - y)
                if bh < 2.0:
                    break
                if random.random() < 0.82:
                    if bw > 5.0 and random.random() < 0.4:
                        hueco = random.uniform(1.0, bw - 3.5)
                        obs.append(rect_poly(x, y, hueco, bh))
                        obs.append(rect_poly(x + hueco + 2.3, y,
                                             bw - hueco - 2.3, bh))
                    else:
                        obs.append(rect_poly(x, y, bw, bh))
                y += bh + random.choice(calles)
            x += bw + random.choice(calles)

        for _ in range(2):
            obs.append(obb_corners(random.uniform(borde + 3, W - borde - 3),
                                   random.uniform(borde + 3, H - borde - 3),
                                   random.uniform(0, math.pi),
                                   random.uniform(2.0, 4.0), 1.6))

        for _ in range(int(densidad * 25)):
            obs.append(obb_corners(random.uniform(borde, W - borde),
                                   random.uniform(borde, H - borde),
                                   random.uniform(0, math.pi),
                                   random.uniform(1.5, 3.0),
                                   random.uniform(1.0, 1.8)))

        self.origen = "aleatorio"
        self.rects = None
        self._empaquetar(obs)

    def desde_rects(self, rects, nombre="importado"):
        """Construye el mapa a partir de rectángulos [x, y, ancho, alto] en metros
        (los que produce la conversión de una imagen)."""
        obs = [rect_poly(rx, ry, rw, rh) for rx, ry, rw, rh in rects]
        self.origen = nombre
        self.rects = [list(map(float, r)) for r in rects]
        self._empaquetar(obs)

    def libre(self, x, y, theta, length, width, margen=0.0):
        """¿Cabe el vehículo (OBB) sin tocar bordes ni obstáculos?"""
        Lm = length + 2.0 * margen
        Wm = width + 2.0 * margen
        return bool(_k_free_sb(x, y, theta, Lm, Wm, W, H,
                               self.np_c, self.np_ax, self.np_bb, self.np_c.shape[0],
                               _EMPTY_C, _EMPTY_C, _EMPTY_BB, 0, 1.0, 0.0))


# --------------------------------------------------------------------------- #
# Mapas desde IMAGEN: negro → obstáculo, blanco → libre. Persistencia en disco.
# --------------------------------------------------------------------------- #
# La imagen se muestrea en una rejilla de celdas; cada celda cuyo brillo medio
# queda por debajo del umbral se considera obstáculo. Después las celdas se
# fusionan en rectángulos maximales (pocos OBB grandes en vez de miles de
# celdas sueltas) para que el motor SAT siga siendo rápido.
IMG_NX = 160        # celdas horizontales de la rejilla de muestreo
IMG_NY = 96         # celdas verticales (misma proporción que el mundo 40×24)
IMG_UMBRAL = 128    # brillo 0-255: por debajo → obstáculo (cercano al negro)


def _leer_imagen_gris(ruta):
    """Devuelve (ancho, alto, funcion_pixel) donde funcion_pixel(x, y) → brillo
    0-255. Usa Pillow si está instalado (cualquier formato); si no, recurre al
    PhotoImage de Tk (PNG/GIF/PPM/PGM)."""
    try:
        from PIL import Image
        img = Image.open(ruta).convert("L")
        px = img.load()
        return img.width, img.height, lambda x, y: px[x, y]
    except ImportError:
        pass
    import tkinter as tk
    img = tk.PhotoImage(file=ruta)      # requiere un Tk ya creado
    w, h = img.width(), img.height()

    def pixel(x, y):
        v = img.get(x, y)
        if isinstance(v, str):
            v = tuple(int(t) for t in v.split())
        return (v[0] + v[1] + v[2]) // 3 if len(v) >= 3 else v[0]
    return w, h, pixel


def imagen_a_grid(ruta, nx=IMG_NX, ny=IMG_NY, umbral=IMG_UMBRAL):
    """Rejilla booleana nx×ny (True = obstáculo) a partir de una imagen: cada
    celda promedia una muestra de píxeles de su bloque correspondiente."""
    iw, ih, pixel = _leer_imagen_gris(ruta)
    grid = [[False] * ny for _ in range(nx)]
    for ix in range(nx):
        x0 = ix * iw // nx
        x1 = max(x0 + 1, (ix + 1) * iw // nx)
        for iy in range(ny):
            y0 = iy * ih // ny
            y1 = max(y0 + 1, (iy + 1) * ih // ny)
            # Muestra hasta 3×3 puntos del bloque (suficiente y rápido sin PIL).
            sx = max(1, (x1 - x0) // 3)
            sy = max(1, (y1 - y0) // 3)
            total = 0; cuenta = 0
            for px_ in range(x0, x1, sx):
                for py_ in range(y0, y1, sy):
                    total += pixel(min(px_, iw - 1), min(py_, ih - 1))
                    cuenta += 1
            grid[ix][iy] = (total / cuenta) < umbral
    return grid


def grid_a_rects(grid):
    """Fusiona las celdas ocupadas en rectángulos: primero tiras horizontales por
    fila, luego tiras idénticas de filas consecutivas se unen verticalmente.
    Devuelve rectángulos de celdas (ix, iy, n_celdas_x, n_celdas_y)."""
    nx = len(grid); ny = len(grid[0]) if nx else 0
    abiertas = {}                  # (ix0, w) → [iy0, alto]
    rects = []
    for iy in range(ny):
        tiras = []
        ix = 0
        while ix < nx:
            if grid[ix][iy]:
                ix0 = ix
                while ix < nx and grid[ix][iy]:
                    ix += 1
                tiras.append((ix0, ix - ix0))
            else:
                ix += 1
        nuevas = {}
        for t in tiras:
            if t in abiertas:
                r = abiertas.pop(t)
                r[1] += 1
                nuevas[t] = r
            else:
                nuevas[t] = [iy, 1]
        for (ix0, w), (iy0, alto) in abiertas.items():
            rects.append((ix0, iy0, w, alto))
        abiertas = nuevas
    for (ix0, w), (iy0, alto) in abiertas.items():
        rects.append((ix0, iy0, w, alto))
    return rects


def imagen_a_rects(ruta):
    """Imagen → rectángulos de obstáculo en METROS (mapeados al mundo W×H)."""
    grid = imagen_a_grid(ruta)
    res_x = W / len(grid)
    res_y = H / len(grid[0])
    return [(ix * res_x, iy * res_y, w * res_x, h * res_y)
            for ix, iy, w, h in grid_a_rects(grid)]


def guardar_mapa(nombre, rects):
    """Guarda el mapa (rectángulos en metros) en `mapas/<nombre>.json` para poder
    recargarlo en cualquier otra ejecución del programa."""
    os.makedirs(MAPAS_DIR, exist_ok=True)
    nombre = "".join(c for c in nombre if c.isalnum() or c in "-_ ").strip()
    if not nombre:
        nombre = "mapa"
    ruta = os.path.join(MAPAS_DIR, nombre + ".json")
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump({"version": 1, "mundo": [W, H],
                   "rects": [list(map(float, r)) for r in rects]}, f)
    return ruta


def cargar_mapa(ruta):
    """Carga un mapa guardado. Si el mundo del fichero tenía otro tamaño, los
    rectángulos se reescalan al mundo actual."""
    with open(ruta, "r", encoding="utf-8") as f:
        datos = json.load(f)
    w0, h0 = datos.get("mundo", [W, H])
    fx, fy = W / w0, H / h0
    return [(rx * fx, ry * fy, rw * fx, rh * fy)
            for rx, ry, rw, rh in datos["rects"]]


def listar_mapas():
    """Nombres de los mapas guardados en la carpeta `mapas/`."""
    if not os.path.isdir(MAPAS_DIR):
        return []
    return sorted(f[:-5] for f in os.listdir(MAPAS_DIR) if f.endswith(".json"))


# --------------------------------------------------------------------------- #
# Vehículo  (su trayectoria es una pose por paso fino de DT segundos)
# --------------------------------------------------------------------------- #
class Vehiculo:
    """Vehículo totalmente parametrizable. 'meta_th' es el ÁNGULO DE LLEGADA
    exigido en el destino (radianes), o None si la orientación final es libre."""

    def __init__(self, idx, inicio, meta, length=VEH_LEN, width=VEH_WID,
                 v_max=VEH_VMAX, a_max=VEH_AMAX, giro_max=None,
                 meta_th=None, grupo=1, prioridad=1, vid=None):
        self.idx = idx                # índice interno (color, dibujo)
        self.vid = vid if vid is not None else idx + 1   # identificador de usuario
        self.inicio = inicio          # (x, y, theta)
        self.meta = meta              # (x, y)
        self.meta_th = meta_th        # ángulo de llegada exigido (rad) o None
        self.grupo = grupo            # grupo de prioridad (menor → antes)
        self.prioridad = prioridad    # prioridad DENTRO del grupo (menor → antes)
        self.length = length
        self.width = width
        self.wheelbase = 0.55 * length
        self.delta_max = giro_max if giro_max is not None else math.radians(VEH_GIRO)
        self.v_max = v_max
        self.a_max = a_max            # aceleración/frenado máximos (m/s²)
        self.traj = []                # [(x, y, theta), ...]
        self.dt_plan = DT
        self.mision_ok = False        # ¿su última misión obtuvo ruta?

    @property
    def radio_giro_min(self):
        return self.wheelbase / math.tan(self.delta_max)

    @property
    def diag(self):
        return math.hypot(self.length, self.width)

    @property
    def pose_final(self):
        """Pose en la que queda aparcado al terminar (o el inicio si no hay ruta)."""
        return self.traj[-1] if self.traj else self.inicio

    def pose_en_tiempo(self, t):
        """Pose en el instante t (s). Tras el final se queda aparcado en la meta."""
        if not self.traj:
            return self.inicio
        i = int(round(t / DT))
        if i >= len(self.traj):
            return self.traj[-1]
        if i < 0:
            return self.traj[0]
        return self.traj[i]

    @property
    def duracion(self):
        return max(0, len(self.traj) - 1) * DT


# --------------------------------------------------------------------------- #
# Reservas espacio-tiempo: trayectorias ya comprometidas por otros vehículos
# --------------------------------------------------------------------------- #
class Reservas:
    """Guarda las trayectorias temporales ya comprometidas. Para pasos posteriores
    al final de una trayectoria, el vehículo sigue aparcado en su última pose."""

    def __init__(self):
        self.items = []               # (traj, length, width)

    def add(self, traj, length, width):
        self.items.append((traj, length, width))

    def copia(self):
        r = Reservas()
        r.items = list(self.items)
        return r


# --------------------------------------------------------------------------- #
# Hybrid A* COOPERATIVO en espacio-tiempo (x, y, θ, v, k)
# --------------------------------------------------------------------------- #
class Planificador:
    def __init__(self, entorno):
        self.env = entorno
        self.res_pos = 0.7
        self.h_res = 0.5
        self._occ_sig = None
        self.res_v = 0.5
        self.dt = 0.4
        self.subpasos = 4
        self.goal_tol = 1.6           # tolerancia GRUESA: dispara/acepta el remate
        self.goal_tol_fin = 0.20      # tolerancia FINA: el piloto conduce hasta aquí
        #                              (para no quedarse parado a un paso de la meta)
        self.v_tol = 0.45
        self.ang_tol_meta = math.radians(10)   # tolerancia del ÁNGULO DE LLEGADA
        self.k_max = 1600
        self.dist_tiro = 10.0
        self.ang_tiro = math.radians(30)
        self.ld_tiro = 1.6            # lookahead del remate: acerca el conector al
        #                              óptimo tipo Dubins (arco al radio necesario +
        #                              tangente recta). Bajarlo → más cerca del radio
        #                              mínimo (más corto) a riesgo de sobreoscilar;
        #                              subirlo → arcos más amplios (más largos).
        self.margen = 0.10
        self.margen_din = 0.15
        self.bloqueos = []
        self.v_max_c = 5.0
        self.a_max = 1.0
        self.v_rev = 2.5
        self.res_ang = math.radians(10)
        self.max_exp = 600000
        self.peso_h = 1.6
        self.dir_fracs = []
        self.configurar_calidad(3)
        self.deadline = None
        self.tick = None
        # Conector reversible: entra en acción solo cuando la búsqueda se atasca
        # asentando el ángulo de llegada (más nodos que 'umbral_maniobra'), y como
        # mucho una vez cada 'paso_maniobra' expansiones para no encarecerla.
        #
        # NOTA DE CALIBRACIÓN: el ritmo real de expansión del núcleo (con SAT +
        # modelo de bicicleta en Python/Numba, sin vectorizar) ronda unos pocos
        # miles de nodos/s, no cientos de miles. Con el umbral antiguo (80 000)
        # la búsqueda tardaba 20-30 s en siquiera INTENTAR el conector reversible,
        # y con angulo_llegada exigido (el caso más común, ya que el generador
        # aleatorio casi siempre pide un ángulo concreto) la búsqueda estándar
        # rara vez lo resuelve sola: el resultado era agotar el presupuesto de
        # tiempo/expansiones sin ruta, dejando vehículos «aparcados» sin más.
        # Bajarlo deja que el conector entre en juego mucho antes.
        self.umbral_maniobra = 3_000
        self.paso_maniobra = 600
        self._exp_ult_maniobra = 0

    def configurar_calidad(self, nivel):
        """Equilibrio TIEMPO ↔ CALIDAD de ruta. A mayor nivel: más ángulos de giro
        (más densos cerca de 0 para corregir el rumbo con finura), resolución
        angular más fina y más expansiones permitidas.

        El 3er valor es el PESO del heurístico (A* ponderado): >1 acelera pero
        aleja del óptimo → rutas más largas y curvas. Baja de verdad con la
        calidad: el óptimo es «recto cuando conviene, giro firme cuando hace
        falta», y esa forma emerge sola al acercar el peso a 1. Medido en mapas
        con obstáculos y reservas, peso≈1.5 (nivel 3) no pierde robustez; por
        debajo de ~1.2 los casos difíciles se disparan en tiempo, reservado a la
        calidad alta."""
        tabla = {
            1: (17, 12, 1.5,  800000),
            2: (23, 11, 1.35, 1100000),
            3: (31, 10, 1.25, 1600000),
            4: (41,  8, 1.15, 2400000),
            5: (55,  6, 1.05, 3600000),
        }
        n_dir, ang, peso, mx = tabla.get(int(nivel), tabla[3])
        half = n_dir // 2
        fracs = [0.0]
        for i in range(1, half + 1):
            f = (i / half) ** 1.3
            fracs += [f, -f]
        self.dir_fracs = sorted(fracs)
        self.res_ang = math.radians(ang)
        self.peso_h = peso
        self.max_exp = mx

    def _clave(self, x, y, th, v, k):
        return (int(x / self.res_pos), int(y / self.res_pos),
                int((th % (2 * math.pi)) / self.res_ang),
                int(round(v / self.res_v)), k)

    # ------------------- empaquetado para los kernels -------------------- #
    def _pack_bloqueos(self):
        B = len(self.bloqueos)
        if B == 0:
            return _EMPTY_C, _EMPTY_C, _EMPTY_BB
        c = np.zeros((B, 4, 2)); ax = np.zeros((B, 4, 2)); bb = np.zeros((B, 3))
        for i, (poly, (cx, cy, rr)) in enumerate(self.bloqueos):
            for j in range(4):
                c[i, j, 0] = poly[j][0]; c[i, j, 1] = poly[j][1]
            for j in range(4):
                x1, y1 = poly[j]; x2, y2 = poly[(j + 1) % 4]
                ex = x2 - x1; ey = y2 - y1
                nx = -ey; ny = ex
                d = hypot(nx, ny)
                if d > 1e-12:
                    ax[i, j, 0] = nx / d; ax[i, j, 1] = ny / d
            bb[i, 0] = cx; bb[i, 1] = cy; bb[i, 2] = rr
        return c, ax, bb

    def _pack_reservas(self, reservas):
        items = reservas.items
        R = len(items)
        if R == 0:
            return _EMPTY_XY, _EMPTY_OFF, _EMPTY_OFF, _EMPTY_LW, 0
        T = sum(len(t) for t, _, _ in items)
        rx = np.empty((T, 3)); off = np.empty(R, np.int64)
        rlen = np.empty(R, np.int64); rlw = np.empty((R, 2))
        p = 0
        for j, (traj, l, w) in enumerate(items):
            off[j] = p; rlen[j] = len(traj); rlw[j, 0] = l; rlw[j, 1] = w
            for pose in traj:
                rx[p, 0] = pose[0]; rx[p, 1] = pose[1]; rx[p, 2] = pose[2]
                p += 1
        return rx, off, rlen, rlw, R

    # ------------------- heurístico con obstáculos ----------------------- #
    def _asegurar_ocupacion(self):
        """Rejilla de ocupación del mapa, inflada por el semiancho del vehículo.
        Depende solo de los obstáculos y del tamaño del vehículo, así que se
        calcula una vez por mapa y se reutiliza mientras no cambien."""
        infl = 0.5 * self._wid + self.margen
        sig = (id(self.env.obstaculos), round(infl, 3), self.h_res)
        if self._occ_sig == sig:
            return
        res = self.h_res
        nx = int(W / res) + 1
        ny = int(H / res) + 1
        self._occ = _k_occ(nx, ny, res, infl, W, H,
                           self.env.np_c, self.env.np_ax, self.env.np_bb,
                           self.env.np_c.shape[0])
        self._occ_nx, self._occ_ny = nx, ny
        self._occ_sig = sig

    def _construir_heuristica(self, gx, gy):
        """Campo de distancias desde la meta por FAST MARCHING (Eikonal |∇T|=1)
        sobre la rejilla libre. A diferencia del Dijkstra 8-conexo (octil), cuyo
        gradiente desciende por el eje dominante y luego en diagonal —lo que induce
        el 'ir plano y recalibrar de golpe'— y escalona alrededor de las celdas de
        obstáculo —lo que provoca serpenteo—, el campo Eikonal es la distancia
        euclídea-que-rodea-obstáculos: su gradiente apunta RECTO a la meta en
        espacio abierto y solo se curva cuando un obstáculo lo obliga de verdad."""
        self._asegurar_ocupacion()
        res = self.h_res
        nx, ny, occ = self._occ_nx, self._occ_ny, self._occ
        INF = float("inf")
        dist = np.full((nx, ny), INF)
        frozen = np.zeros((nx, ny), dtype=bool)
        gi = min(nx - 1, max(0, int(round(gx / res))))
        gj = min(ny - 1, max(0, int(round(gy / res))))
        if not occ[gi, gj]:
            bd = INF
            for ix in range(nx):
                for iy in range(ny):
                    if occ[ix, iy]:
                        dd = (ix * res - gx) ** 2 + (iy * res - gy) ** 2
                        if dd < bd:
                            bd, gi, gj = dd, ix, iy
        dist[gi, gj] = 0.0
        pq = [(0.0, gi, gj)]
        while pq:
            _d, ix, iy = heapq.heappop(pq)
            if frozen[ix, iy]:
                continue
            frozen[ix, iy] = True     # valor definitivo (fast marching)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                jx, jy = ix + dx, iy + dy
                if not (0 <= jx < nx and 0 <= jy < ny) or not occ[jx, jy] or frozen[jx, jy]:
                    continue
                # Actualización upwind de Godunov con los vecinos ya congelados de
                # cada eje (a: horizontal, b: vertical).
                a = INF
                if jx - 1 >= 0 and frozen[jx - 1, jy] and dist[jx - 1, jy] < a:
                    a = dist[jx - 1, jy]
                if jx + 1 < nx and frozen[jx + 1, jy] and dist[jx + 1, jy] < a:
                    a = dist[jx + 1, jy]
                b = INF
                if jy - 1 >= 0 and frozen[jx, jy - 1] and dist[jx, jy - 1] < b:
                    b = dist[jx, jy - 1]
                if jy + 1 < ny and frozen[jx, jy + 1] and dist[jx, jy + 1] < b:
                    b = dist[jx, jy + 1]
                if a == INF:
                    nd = b + res
                elif b == INF:
                    nd = a + res
                elif abs(a - b) >= res:
                    nd = min(a, b) + res
                else:
                    nd = 0.5 * (a + b + math.sqrt(2.0 * res * res - (a - b) ** 2))
                if nd < dist[jx, jy]:
                    dist[jx, jy] = nd
                    heapq.heappush(pq, (nd, jx, jy))
        self._hdist = dist
        self._h_gx, self._h_gy = gx, gy

    def _h_time(self, x, y):
        """Tiempo estimado a la meta = distancia (campo Eikonal) / v_max. Se
        interpola BILINEALMENTE entre las 4 celdas vecinas para que el gradiente que
        ve el A* sea continuo (sin escalones), reforzando el guiado recto. Respaldo:
        celda más próxima, o euclídeo fuera de rejilla / junto a celdas bloqueadas."""
        INF = float("inf")
        res = self.h_res
        fx = x / res
        fy = y / res
        ix = int(fx)
        iy = int(fy)
        if 0 <= ix < self._occ_nx - 1 and 0 <= iy < self._occ_ny - 1:
            d00 = self._hdist[ix, iy]; d10 = self._hdist[ix + 1, iy]
            d01 = self._hdist[ix, iy + 1]; d11 = self._hdist[ix + 1, iy + 1]
            if d00 < INF and d10 < INF and d01 < INF and d11 < INF:
                tx = fx - ix; ty = fy - iy
                d = (d00 * (1 - tx) * (1 - ty) + d10 * tx * (1 - ty)
                     + d01 * (1 - tx) * ty + d11 * tx * ty)
                return d / self.v_max_c
        ixr = int(round(x / res))
        iyr = int(round(y / res))
        if 0 <= ixr < self._occ_nx and 0 <= iyr < self._occ_ny:
            d = self._hdist[ixr, iyr]
            if d < INF:
                return d / self.v_max_c
        return math.hypot(x - self._h_gx, y - self._h_gy) / self.v_max_c

    # ----------------------------- búsqueda ------------------------------ #
    def planificar(self, veh, reservas, inicio=None):
        """Devuelve la trayectoria [(x,y,th), ...] (una pose por paso fino de DT,
        con la velocidad ya incorporada) o None si no halla solución. El estado
        incluye la velocidad (x,y,θ,v,k) y las acciones eligen la aceleración
        (acotada) y la dirección, de modo que el planificador puede acelerar o
        frenar en cualquier momento si eso da una ruta mejor.

        'inicio' permite planificar desde una pose distinta de veh.inicio (p. ej.
        una misión nueva desde la plaza donde el vehículo quedó aparcado).

        Si veh.meta_th no es None, la llegada solo se acepta con la orientación
        final dentro de ang_tol_meta alrededor de ese ángulo."""
        self._len, self._wid = veh.length, veh.width
        self.diag = veh.diag
        L = veh.wheelbase
        dmax = veh.delta_max
        sx, sy, sth = inicio if inicio is not None else veh.inicio
        gx, gy = veh.meta
        con_ang = 0 if veh.meta_th is None else 1
        gth = veh.meta_th if con_ang else 0.0
        ang_tol = self.ang_tol_meta
        self.v_max_c = veh.v_max
        self.a_max = veh.a_max
        self.v_rev = 0.5 * veh.v_max

        self._construir_heuristica(gx, gy)

        # Repertorio de acciones: (aceleración, dirección), con direcciones finas
        # cerca de 0 para corregir el rumbo suavemente desde el principio.
        a = self.a_max
        deltas = [f * dmax for f in self.dir_fracs]
        acciones = [(av, d) for av in (a, 0.0, -a) for d in deltas]
        A = len(acciones)
        acc = np.array([ac for ac, _ in acciones], dtype=np.float64)
        dl = np.array([d for _, d in acciones], dtype=np.float64)

        oc, oax, obb = self.env.np_c, self.env.np_ax, self.env.np_bb
        K = oc.shape[0]
        blc, blax, blbb = self._pack_bloqueos()
        B = blc.shape[0]
        rx, roff, rlen, rlw, R = self._pack_reservas(reservas)

        ns = self.subpasos
        h = self.dt / ns
        feas = np.empty(A, np.int8)
        osub = np.empty((A, ns, 3))
        ov = np.empty(A)

        # Paso a partir del cual todas las reservas están detenidas.
        res_maxlen = 0
        for traj, _, _ in reservas.items:
            if len(traj) > res_maxlen:
                res_maxlen = len(traj)

        # El grafo de reconstrucción se indexa por ID único de nodo (no por la
        # clave discretizada), de modo que cada tramo arranca donde acaba el de su
        # padre → trayectoria continua. La clave solo poda estados dominados.
        clave0 = self._clave(sx, sy, sth, 0.0, 0)
        nid = 0
        padre_id = {}
        arista_id = {}
        delta_prev = {0: 0.0}
        h0 = self._h_time(sx, sy)
        abierto = [(self.peso_h * h0, 0, sx, sy, sth, 0.0, 0, 0.0)]
        mejor_g = {clave0: 0.0}
        expand = 0
        self._exp_ult_maniobra = 0
        self._last_tick = time.perf_counter()
        self.motivo = "limite"

        while abierto and expand < self.max_exp:
            _f, cid, x, y, th, v, k, g = heapq.heappop(abierto)
            expand += 1

            if (expand & 255) == 0:
                now = time.perf_counter()
                if self.deadline is not None and now > self.deadline:
                    self.motivo = "limite"
                    return None
                if self.tick is not None and now - self._last_tick > 0.01:
                    self._last_tick = now
                    self.tick(expand)

            d_goal = math.hypot(x - gx, y - gy)
            if d_goal <= self.dist_tiro:
                # Se intenta PRIMERO conducir hasta el punto exacto (tolerancia
                # fina) con el piloto: así no se acepta un nodo que se queda parado a
                # un paso de la meta. Remata solo si ya apunta bien (o está muy
                # cerca), para no trazar un arco amplio desde una orientación mala.
                # Con ángulo de llegada, el "apuntar bien" se mide hacia la
                # zanahoria del eje de llegada, no hacia la meta.
                if con_ang:
                    s0 = min(0.62 * d_goal, 4.0 * L)
                    txa = gx - s0 * math.cos(gth); tya = gy - s0 * math.sin(gth)
                else:
                    txa = gx; tya = gy
                alpha0 = math.atan2(tya - y, txa - x) - th
                alpha0 = math.atan2(math.sin(alpha0), math.cos(alpha0))
                if abs(alpha0) <= self.ang_tiro or d_goal <= self.goal_tol * 1.5:
                    n, poses = _k_tiro(x, y, th, v, k, gx, gy, gth, con_ang, ang_tol,
                                       L, dmax, self.ld_tiro,
                                       self.v_max_c, self.a_max, self.goal_tol_fin,
                                       self.v_tol, h, self._len, self._wid,
                                       self.margen, self.margen_din, self.diag,
                                       W, H, oc, oax, obb, K, blc, blax, blbb, B,
                                       rx, roff, rlen, rlw, R)
                    if n > 0:
                        tiro = [(poses[i, 0], poses[i, 1], poses[i, 2]) for i in range(n)]
                        kfin_t = k + n
                        Ld = self._len + 2.0 * self.margen_din
                        Wd = self._wid + 2.0 * self.margen_din
                        fx, fy, fth = tiro[-1]
                        kmax = kfin_t if kfin_t > res_maxlen else res_maxlen
                        if _k_aparca(fx, fy, fth, kfin_t, kmax, Ld, Wd,
                                     rx, roff, rlen, rlw, R, self.diag, self.margen_din):
                            base = self._reconstruir(padre_id, arista_id, cid,
                                                     (sx, sy, sth))
                            self.motivo = "ok"
                            return base + tiro

            # Respaldo: ya está dentro de la tolerancia gruesa y casi parado (el
            # remate no aportó o no era viable) → acepta aquí si la plaza queda
            # libre y, con ángulo de llegada exigido, si la orientación cuadra.
            if d_goal <= self.goal_tol and abs(v) <= self.v_tol:
                ang_ok = True
                if con_ang:
                    ang_ok = abs(ang_norm(th - gth)) <= 1.5 * ang_tol
                if ang_ok:
                    Ld = self._len + 2.0 * self.margen_din
                    Wd = self._wid + 2.0 * self.margen_din
                    kfin = k if k > res_maxlen else res_maxlen
                    if _k_aparca(x, y, th, k, kfin, Ld, Wd,
                                 rx, roff, rlen, rlw, R, self.diag, self.margen_din):
                        self.motivo = "ok"
                        return self._reconstruir(padre_id, arista_id, cid, (sx, sy, sth))

            # Conector REVERSIBLE (adelante-y-atrás para cuadrar el ángulo): solo
            # cuando la búsqueda ya se atasca con el ángulo de llegada exigido y el
            # coche está cerca de la meta. Se prueba de forma espaciada para no
            # encarecer el caso normal. Si encuentra maniobra, cierra la ruta.
            if (con_ang and d_goal <= self.dist_tiro
                    and expand > self.umbral_maniobra
                    and expand - self._exp_ult_maniobra >= self.paso_maniobra):
                self._exp_ult_maniobra = expand
                nm, pm = _k_maniobra(x, y, th, k, gx, gy, gth, ang_tol,
                                     L, dmax, self.ld_tiro, self.v_max_c, self.a_max,
                                     self.goal_tol_fin, self.v_tol, h,
                                     self._len, self._wid, self.margen, self.margen_din,
                                     self.diag, W, H, oc, oax, obb, K,
                                     blc, blax, blbb, B, rx, roff, rlen, rlw, R,
                                     res_maxlen)
                if nm > 0:
                    maniobra = [(pm[i, 0], pm[i, 1], pm[i, 2]) for i in range(nm)]
                    base = self._reconstruir(padre_id, arista_id, cid, (sx, sy, sth))
                    self.motivo = "ok"
                    return base + maniobra

            if k >= self.k_max:
                continue

            dprev = delta_prev.get(cid, 0.0)
            _k_expand(x, y, th, v, k, acc, dl, ns, h, L,
                      self.v_max_c, self.v_rev, self._len, self._wid,
                      self.margen, self.margen_din, self.diag, W, H,
                      oc, oax, obb, K, blc, blax, blbb, B,
                      rx, roff, rlen, rlw, R, feas, osub, ov)

            for ai in range(A):
                if not feas[ai]:
                    continue
                delta = dl[ai]
                nv = ov[ai]
                nx = osub[ai, ns - 1, 0]
                ny = osub[ai, ns - 1, 1]
                nth = osub[ai, ns - 1, 2]
                nk = k + ns
                # COSTE = TIEMPO. Una ruta más larga acumula más pasos y sale peor
                # por sí sola; no se penaliza girar salvo desempates ε.
                ng = g + self.dt
                if nv < 0:
                    ng += 0.6 * self.dt               # marcha atrás: maniobra indeseada
                if abs(nv) < 1e-3 and acc[ai] <= 0.0:
                    ng += 0.20 * self.dt              # pararse sin motivo cuesta tiempo
                ng += 0.10 * self.dt * abs(delta) / dmax          # ε: prefiere ir recto
                ng += 0.08 * self.dt * abs(delta - dprev)         # ε: sin temblor
                key = self._clave(nx, ny, nth, nv, nk)
                if ng < mejor_g.get(key, float("inf")):
                    mejor_g[key] = ng
                    nid += 1
                    padre_id[nid] = cid
                    arista_id[nid] = osub[ai].copy()
                    delta_prev[nid] = delta
                    hh = self._h_time(nx, ny)
                    heapq.heappush(abierto,
                                   (ng + self.peso_h * hh, nid, nx, ny, nth, nv, nk, ng))

        # Frontera vacía → no existe ruta (definitivo). Si quedaban nodos, se
        # alcanzó el techo de expansiones → resultado inconcluso.
        self.motivo = "sin_ruta" if not abierto else "limite"
        return None

    def _reconstruir(self, padre_id, arista_id, cid, inicio):
        tramos = []
        while cid in padre_id:
            tramos.append(arista_id[cid])
            cid = padre_id[cid]
        tramos.reverse()
        traj = [(inicio[0], inicio[1], inicio[2])]
        for t in tramos:
            for r in t:
                traj.append((r[0], r[1], r[2]))
        return traj


def warmup():
    """Fuerza la compilación de los kernels con datos mínimos, para que la primera
    planificación no pague ese coste. Con la caché en disco, a partir de la 2ª
    ejecución del programa es casi instantáneo."""
    oc, oax, obb = _pack_polys([obb_corners(5, 5, 0.3, 2.0, 1.0)])
    eb, ebb = _EMPTY_C, _EMPTY_BB
    rx = np.array([[5.0, 5.0, 0.0]]); roff = np.array([0], np.int64)
    rlen = np.array([1], np.int64); rlw = np.array([[1.3, 0.7]])
    acc = np.array([1.0, 0.0]); dl = np.array([0.1, -0.1])
    feas = np.empty(2, np.int8); osub = np.empty((2, 4, 3)); ov = np.empty(2)
    _k_expand(2.0, 2.0, 0.0, 0.0, 0, acc, dl, 4, 0.1, 0.7,
              2.5, 1.25, 1.3, 0.7, 0.1, 0.15, 1.48, W, H,
              oc, oax, obb, 1, eb, eb, ebb, 0, rx, roff, rlen, rlw, 1, feas, osub, ov)
    _k_tiro(2.0, 2.0, 0.0, 0.0, 0, 6.0, 6.0, 0.0, 1, 0.2,
            0.7, 0.6, 2.0, 2.5, 1.0, 1.6, 0.45, 0.1,
            1.3, 0.7, 0.1, 0.15, 1.48, W, H, oc, oax, obb, 1,
            eb, eb, ebb, 0, rx, roff, rlen, rlw, 1)
    _k_aparca(2.0, 2.0, 0.0, 0, 1, 1.6, 1.0, rx, roff, rlen, rlw, 1, 1.48, 0.15)
    _k_maniobra(2.0, 2.0, 0.0, 0, 6.0, 6.0, 0.0, 0.2,
                0.7, 0.6, 2.0, 2.5, 1.0, 0.2, 0.45, 0.1,
                1.3, 0.7, 0.1, 0.15, 1.48, W, H, oc, oax, obb, 1,
                eb, eb, ebb, 0, rx, roff, rlen, rlw, 1, 1)
    _k_occ(4, 4, 0.5, 0.45, W, H, oc, oax, obb, 1)


# --------------------------------------------------------------------------- #
# OPTIMIZACIÓN DE FLOTA: órdenes de planificación candidatos y evaluación
# --------------------------------------------------------------------------- #
# La planificación cooperativa procesa los vehículos en un ORDEN: cada uno
# respeta las trayectorias ya comprometidas. Distintos órdenes dan soluciones
# distintas; la capa de optimización explora varios y elige la mejor:
#
#   · "secuencial"  — un único orden DETERMINISTA, sin exploración: se ordena por
#                     (grupo, prioridad) ascendente y, a igualdad, por la posición
#                     en la lista. Rápido y predecible.
#   · "global"      — todos los vehículos forman un solo grupo y se prueban
#                     muchos órdenes (todas las permutaciones si son pocos;
#                     órdenes heurísticos + barajados si son muchos). Gana la
#                     solución con menos fallos y menor tiempo total de flota.
#   · "prioridades" — los vehículos se reparten en buckets por (grupo, prioridad)
#                     y estos se ordenan de menor a mayor (grupo 1 y prioridad 1
#                     van primero). Cada bucket se resuelve por completo antes que
#                     el siguiente; DENTRO de un bucket (mismo grupo y misma
#                     prioridad) el orden se optimiza GLOBALMENTE. Así, un grupo
#                     con todos la misma prioridad va «en bloque pero igual entre
#                     sí», y con prioridades distintas va «en ese orden». El número
#                     de prioridad es local al grupo: puede repetirse entre grupos.
PENAL_FALLO = 1.0e6     # coste ficticio de un vehículo sin ruta al comparar órdenes


def _ordenes_de_grupo(idxs, vehiculos, max_perm):
    """Órdenes candidatos para un grupo de índices. Con grupos pequeños se
    enumeran todas las permutaciones; con grandes, órdenes heurísticos (por
    distancia a la meta, ascendente y descendente) más barajados aleatorios."""
    idxs = list(idxs)
    if len(idxs) <= 1:
        return [tuple(idxs)]
    if math.factorial(len(idxs)) <= max_perm:
        return [tuple(p) for p in itertools.permutations(idxs)]

    def dist(i):
        v = vehiculos[i]
        return math.hypot(v.meta[0] - v.inicio[0], v.meta[1] - v.inicio[1])
    vistos = set()
    ordenes = []
    for cand in (tuple(idxs),
                 tuple(sorted(idxs, key=dist)),
                 tuple(sorted(idxs, key=dist, reverse=True))):
        if cand not in vistos:
            vistos.add(cand)
            ordenes.append(cand)
    rng = random.Random(12345)          # barajados reproducibles
    while len(ordenes) < max_perm:
        cand = idxs[:]
        rng.shuffle(cand)
        cand = tuple(cand)
        if cand not in vistos:
            vistos.add(cand)
            ordenes.append(cand)
    return ordenes


def generar_ordenes(vehiculos, indices, modo, max_cand=24):
    """Lista de órdenes candidatos (tuplas de índices sobre 'vehiculos') según el
    modo de optimización, limitada a max_cand candidatos."""
    indices = list(indices)
    if modo == "secuencial":
        # Un ÚNICO orden determinista: grupo prioritario primero (menor número),
        # y dentro de cada grupo/prioridad se respeta la posición en la lista
        # (sorted es estable). No explora combinaciones.
        orden = sorted(indices, key=lambda i: (vehiculos[i].grupo,
                                               vehiculos[i].prioridad))
        return [tuple(orden)]
    if len(indices) <= 1:
        return [tuple(indices)]
    if modo == "global":
        grupos = [indices]
    else:                                # "prioridades": buckets (grupo, prioridad)
        por_clave = {}                   # ordenados de menor a mayor → primero
        for i in indices:
            v = vehiculos[i]
            por_clave.setdefault((v.grupo, v.prioridad), []).append(i)
        grupos = [por_clave[k] for k in sorted(por_clave)]
    # Reparte el presupuesto de candidatos entre los grupos y combina.
    por_grupo = max(2, int(round(max_cand ** (1.0 / len(grupos)))))
    listas = [_ordenes_de_grupo(g, vehiculos, por_grupo) for g in grupos]
    ordenes = []
    for combo in itertools.islice(itertools.product(*listas), max_cand):
        orden = []
        for parte in combo:
            orden.extend(parte)
        ordenes.append(tuple(orden))
    return ordenes


def bloqueos_metas(vehiculos, indices, excepto):
    """OBB cuadrados en las metas ajenas, para que nadie planifique aparcar
    donde otro vehículo debe aparcar."""
    bloq = []
    for j in indices:
        if j == excepto:
            continue
        veh = vehiculos[j]
        mx, my = veh.meta
        poly = obb_corners(mx, my, 0.0, veh.diag, veh.diag)
        bloq.append((poly, (mx, my, veh.diag / 2)))
    return bloq


def planificar_orden(planificador, vehiculos, orden, reservas_base,
                     inicios=None, tick_veh=None, deadline_dur=None):
    """Planifica cooperativamente los vehículos de 'orden' (índices) sobre las
    reservas base (p. ej. vehículos aparcados de misiones anteriores).

    'inicios' (dict idx → pose) permite arrancar cada vehículo desde una pose
    distinta de su inicio nominal (misiones nuevas desde la plaza actual).

    'deadline_dur', si se indica, es el presupuesto de tiempo (segundos) POR
    VEHÍCULO: se renueva justo antes de planificar cada uno. Sin esto, un
    'planificador.deadline' fijado por el llamador ANTES de la llamada queda
    compartido entre todos los vehículos del orden, de modo que los primeros
    (más fáciles, o simplemente los primeros en la cola) pueden agotar el
    plazo entero y dejar a los últimos sin apenas tiempo para buscar —el
    síntoma es que, con varios vehículos, solo los primeros consiguen ruta y
    el resto se queda "aparcado" aunque exista solución.

    Devuelve (trayectorias, motivos, coste): dicts idx → traj / motivo, y el
    coste total de flota (suma de duraciones + penalización por fallo)."""
    reservas = reservas_base.copia()
    trays = {}
    motivos = {}
    coste = 0.0
    for pos, idx in enumerate(orden):
        veh = vehiculos[idx]
        if tick_veh is not None:
            tick_veh(pos, idx)
        planificador.bloqueos = bloqueos_metas(vehiculos, orden, excepto=idx)
        if deadline_dur is not None:
            planificador.deadline = time.perf_counter() + deadline_dur
        ini = inicios.get(idx) if inicios else None
        traj = planificador.planificar(veh, reservas, inicio=ini)
        motivos[idx] = planificador.motivo
        if traj is not None and len(traj) >= 2:
            trays[idx] = traj
            reservas.add(traj, veh.length, veh.width)
            coste += (len(traj) - 1) * DT
        elif traj:
            trays[idx] = [traj[0], traj[0]]
            reservas.add(trays[idx], veh.length, veh.width)
        else:
            trays[idx] = None
            coste += PENAL_FALLO
    return trays, motivos, coste


# --------------------------------------------------------------------------- #
# EVALUACIÓN MULTINÚCLEO de los órdenes candidatos
# --------------------------------------------------------------------------- #
# Cada orden candidato es una planificación cooperativa INDEPENDIENTE: parte de
# las mismas reservas base y produce su propio (trays, motivos, coste). Por eso
# se reparten los órdenes entre varios procesos —uno por núcleo utilizable— y se
# elige el mejor, exactamente igual que en el bucle secuencial pero en paralelo.
# Se usa el arranque "spawn" (idéntico en Apple/Intel/AMD, Windows y Linux) y la
# caché en disco de Numba, así que cada proceso sólo compila la primera vez.


def nucleos_disponibles():
    """Núcleos realmente utilizables. Respeta la afinidad/cgroups donde exista
    (Linux); si no, recurre al recuento total de CPUs."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


# Estado propio de cada proceso worker: se construye UNA vez (en el inicializador)
# y se reutiliza para todos los órdenes que le toquen a ese proceso.
_WK = {}


def _wk_init(entorno, vehiculos, reservas_base, calidad, inicios, cola, contador):
    """Inicializador de cada proceso: compila los núcleos (caché en disco), crea
    su propio planificador y le asigna un identificador de núcleo correlativo."""
    warmup()
    pl = Planificador(entorno)
    pl.configurar_calidad(calidad)
    with contador.get_lock():
        wid = contador.value
        contador.value += 1
    _WK.update(pl=pl, vehiculos=vehiculos, reservas=reservas_base,
               inicios=inicios, cola=cola, wid=wid, ult=0.0)


def _wk_eval(args):
    """Evalúa UN orden candidato en el proceso worker y devuelve su resultado.
    Publica el progreso (nodos del vehículo en curso) en la cola compartida."""
    orden, max_exp, deadline_dur = args
    pl = _WK["pl"]
    cola = _WK["cola"]
    wid = _WK["wid"]
    pl.max_exp = max_exp
    pl.deadline = time.perf_counter() + deadline_dur

    def tick(expand):
        ahora = time.perf_counter()
        if ahora - _WK["ult"] >= 0.08:          # limita el tráfico entre procesos
            _WK["ult"] = ahora
            try:
                cola.put_nowait((wid, expand))
            except Exception:
                pass

    pl.tick = tick
    try:
        trays, motivos, coste = planificar_orden(
            pl, _WK["vehiculos"], orden, _WK["reservas"], inicios=_WK["inicios"])
    finally:
        pl.tick = None
    fallos = sum(1 for i in orden if trays[i] is None)
    try:
        cola.put_nowait((wid, 0))               # ese núcleo queda libre
    except Exception:
        pass
    return orden, fallos, coste, trays, motivos


# --------------------------------------------------------------------------- #
# Entrada de texto: lista de diccionarios de características
# --------------------------------------------------------------------------- #
# Claves admitidas por vehículo (los ángulos SIEMPRE en grados en el texto):
#
#   id              identificador (entero o texto). Obligatorio para reutilizar.
#   inicio          (x, y) punto inicial en metros
#   giro_inicial    orientación inicial en grados (si falta: mirando a la meta)
#   meta            (x, y) punto destino en metros
#   angulo_llegada  ángulo exigido al llegar, en grados; "libre" o None → sin
#                   exigencia (si falta: la dirección inicio→meta)
#   largo, ancho    dimensiones del vehículo en metros
#   v_max           velocidad máxima (m/s)
#   a_max           aceleración/frenado máximos (m/s²)
#   giro_max        ángulo máximo de dirección en grados (capacidad de giro)
#   grupo           grupo de prioridad (entero); menor → se planifica antes. Un
#                   grupo se resuelve entero antes que el siguiente.
#   prioridad       prioridad DENTRO del grupo (entero); menor → antes. Mismos
#                   grupo y prioridad → optimización global entre ellos. El
#                   número es local al grupo (puede repetirse en otros grupos).
#
# Ejemplo:
#   [{"id": 1, "inicio": (3, 3), "giro_inicial": 0, "meta": (36, 20),
#     "angulo_llegada": 90, "largo": 1.6, "ancho": 0.8, "v_max": 3.0,
#     "a_max": 1.2, "giro_max": 33, "grupo": 1, "prioridad": 1},
#    {"id": 2, "inicio": (36, 3), "meta": (4, 20), "grupo": 2, "prioridad": 1}]
CLAVES_VEH = {"id", "inicio", "giro_inicial", "meta", "angulo_llegada",
              "largo", "ancho", "v_max", "a_max", "giro_max", "grupo", "prioridad"}


def _como_punto(valor, clave):
    try:
        x, y = float(valor[0]), float(valor[1])
    except (TypeError, ValueError, IndexError):
        raise ValueError(f"'{clave}' debe ser un punto (x, y); recibido: {valor!r}")
    if not (0.0 <= x <= W and 0.0 <= y <= H):
        raise ValueError(f"'{clave}' = ({x:g}, {y:g}) queda fuera del mundo "
                         f"{W:g}×{H:g} m.")
    return (x, y)


def _como_num(valor, clave, minimo=None, maximo=None):
    try:
        v = float(valor)
    except (TypeError, ValueError):
        raise ValueError(f"'{clave}' debe ser numérico; recibido: {valor!r}")
    if minimo is not None and v < minimo:
        raise ValueError(f"'{clave}' = {v:g} es menor que el mínimo {minimo:g}.")
    if maximo is not None and v > maximo:
        raise ValueError(f"'{clave}' = {v:g} supera el máximo {maximo:g}.")
    return v


def parsear_especificaciones(texto):
    """Convierte la entrada de texto en una lista de diccionarios NORMALIZADOS
    (ángulos en radianes, números validados). Lanza ValueError con un mensaje
    claro si algo no cuadra. Acepta un diccionario suelto o una lista de ellos."""
    texto = texto.strip()
    if not texto:
        raise ValueError("La entrada de texto está vacía.")
    try:
        datos = ast.literal_eval(texto)
    except (ValueError, SyntaxError) as e:
        raise ValueError(f"El texto no es una lista de diccionarios válida: {e}")
    if isinstance(datos, dict):
        datos = [datos]
    if not isinstance(datos, (list, tuple)) or not datos:
        raise ValueError("Se esperaba una lista de diccionarios (uno por vehículo).")

    especs = []
    ids_vistos = set()
    for i, d in enumerate(datos):
        if not isinstance(d, dict):
            raise ValueError(f"El elemento {i + 1} no es un diccionario: {d!r}")
        extranas = set(d) - CLAVES_VEH
        if extranas:
            raise ValueError(f"Vehículo {i + 1}: claves desconocidas {sorted(extranas)}. "
                             f"Admitidas: {sorted(CLAVES_VEH)}")
        e = {}
        e["id"] = d.get("id", i + 1)
        if e["id"] in ids_vistos:
            raise ValueError(f"El id {e['id']!r} está repetido.")
        ids_vistos.add(e["id"])
        if "inicio" in d:
            e["inicio"] = _como_punto(d["inicio"], "inicio")
        if "meta" in d:
            e["meta"] = _como_punto(d["meta"], "meta")
        if "giro_inicial" in d:
            e["giro_inicial"] = math.radians(_como_num(d["giro_inicial"], "giro_inicial"))
        if "angulo_llegada" in d:
            v = d["angulo_llegada"]
            if v is None or (isinstance(v, str) and v.strip().lower() == "libre"):
                e["angulo_llegada"] = None
            else:
                e["angulo_llegada"] = math.radians(_como_num(v, "angulo_llegada"))
        if "largo" in d:
            e["largo"] = _como_num(d["largo"], "largo", 0.4, 6.0)
        if "ancho" in d:
            e["ancho"] = _como_num(d["ancho"], "ancho", 0.2, 4.0)
        if "largo" in e and "ancho" in e and e["ancho"] > e["largo"]:
            raise ValueError(f"Vehículo {e['id']!r}: el ancho ({e['ancho']:g}) no "
                             f"puede superar al largo ({e['largo']:g}).")
        if "v_max" in d:
            e["v_max"] = _como_num(d["v_max"], "v_max", 0.2, 12.0)
        if "a_max" in d:
            e["a_max"] = _como_num(d["a_max"], "a_max", 0.1, 8.0)
        if "giro_max" in d:
            e["giro_max"] = math.radians(_como_num(d["giro_max"], "giro_max", 5.0, 60.0))
        if "grupo" in d:
            e["grupo"] = int(_como_num(d["grupo"], "grupo", 1))
        if "prioridad" in d:
            e["prioridad"] = int(_como_num(d["prioridad"], "prioridad", 1))
        especs.append(e)
    return especs


def espec_a_vehiculo(e, idx, entorno):
    """Crea un Vehiculo a partir de una especificación normalizada, validando que
    la pose inicial y la plaza de destino sean físicamente posibles."""
    if "inicio" not in e or "meta" not in e:
        raise ValueError(f"Vehículo {e['id']!r}: 'inicio' y 'meta' son obligatorios.")
    largo = e.get("largo", VEH_LEN)
    ancho = e.get("ancho", VEH_WID)
    ix, iy = e["inicio"]
    mx, my = e["meta"]
    th0 = e.get("giro_inicial", math.atan2(my - iy, mx - ix))
    # Ángulo de llegada: si no se indica, se exige la dirección inicio→meta
    # (siempre hay un ángulo concreto salvo que se pida "libre" explícitamente).
    if "angulo_llegada" in e:
        thm = e["angulo_llegada"]
    else:
        thm = math.atan2(my - iy, mx - ix)
    if not entorno.libre(ix, iy, th0, largo, ancho, margen=0.15):
        raise ValueError(f"Vehículo {e['id']!r}: el inicio ({ix:g}, {iy:g}) con "
                         f"orientación {math.degrees(th0):.0f}° no cabe (borde u "
                         f"obstáculo).")
    th_plaza = thm if thm is not None else 0.0
    if not entorno.libre(mx, my, th_plaza, largo, ancho, margen=0.15):
        raise ValueError(f"Vehículo {e['id']!r}: la plaza de destino ({mx:g}, {my:g})"
                         f"{'' if thm is None else f' con llegada a {math.degrees(thm):.0f}°'}"
                         f" no cabe (borde u obstáculo).")
    return Vehiculo(idx, (ix, iy, th0), (mx, my),
                    length=largo, width=ancho,
                    v_max=e.get("v_max", VEH_VMAX),
                    a_max=e.get("a_max", VEH_AMAX),
                    giro_max=e.get("giro_max"),
                    meta_th=thm,
                    grupo=e.get("grupo", 1),
                    prioridad=e.get("prioridad", 1),
                    vid=e["id"])


def especs_a_texto(especs):
    """Serializa una lista de diccionarios de especificación a texto editable
    (un vehículo por línea), reutilizable tal cual como entrada del programa."""
    return "[\n" + "".join(" " + repr(e) + ",\n" for e in especs) + "]\n"


TEXTO_EJEMPLO = (
    "[\n"
    ' {"id": 1, "inicio": (3, 3),  "giro_inicial": 0,  "meta": (36, 20),\n'
    '  "angulo_llegada": 90, "largo": 1.6, "ancho": 0.8, "v_max": 3.0,\n'
    '  "a_max": 1.2, "giro_max": 33, "grupo": 1, "prioridad": 1},\n'
    ' {"id": 2, "inicio": (36, 3), "meta": (4, 20), "grupo": 1, "prioridad": 1},\n'
    ' {"id": 3, "inicio": (20, 2), "meta": (20, 21), "angulo_llegada": "libre",\n'
    '  "grupo": 2, "prioridad": 1},\n'
    "]\n"
)


# --------------------------------------------------------------------------- #
# Ejecución: reproduce las trayectorias coordinadas (ya libres de colisión)
# --------------------------------------------------------------------------- #
def construir_frames(vehiculos):
    """Muestrea todas las trayectorias a intervalos DT y produce los fotogramas
    {idx: (x,y,theta)} para la reproducción fluida."""
    if not vehiculos:
        return []
    T = max(v.duracion for v in vehiculos)
    frames = []
    t = 0.0
    while t <= T + 1e-9:
        frames.append({v.idx: v.pose_en_tiempo(t) for v in vehiculos})
        t += DT
    frames.append({v.idx: v.pose_en_tiempo(T) for v in vehiculos})
    return frames


# --------------------------------------------------------------------------- #
# Interfaz gráfica
# --------------------------------------------------------------------------- #
CAP_COMPLETO = 6_000_000     # techo de expansiones para agotar la búsqueda


class App:
    def __init__(self, root):
        import tkinter as tk
        self.tk = tk
        self.root = root
        root.title("Optimizador de flotas · espacio continuo + SAT  "
                   "[núcleo nativo Numba]")
        root.resizable(False, False)

        self.env = Entorno()
        self.planificador = Planificador(self.env)
        self.vehiculos = []
        self.frames = []
        self.frame = 0
        self.anim_id = None
        self.reproduciendo = False
        self._warmed = False

        self._ocupado = False
        self._plan_msg = ""

        self._construir_ui()
        self.env.generar(0.0)
        self._actualizar_densidad()
        self._dibujar_estatico()

    # ------------------------------ robustez ----------------------------- #
    def _seguro(self, fn):
        """Envuelve un callback para que ninguna combinación de pulsaciones pueda
        colgar la aplicación: ignora acciones reentrantes, absorbe los errores de
        Tk al cerrar y muestra cualquier excepción en un diálogo."""
        tk = self.tk
        from tkinter import messagebox
        def envuelto(*args, **kwargs):
            if self._ocupado:
                return None
            try:
                return fn(*args, **kwargs)
            except tk.TclError:
                return None
            except Exception as e:  # noqa: BLE001
                try:
                    messagebox.showerror("Error inesperado",
                                         f"{type(e).__name__}: {e}")
                except Exception:
                    pass
                return None
        return envuelto

    # -------------------------------- UI --------------------------------- #
    def _construir_ui(self):
        tk = self.tk
        from tkinter import ttk
        # Estilo para destacar ligeramente el botón principal (Calcular y simular).
        estilo = ttk.Style()
        estilo.configure("Destacado.TButton", font=("", 12, "bold"))
        cont = ttk.Frame(self.root, padding=8)
        cont.grid(row=0, column=0)

        panel = ttk.Frame(cont, padding=(0, 0, 12, 0))
        panel.grid(row=0, column=0, rowspan=2, sticky="n")

        fila = [0]

        def sep():
            ttk.Separator(panel, orient="horizontal").grid(
                row=fila[0], column=0, columnspan=2, sticky="ew", pady=6)
            fila[0] += 1

        ttk.Label(panel, text="Parámetros", font=("", 13, "bold")).grid(
            row=fila[0], column=0, columnspan=2, sticky="w", pady=(0, 4))
        fila[0] += 1
        self.e_num = self._campo(panel, fila, "Nº de vehículos:", "4")
        self.e_dens = self._campo(panel, fila, "Densidad obstáculos (0-1):", "0.0")

        self.calidad = tk.IntVar(value=3)
        self.calidad_txt = tk.StringVar()
        ttk.Label(panel, textvariable=self.calidad_txt, width=38,
                  anchor="w").grid(
            row=fila[0], column=0, columnspan=2, sticky="w", pady=(4, 0))
        fila[0] += 1
        tk.Scale(panel, from_=1, to=5, resolution=1, tickinterval=1,
                 orient="horizontal", variable=self.calidad, showvalue=False,
                 command=self._calidad_cambia).grid(
            row=fila[0], column=0, columnspan=2, sticky="ew")
        fila[0] += 1
        self._calidad_cambia()

        sep()
        ttk.Label(panel, text="Vehículos", font=("", 11, "bold")).grid(
            row=fila[0], column=0, columnspan=2, sticky="w")
        fila[0] += 1
        self.modo = tk.StringVar(value="aleatorio")
        for txt, val in (("Aleatorios (todo al azar)", "aleatorio"),
                         ("Manuales (caja de texto inferior)", "manual")):
            ttk.Radiobutton(panel, text=txt, variable=self.modo,
                            value=val).grid(row=fila[0], column=0,
                                            columnspan=2, sticky="w")
            fila[0] += 1
        ttk.Button(panel, text="Generar / cargar vehículos",
                   command=self._seguro(self.generar_posiciones)).grid(
            row=fila[0], column=0, columnspan=2, sticky="ew", pady=(4, 0))
        fila[0] += 1

        sep()
        ttk.Label(panel, text="Optimización de flota", font=("", 11, "bold")).grid(
            row=fila[0], column=0, columnspan=2, sticky="w")
        fila[0] += 1
        self.opt = tk.StringVar(value="secuencial")
        for txt, val in (("Global (explora órdenes, mejor total)", "global"),
                         ("Prioridades personalizadas", "prioridades"),
                         ("Secuencial (grupo+prioridad, sin explorar)", "secuencial")):
            ttk.Radiobutton(panel, text=txt, variable=self.opt,
                            value=val).grid(row=fila[0], column=0,
                                            columnspan=2, sticky="w")
            fila[0] += 1
        self.e_cand = self._campo(panel, fila, "Órdenes a explorar (máx):", "12")

        sep()
        # Botón principal destacado.
        ttk.Button(panel, text="▶  Calcular y simular", style="Destacado.TButton",
                   command=self._seguro(self.calcular_y_simular)).grid(
            row=fila[0], column=0, columnspan=2, sticky="ew", pady=1)
        fila[0] += 1
        for txt, cmd in (("⟳  Nuevas rutas para ids del texto", self.aplicar_nuevas_rutas),
                         ("↺  Reproducir de nuevo", self.reproducir),
                         ("⏸  Pausar / reanudar", self.pausar),
                         ("⟲  Reiniciar", self.reiniciar)):
            ttk.Button(panel, text=txt, command=self._seguro(cmd)).grid(
                row=fila[0], column=0, columnspan=2, sticky="ew", pady=1)
            fila[0] += 1

        sep()
        ttk.Label(panel, text="Mapa", font=("", 11, "bold")).grid(
            row=fila[0], column=0, columnspan=2, sticky="w")
        fila[0] += 1
        for txt, cmd in (("Nuevo mapa aleatorio", self.nuevo_mapa),
                         ("Importar imagen como mapa…", self.importar_imagen),
                         ("Guardar mapa actual…", self.guardar_mapa_actual),
                         ("Cargar mapa guardado…", self.cargar_mapa_guardado)):
            ttk.Button(panel, text=txt, command=self._seguro(cmd)).grid(
                row=fila[0], column=0, columnspan=2, sticky="ew", pady=1)
            fila[0] += 1

        sep()
        ttk.Button(panel, text="✕  Salir", command=self.root.destroy).grid(
            row=fila[0], column=0, columnspan=2, sticky="ew", pady=(1, 0))
        fila[0] += 1

        self.canvas = tk.Canvas(cont, width=int(W * SCALE), height=int(H * SCALE),
                                bg=COL_FONDO, highlightthickness=2,
                                highlightbackground=COL_BORDE)
        self.canvas.grid(row=0, column=1)
        self.canvas.bind("<Motion>", self._on_canvas_motion)
        self.canvas.bind("<Leave>", self._on_canvas_leave)
        self._cursor_info = None

        caja = ttk.Frame(cont)
        caja.grid(row=1, column=1, sticky="nsew", pady=(6, 0))
        ttk.Label(caja, text="Vehículos manuales / nuevas rutas "
                             "(lista de diccionarios; ángulos en grados):").grid(
            row=0, column=0, sticky="w")
        self.texto = tk.Text(caja, height=8, width=int(W * SCALE / 8),
                             font=("Menlo", 10), undo=True)
        self.texto.grid(row=1, column=0, sticky="ew")
        self.texto.insert("1.0", TEXTO_EJEMPLO)

        self.estado = tk.StringVar(value="Listo. Ajusta parámetros y genera vehículos.")
        ttk.Label(cont, textvariable=self.estado, relief="sunken",
                  anchor="w", padding=4, width=1).grid(
            row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))

    def _campo(self, panel, fila, etiqueta, valor):
        from tkinter import ttk
        ttk.Label(panel, text=etiqueta).grid(row=fila[0], column=0, sticky="w", pady=1)
        e = ttk.Entry(panel, width=7)
        e.insert(0, valor)
        e.grid(row=fila[0], column=1, sticky="e", pady=1)
        fila[0] += 1
        return e

    def _calidad_cambia(self, *_):
        nivel = int(round(float(self.calidad.get())))
        self.calidad_txt.set(f"Calidad de ruta (1 rápida ⟷ 5 máxima):  {nivel}")
        self.planificador.configurar_calidad(nivel)

    def _on_canvas_motion(self, ev):
        x_m = ev.x / SCALE
        y_m = ev.y / SCALE
        texto = f"x: {x_m:.1f}, y: {y_m:.1f}"
        
        tx = ev.x + 12
        ty = ev.y + 12
        if ev.x > (W * SCALE) - 90:
            tx = ev.x - 80
        if ev.y > (H * SCALE) - 30:
            ty = ev.y - 20
        
        if self._cursor_info is None:
            rect = self.canvas.create_rectangle(0, 0, 1, 1, fill="white", outline="black", tags="tooltip")
            txt = self.canvas.create_text(tx, ty, text=texto, fill="black", anchor="nw", font=("", 8), tags="tooltip")
            self._cursor_info = (rect, txt)
        else:
            rect, txt = self._cursor_info
            self.canvas.itemconfigure(txt, text=texto)
            self.canvas.coords(txt, tx, ty)
            self.canvas.tag_raise(rect)
            self.canvas.tag_raise(txt)
            
        rect, txt = self._cursor_info
        bbox = self.canvas.bbox(txt)
        if bbox:
            self.canvas.coords(rect, bbox[0]-2, bbox[1]-1, bbox[2]+2, bbox[3]+1)

    def _on_canvas_leave(self, ev):
        if self._cursor_info is not None:
            self.canvas.delete("tooltip")
            self._cursor_info = None

    def _params(self):
        from tkinter import messagebox
        try:
            n = int(self.e_num.get())
            dens = float(self.e_dens.get())
        except ValueError:
            messagebox.showerror("Error", "Los parámetros deben ser numéricos.")
            return None
        if not (1 <= n <= len(PALETA)):
            messagebox.showerror("Error", f"Nº de vehículos entre 1 y {len(PALETA)}.")
            return None
        return n, max(0.0, min(1.0, dens))

    def _max_cand(self):
        try:
            return max(1, min(120, int(self.e_cand.get())))
        except ValueError:
            return 12

    # ------------------------------- mapa -------------------------------- #
    def _actualizar_densidad(self):
        """La densidad de obstáculos solo tiene sentido en mapas aleatorios; con un
        mapa importado de imagen se deshabilita el campo."""
        importado = self.env.rects is not None
        try:
            self.e_dens.configure(state=("disabled" if importado else "normal"))
        except self.tk.TclError:
            pass

    def _mapa_cambiado(self, msg):
        self._detener()
        self.vehiculos, self.frames = [], []
        self.frame = 0
        self._actualizar_densidad()
        self.estado.set(msg)
        self._dibujar_estatico()

    def nuevo_mapa(self):
        p = self._params()
        dens = p[1] if p else 0.0
        self.env.generar(dens)
        self._mapa_cambiado("Nuevo mapa aleatorio. Genera vehículos.")

    def importar_imagen(self):
        from tkinter import filedialog, messagebox, simpledialog
        ruta = filedialog.askopenfilename(
            title="Imagen a convertir en mapa (negro = obstáculo)",
            filetypes=[("Imágenes", "*.png *.gif *.ppm *.pgm *.jpg *.jpeg *.bmp *.tif *.tiff"),
                       ("Todos", "*.*")])
        if not ruta:
            return
        self._ocupado = True
        try:
            self.estado.set("Convirtiendo imagen en mapa…")
            self.root.update()
            rects = imagen_a_rects(ruta)
        except Exception as e:  # noqa: BLE001
            self._ocupado = False
            messagebox.showerror("Imagen no válida",
                                 f"No se pudo leer/convertir la imagen:\n{e}\n\n"
                                 "Sin Pillow instalado solo se admiten PNG/GIF/PPM "
                                 "(pip install pillow para el resto).")
            return
        self._ocupado = False
        if len(rects) == 0:
            messagebox.showwarning("Mapa vacío",
                                   "La imagen no contiene píxeles oscuros: el mapa "
                                   "queda completamente libre.")
        nombre = os.path.splitext(os.path.basename(ruta))[0]
        self.env.desde_rects(rects, nombre)
        self._mapa_cambiado(f"Mapa importado de «{nombre}» "
                            f"({len(rects)} bloques). Genera vehículos.")
        if messagebox.askyesno("Guardar mapa",
                               "¿Guardar este mapa en la carpeta «mapas/» para "
                               "poder reutilizarlo en otras ejecuciones?"):
            nom = simpledialog.askstring("Nombre del mapa",
                                         "Nombre con el que guardarlo:",
                                         initialvalue=nombre)
            if nom:
                ruta_j = guardar_mapa(nom, rects)
                self.estado.set(f"Mapa guardado en {ruta_j}")

    def guardar_mapa_actual(self):
        from tkinter import messagebox, simpledialog
        if self.env.rects is None:
            messagebox.showinfo("Solo mapas importados",
                                "Se guardan los mapas creados desde una imagen. "
                                "Importa una imagen primero.")
            return
        nom = simpledialog.askstring("Nombre del mapa",
                                     "Nombre con el que guardarlo:",
                                     initialvalue=self.env.origen)
        if nom:
            ruta_j = guardar_mapa(nom, self.env.rects)
            self.estado.set(f"Mapa guardado en {ruta_j}")

    def cargar_mapa_guardado(self):
        from tkinter import filedialog, messagebox
        os.makedirs(MAPAS_DIR, exist_ok=True)
        if not listar_mapas():
            messagebox.showinfo("Sin mapas guardados",
                                "Aún no hay mapas en la carpeta «mapas/». "
                                "Importa una imagen y guárdala primero.")
            return
        ruta = filedialog.askopenfilename(title="Mapa guardado",
                                          initialdir=MAPAS_DIR,
                                          filetypes=[("Mapa JSON", "*.json")])
        if not ruta:
            return
        try:
            rects = cargar_mapa(ruta)
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Mapa corrupto", f"No se pudo cargar:\n{e}")
            return
        nombre = os.path.splitext(os.path.basename(ruta))[0]
        self.env.desde_rects(rects, nombre)
        self._mapa_cambiado(f"Mapa «{nombre}» cargado "
                            f"({len(rects)} bloques). Genera vehículos.")

    # --------------------------- crear vehículos -------------------------- #
    def generar_posiciones(self):
        from tkinter import messagebox
        p = self._params()
        if not p:
            return
        n, _ = p
        self._detener()
        self.frames = []
        self.frame = 0
        modo = self.modo.get()

        if modo in ("manual", "personalizado"):
            try:
                especs = parsear_especificaciones(self.texto.get("1.0", "end"))
                self.vehiculos = [espec_a_vehiculo(e, i, self.env)
                                  for i, e in enumerate(especs)]
            except ValueError as e:
                messagebox.showerror("Entrada de texto no válida", str(e))
                return
            if len(self.vehiculos) > len(PALETA):
                messagebox.showerror("Demasiados vehículos",
                                     f"Máximo {len(PALETA)} vehículos.")
                self.vehiculos = []
                return
            self.estado.set(f"{len(self.vehiculos)} vehículos manuales "
                            "cargados. Pulsa «Calcular y simular».")
            self._dibujar_estatico()
            return

        # ALEATORIO: se genera una lista de diccionarios de longitud n (con TODO
        # al azar), se vuelca al cuadro de texto y se crean los vehículos a partir
        # de ese texto — el mismo camino que el modo manual. Así el usuario
        # ve y puede editar los vehículos generados.
        especs = self._aleatorios_spec(n)
        if especs is None:
            messagebox.showwarning("Sin espacio",
                "No se hallaron posiciones libres. Reduce vehículos/obstáculos.")
            return
        self.texto.delete("1.0", "end")
        self.texto.insert("1.0", especs_a_texto(especs))
        try:
            normal = parsear_especificaciones(self.texto.get("1.0", "end"))
            self.vehiculos = [espec_a_vehiculo(e, i, self.env)
                              for i, e in enumerate(normal)]
        except ValueError as e:
            messagebox.showerror("Entrada de texto no válida", str(e))
            return
        self.estado.set(f"{n} vehículos aleatorios generados en el texto "
                        "(edítalos si quieres). Pulsa «Calcular y simular».")
        self._dibujar_estatico()

    def _muestrear(self, length, width, otros, sep):
        for _ in range(4000):
            x = random.uniform(length, W - length)
            y = random.uniform(length, H - length)
            th = random.uniform(0, 2 * math.pi)
            if not self.env.libre(x, y, th, length, width, margen=0.3):
                continue
            if all(math.hypot(x - ox, y - oy) >= sep for ox, oy in otros):
                return (x, y, th)
        return None

    def _angulo_llegada(self, ix, iy, mx, my, length, width):
        """Ángulo de llegada CONCRETO y totalmente al azar para la plaza (mx, my).
        None si ninguna de las orientaciones al azar cabe."""
        candidatos = [random.uniform(0, 2 * math.pi) for _ in range(25)]
        for c in candidatos:
            if self.env.libre(mx, my, c, length, width, margen=0.25):
                return c
        return None

    def _aleatorios_spec(self, n):
        """Genera n especificaciones (diccionarios en GRADOS, como los del texto)
        con TODO al azar: tamaño, dinámica, capacidad de giro, pose inicial,
        destino, ángulo de llegada, grupo y prioridad. Devuelve la lista o None si
        no encuentra sitio libre para todos."""
        especs = []
        inicios, metas = [], []
        for i in range(n):
            largo = random.uniform(1.0, 1.8)
            ancho = largo * random.uniform(0.50, 0.62)
            sep = largo * 1.6
            ini = self._muestrear(largo, ancho, inicios, sep)
            met = self._muestrear(largo, ancho, metas, sep)
            if ini is None or met is None:
                return None
            ix, iy, ith = ini
            mx, my, _ = met
            thm = self._angulo_llegada(ix, iy, mx, my, largo, ancho)
            th0 = ith
            especs.append({
                "id": i + 1,
                "inicio": (round(ix, 2), round(iy, 2)),
                "giro_inicial": round(math.degrees(th0), 1),
                "meta": (round(mx, 2), round(my, 2)),
                "angulo_llegada": (round(math.degrees(thm), 1)
                                   if thm is not None else "libre"),
                "largo": round(largo, 2),
                "ancho": round(ancho, 2),
                "v_max": round(random.uniform(1.8, 3.2), 2),
                "a_max": round(random.uniform(0.8, 1.4), 2),
                "giro_max": round(random.uniform(28.0, 40.0), 1),
                "grupo": random.randint(1, 2),
                "prioridad": random.randint(1, 3),
            })
            inicios.append((ix, iy))
            metas.append((mx, my))          # solo (x, y): _muestrear espera pares
        return especs

    def _reubicar(self, veh):
        """Modo aleatorio: si no existe ruta, prueba con extremos nuevos."""
        sep = veh.length * 1.6
        otros_ini = [v.inicio[:2] for v in self.vehiculos if v.idx != veh.idx]
        otros_meta = [v.meta for v in self.vehiculos if v.idx != veh.idx]
        ini = self._muestrear(veh.length, veh.width, otros_ini, sep)
        met = self._muestrear(veh.length, veh.width, otros_meta, sep)
        if ini is None or met is None:
            return False
        ix, iy, ith = ini
        thm = self._angulo_llegada(ix, iy, met[0], met[1], veh.length, veh.width)
        th0 = ith
        veh.inicio = (ix, iy, th0)
        veh.meta = (met[0], met[1])
        veh.meta_th = thm
        return True



    # --------------------------- planificación --------------------------- #
    def calcular_y_simular(self):
        from tkinter import messagebox
        if not self.vehiculos:
            messagebox.showinfo("Faltan vehículos",
                                "Genera o carga primero todos los vehículos.")
            return
        self._detener()
        for v in self.vehiculos:
            v.traj = []
            v.mision_ok = False
        self._ocupado = True
        try:
            self._warm()
            indices = list(range(len(self.vehiculos)))
            base = Reservas()
            trays, motivos = self._planificar_flota(
                indices, base, inicios=None,
                reubicable=(self.modo.get() == "aleatorio"))
            self._aplicar_resultado(indices, trays, motivos, k_inicio=0)
        finally:
            self._ocupado = False

    def _warm(self):
        if not self._warmed:
            self.estado.set("Compilando núcleos nativos (solo la 1ª vez)…")
            self.root.update()
            warmup()
            self._warmed = True

    def _tick_plan(self, expand):
        self.estado.set(f"{self._plan_msg}  ({expand:,} nodos explorados)")
        self.root.update()

    def _planificar_flota(self, indices, reservas_base, inicios, reubicable):
        """Optimiza la flota: genera los órdenes candidatos según el modo elegido,
        evalúa cada uno con planificación cooperativa completa y se queda con el
        mejor (menos fallos; a igualdad, menor tiempo total). Los vehículos que
        sigan sin ruta en el mejor orden se rescatan al final agotando la búsqueda
        (y, si 'reubicable', probando extremos nuevos)."""
        modo_opt = self.opt.get()
        ordenes = generar_ordenes(self.vehiculos, indices, modo_opt,
                                  self._max_cand())
        pl = self.planificador
        pl.tick = self._tick_plan
        cap_prev = pl.max_exp
        n = len(indices)
        mejor = None            # (fallos, coste, trays, motivos, orden)
        try:
            explorar = len(ordenes) > 1
            # Con varios órdenes y varios núcleos, se evalúan EN PARALELO (un
            # proceso por núcleo). Con uno solo, o si el paralelo falla por
            # cualquier motivo, se usa el bucle secuencial de siempre.
            hecho_paralelo = False
            if explorar and nucleos_disponibles() >= 2 and n >= 3:
                try:
                    dur = max(10.0, 3.0 * n)
                    mejor = self._evaluar_ordenes_paralelo(
                        ordenes, reservas_base, inicios, cap_prev, dur)
                    hecho_paralelo = True
                    if mejor is None:            # ventana cerrada durante el cálculo
                        return {}, {}
                except Exception:
                    hecho_paralelo = False       # se recae al modo secuencial

            if not hecho_paralelo:
                mejor = None
                for oi, orden in enumerate(ordenes):
                    if not self.root.winfo_exists():
                        return {}, {}
                    etiqueta = (f"Orden {oi + 1}/{len(ordenes)} "
                                f"[{'·'.join(str(self.vehiculos[i].vid) for i in orden)}]"
                                if explorar else "Planificación cooperativa")

                    def tick_veh(pos, idx, _et=etiqueta):
                        self._plan_msg = (f"{_et} · vehículo {pos + 1}/{n} "
                                          f"(id {self.vehiculos[idx].vid})")
                        self.estado.set(self._plan_msg + "…")
                        self.root.update()

                    if explorar:
                        # Presupuesto por candidato: cap de calidad + límite
                        # temporal, para que explorar muchos órdenes siga siendo
                        # interactivo.
                        pl.max_exp = cap_prev
                        pl.deadline = time.perf_counter() + max(10.0, 3.0 * n)
                    else:
                        pl.max_exp = CAP_COMPLETO
                        pl.deadline = None
                    trays, motivos, coste = planificar_orden(
                        pl, self.vehiculos, orden, reservas_base,
                        inicios=inicios, tick_veh=tick_veh)
                    fallos = sum(1 for i in orden if trays[i] is None)
                    if mejor is None or (fallos, coste) < (mejor[0], mejor[1]):
                        mejor = (fallos, coste, trays, motivos, orden)
                    if fallos == 0 and not explorar:
                        break

            fallos, _, trays, motivos, orden = mejor
            # ---- rescate de los vehículos sin ruta en el mejor orden ---- #
            if fallos:
                pl.deadline = None
                pl.max_exp = CAP_COMPLETO
                reservas = reservas_base.copia()
                for i in orden:
                    if trays[i] is not None:
                        v = self.vehiculos[i]
                        reservas.add(trays[i], v.length, v.width)
                for i in orden:
                    if trays[i] is not None:
                        continue
                    veh = self.vehiculos[i]
                    self._plan_msg = (f"Rescate del vehículo id {veh.vid} "
                                      "(búsqueda exhaustiva)")
                    self.estado.set(self._plan_msg + "…")
                    self.root.update()
                    pl.bloqueos = bloqueos_metas(self.vehiculos, orden, excepto=i)
                    ini = inicios.get(i) if inicios else None
                    traj = pl.planificar(veh, reservas, inicio=ini)
                    motivos[i] = pl.motivo
                    intentos = 0
                    while (pl.motivo == "sin_ruta" and reubicable
                           and inicios is None and intentos < 12):
                        if not self._reubicar(veh):
                            break
                        pl.bloqueos = bloqueos_metas(self.vehiculos, orden, excepto=i)
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
            pl.tick = None
            pl.deadline = None
            pl.max_exp = cap_prev

    def _evaluar_ordenes_paralelo(self, ordenes, reservas_base, inicios,
                                  max_exp, dur):
        """Evalúa los órdenes candidatos repartidos entre los núcleos disponibles
        (un proceso por núcleo, hasta tantos como órdenes). Devuelve el mejor
        (fallos, coste, trays, motivos, orden), o None si se cierra la ventana.

        Mientras corre, muestra abajo cuántos núcleos están trabajando y los
        nodos que explora cada uno en el vehículo que planifica en ese momento."""
        ctx = multiprocessing.get_context("spawn")
        nucleos = nucleos_disponibles()
        workers = max(1, min(len(ordenes), nucleos))
        cola = ctx.Queue()
        contador = ctx.Value("i", 0)
        calidad = int(round(float(self.calidad.get())))
        nodos = {}                     # wid -> últimos nodos reportados
        mejor = None
        ex = ProcessPoolExecutor(
            max_workers=workers, mp_context=ctx, initializer=_wk_init,
            initargs=(self.env, self.vehiculos, reservas_base, calidad,
                      inicios, cola, contador))
        try:
            fut_idx = {ex.submit(_wk_eval, (orden, max_exp, dur)): oi
                       for oi, orden in enumerate(ordenes)}
            pendientes = set(fut_idx)
            total = len(pendientes)
            clave_mejor = None         # (fallos, coste, idx): desempata por índice
            while pendientes:
                if not self.root.winfo_exists():
                    return None
                while True:            # vaciar la cola de progreso
                    try:
                        wid, expand = cola.get_nowait()
                    except queue.Empty:
                        break
                    nodos[wid] = expand
                for f in [f for f in pendientes if f.done()]:
                    pendientes.discard(f)
                    try:
                        orden, fallos, coste, trays, motivos = f.result()
                    except Exception:
                        continue
                    clave = (fallos, coste, fut_idx[f])
                    if clave_mejor is None or clave < clave_mejor:
                        clave_mejor = clave
                        mejor = (fallos, coste, trays, motivos, orden)
                activos = min(workers, len(pendientes))
                detalle = "   ".join(f"núcleo {w + 1}: {nodos.get(w, 0):,}"
                                     for w in range(workers))
                self._plan_msg = (
                    f"Explorando {total} órdenes en paralelo · "
                    f"{activos}/{nucleos} núcleos activos "
                    f"({total - len(pendientes)}/{total} listos)")
                self.estado.set(f"{self._plan_msg}   ⟶   {detalle}")
                self.root.update()
                time.sleep(0.02)
        finally:
            try:
                ex.shutdown(wait=False, cancel_futures=True)
            except TypeError:          # cancel_futures: Python < 3.9
                ex.shutdown(wait=False)
            cola.close()
        return mejor

    def _aplicar_resultado(self, indices, trays, motivos, k_inicio):
        """Vuelca las trayectorias en los vehículos (desplazadas k_inicio pasos si
        son misiones nuevas), reconstruye los fotogramas e informa del resultado."""
        from tkinter import messagebox
        if not self.root.winfo_exists():
            return
        ok = 0; sin_ruta = 0; inconcluso = 0
        for i in indices:
            veh = self.vehiculos[i]
            traj = trays.get(i)
            if traj is None:
                veh.mision_ok = False
                if k_inicio == 0:
                    veh.traj = []
                if motivos.get(i) == "sin_ruta":
                    sin_ruta += 1
                else:
                    inconcluso += 1
                continue
            veh.mision_ok = True
            ok += 1
            if k_inicio == 0:
                veh.traj = traj
            else:
                base = veh.traj if veh.traj else [veh.inicio]
                ultima = base[-1]
                relleno = [ultima] * max(0, k_inicio - len(base))
                veh.traj = base + relleno + traj[1:]
            veh.dt_plan = DT

        self._dibujar_estatico()
        if ok == 0:
            if inconcluso:
                messagebox.showwarning("Búsqueda inconclusa",
                    "No se agotó la búsqueda dentro del techo de nodos: el "
                    "resultado es INCONCLUSO (podría existir ruta, pero el "
                    "espacio de búsqueda era demasiado grande).\nSube la "
                    "«Calidad de ruta», usa menos vehículos/obstáculos o cambia "
                    "las posiciones.")
            else:
                messagebox.showwarning("Sin ruta",
                    "No existe ruta para ningún vehículo: la búsqueda agotó todo "
                    "el espacio alcanzable sin llegar a la meta.\nLas posiciones "
                    "elegidas no tienen solución en este mapa.")
            self.estado.set("Sin rutas. Ajusta parámetros y reintenta.")
            return

        self.frames = construir_frames(self.vehiculos)
        avisos = []
        if sin_ruta:
            avisos.append(f"{sin_ruta} sin ruta (no existe)")
        if inconcluso:
            avisos.append(f"{inconcluso} inconcluso (techo de búsqueda)")
        aviso = f"  ⚠ {'; '.join(avisos)}" if avisos else ""
        total = sum(v.duracion for v in self.vehiculos if v.mision_ok)
        self.estado.set(f"{ok} rutas sin colisiones · tiempo total de flota "
                        f"{total:.1f} s ({len(self.frames)} fotogramas).{aviso}  "
                        "Reproduciendo…")
        self.frame = max(0, min(k_inicio, len(self.frames) - 1))
        self.reproduciendo = True
        self._anim()

    # ------------------- reutilización: nuevas rutas ---------------------- #
    def aplicar_nuevas_rutas(self):
        """Relee la entrada de texto como NUEVAS MISIONES: solo los vehículos cuyo
        'id' aparezca en la lista reciben ruta nueva, arrancando desde la plaza
        donde quedaron aparcados; el resto permanece quieto y es respetado como
        obstáculo. Las nuevas misiones comienzan cuando toda la flota ha terminado
        el movimiento anterior."""
        from tkinter import messagebox
        if not self.vehiculos or not self.frames:
            messagebox.showinfo("Nada que reasignar",
                                "Primero calcula y simula unas rutas; después "
                                "escribe las misiones nuevas en el texto.")
            return
        try:
            especs = parsear_especificaciones(self.texto.get("1.0", "end"))
        except ValueError as e:
            messagebox.showerror("Entrada de texto no válida", str(e))
            return

        por_id = {v.vid: v for v in self.vehiculos}
        lote = []
        try:
            for e in especs:
                if e["id"] not in por_id:
                    raise ValueError(f"No existe ningún vehículo con id {e['id']!r}. "
                                     f"Ids disponibles: {sorted(por_id)}")
                if "inicio" in e or "giro_inicial" in e:
                    raise ValueError(f"Vehículo {e['id']!r}: en una misión nueva no "
                                     "se cambia 'inicio'/'giro_inicial' (arranca "
                                     "desde donde quedó aparcado).")
                if "largo" in e or "ancho" in e:
                    raise ValueError(f"Vehículo {e['id']!r}: el tamaño del vehículo "
                                     "no cambia en una misión nueva.")
                if "meta" not in e:
                    raise ValueError(f"Vehículo {e['id']!r}: la misión nueva "
                                     "necesita 'meta'.")
                veh = por_id[e["id"]]
                mx, my = e["meta"]
                px, py, pth = veh.pose_final
                if "angulo_llegada" in e:
                    thm = e["angulo_llegada"]
                else:
                    thm = math.atan2(my - py, mx - px)
                th_plaza = thm if thm is not None else 0.0
                if not self.env.libre(mx, my, th_plaza, veh.length, veh.width,
                                      margen=0.15):
                    raise ValueError(f"Vehículo {e['id']!r}: la plaza nueva "
                                     f"({mx:g}, {my:g}) no cabe.")
                lote.append((veh, e, thm))
        except ValueError as e:
            messagebox.showerror("Misiones nuevas no válidas", str(e))
            return

        self._detener()
        # Las misiones nuevas comienzan cuando todo lo anterior ha acabado: a
        # partir de ahí, los no mencionados son obstáculos ESTÁTICOS aparcados.
        k_inicio = max(len(v.traj) if v.traj else 1 for v in self.vehiculos)
        ids_lote = {veh.vid for veh, _, _ in lote}
        base = Reservas()
        for v in self.vehiculos:
            if v.vid not in ids_lote:
                base.add([v.pose_final], v.length, v.width)

        indices = []
        inicios = {}
        for veh, e, thm in lote:
            veh.meta = tuple(e["meta"])
            veh.meta_th = thm
            if "grupo" in e:
                veh.grupo = e["grupo"]
            if "prioridad" in e:
                veh.prioridad = e["prioridad"]
            if "v_max" in e:
                veh.v_max = e["v_max"]
            if "a_max" in e:
                veh.a_max = e["a_max"]
            if "giro_max" in e:
                veh.delta_max = e["giro_max"]
            indices.append(veh.idx)
            inicios[veh.idx] = veh.pose_final

        self._ocupado = True
        try:
            self._warm()
            trays, motivos = self._planificar_flota(indices, base,
                                                    inicios=inicios,
                                                    reubicable=False)
            self._aplicar_resultado(indices, trays, motivos, k_inicio=k_inicio)
        finally:
            self._ocupado = False

    # ----------------------------- reproducción --------------------------- #
    def reproducir(self):
        if not self.frames:
            from tkinter import messagebox
            messagebox.showinfo("Nada que reproducir", "Primero calcula y simula.")
            return
        self._detener()
        self.frame = 0
        self.reproduciendo = True
        self._anim()

    def _anim(self):
        if not self.reproduciendo or not self.frames:
            return
        try:
            if not self.root.winfo_exists():
                return
            self.frame = max(0, min(self.frame, len(self.frames) - 1))
            self._dibujar_frame()
            if self.frame >= len(self.frames) - 1:
                self.reproduciendo = False
                self.estado.set("Reproducción finalizada. Puedes escribir misiones "
                                "nuevas en el texto y pulsar «Nuevas rutas».")
                return
            self.frame += 1
            self.anim_id = self.root.after(40, self._anim)
        except self.tk.TclError:
            self.reproduciendo = False

    def pausar(self):
        if not self.frames:
            return
        if self.reproduciendo:
            self.reproduciendo = False
            if self.anim_id:
                self.root.after_cancel(self.anim_id)
            self.estado.set("Pausado.")
        else:
            self.reproduciendo = True
            self.estado.set("Reanudando…")
            self._anim()

    def reiniciar(self):
        self._detener()
        self.frame = 0
        self.estado.set("Reiniciado al primer fotograma.")
        if self.frames:
            self._dibujar_frame()
        else:
            self._dibujar_estatico()

    def _detener(self):
        self.reproduciendo = False
        if self.anim_id:
            self.root.after_cancel(self.anim_id)
            self.anim_id = None

    # -------------------------------- dibujo ------------------------------ #
    def _poly_px(self, poly):
        out = []
        for x, y in poly:
            out.extend((x * SCALE, y * SCALE))
        return out

    def _fondo(self):
        c = self.canvas
        c.delete("all")
        for poly in self.env.obstaculos:
            c.create_polygon(self._poly_px(poly), fill=COL_OBST, outline=COL_OBST)
        for veh in self.vehiculos:
            col = PALETA[veh.idx % len(PALETA)]
            mx, my = veh.meta
            c.create_oval(mx * SCALE - 7, my * SCALE - 7,
                          mx * SCALE + 7, my * SCALE + 7, outline=col, width=2)
            c.create_text(mx * SCALE, my * SCALE, text="◎", fill=col,
                          font=("", 13, "bold"))
            if veh.meta_th is not None:
                # Flecha del ÁNGULO DE LLEGADA exigido en el destino.
                ax = mx + 1.2 * math.cos(veh.meta_th)
                ay = my + 1.2 * math.sin(veh.meta_th)
                c.create_line(mx * SCALE, my * SCALE, ax * SCALE, ay * SCALE,
                              fill=col, width=2, arrow="last")
            poly = obb_corners(veh.inicio[0], veh.inicio[1], veh.inicio[2],
                               veh.length, veh.width)
            c.create_polygon(self._poly_px(poly), outline=col, fill="",
                             width=1, dash=(3, 3))

    def _rutas(self):
        for veh in self.vehiculos:
            if veh.traj:
                col = PALETA[veh.idx % len(PALETA)]
                pts = []
                for x, y, _ in veh.traj:
                    pts.extend((x * SCALE, y * SCALE))
                if len(pts) >= 4:
                    self.canvas.create_line(pts, fill=col, width=1, smooth=True)

    def _dibujar_estatico(self):
        self._fondo()
        for veh in self.vehiculos:
            self._dibujar_coche(veh.inicio[0], veh.inicio[1], veh.inicio[2],
                                veh.length, veh.width,
                                PALETA[veh.idx % len(PALETA)], veh.vid)

    def _dibujar_frame(self):
        self._fondo()
        self._rutas()
        fr = self.frames[self.frame]
        for veh in self.vehiculos:
            x, y, th = fr[veh.idx]
            self._dibujar_coche(x, y, th, veh.length, veh.width,
                                PALETA[veh.idx % len(PALETA)], veh.vid)
        self.canvas.create_text(10, 12, anchor="w",
                                text=f"t = {self.frame * DT:.1f} s   "
                                     f"({self.frame}/{len(self.frames) - 1})",
                                fill=COL_BORDE, font=("", 11, "bold"))

    def _dibujar_coche(self, x, y, th, length, width, col, num):
        c = self.canvas
        c.create_polygon(self._poly_px(obb_corners(x, y, th, length, width)),
                         fill=col, outline="#101010", width=1)
        nx = x + math.cos(th) * length * 0.35
        ny = y + math.sin(th) * length * 0.35
        c.create_line(x * SCALE, y * SCALE, nx * SCALE, ny * SCALE,
                      fill="white", width=2)
        c.create_text(x * SCALE, y * SCALE, text=str(num), fill="white",
                      font=("", 10, "bold"))


def main():
    import tkinter as tk
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
