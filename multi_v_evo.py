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
# (1 - STEER_SPEED_C) del tope; a coche parado, al tope entero. Como en la vida
# real: a más velocidad, menos giro. Es el ÚNICO mecanismo del giro (no hay tope
# de aceleración lateral), y gobierna DOS efectos ligados:
#   · Frena en las curvas: para trazar una curva más cerrada de lo que el volante
#     (ya reducido por la velocidad) permite, el coche DEBE bajar de velocidad.
#     A más valor, más curvas obligan a frenar.
#   · Gana giro al ralentizar: al ir despacio (p. ej. la maniobra final) recupera
#     volante y cierra el radio. A más valor, mayor es esa ganancia.
# 0.35 casi no frenaba en curva ni ganaba giro; 0.55 frenaba demasiado. 0.45 es
# el punto medio. Súbelo si quieres más de ambos efectos; bájalo si frena de más.
STEER_SPEED_C = 0.5


# Velocidad de crucero de la MANIOBRA FINAL, como fracción de la v_max del
# vehículo. La última maniobra (tramo de reorientación + remate) se hace TODA a
# esta velocidad reducida: más cómoda para el pasajero y, al ir lento, con casi
# todo el volante disponible (acoplamiento giro–velocidad) para el giro cerrado
# de entrada. Antes solo iba lento el primer tramo y el remate volvía a acelerar
# a v_max; con esto no. El frenado final hasta parar sigue mandando cerca de la
# meta. Subir = maniobra más rápida; bajar = más lenta y con más giro.
V_MAN_FRAC = 0.45


# Tope FINAL de la RELAJACIÓN del ángulo de llegada (etapa 2). La etapa 2 abre el
# ángulo hasta LIBRE (±180°, sentido incluido) como máxima red de seguridad para
# llegar. La calidad 5 se detiene en ±90° (en la máxima calidad no interesa
# aparcar mirando al revés). El tope se fija por nivel en self.ang_tol_final,
# dentro de configurar_calidad.


# NOTA: se eliminó el límite de velocidad por aceleración lateral (un tope de
# rapidez universal en función de la curvatura). Rebajaba la velocidad al girar
# de forma ARTIFICIAL e igual para todos los vehículos, y hacía las curvas
# regulares. La física del giro la lleva ahora SOLO el acoplamiento
# giro–velocidad (STEER_SPEED_C): la rapidez limita el volante disponible —propio
# de cada vehículo por su delta_max— y el coche opera dentro de ese límite. La
# única rebaja de velocidad es la del frenado realista de aproximación a la meta.


# Relajación del ÁNGULO DE LLEGADA (red de seguridad para que todo vehículo
# LLEGUE aunque no logre encajar la orientación exacta). La tolerancia se
# ensancha SOLO según el ESFUERZO EN NODOS EXPLORADOS DENTRO DE LA ZONA DE LA
# META (d_goal ≤ dist_tiro), sin relojes de tiempo: así el resultado no
# depende de la velocidad de la máquina y una ruta larga no llega con la
# tolerancia ya regalada solo por el viaje. Los umbrales (por nivel de
# calidad) están en la tabla de configurar_calidad.


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
def _k_hits_dyn(x, y, th, kk, Ld, Wd, rx, roff, rlen, rlw, rrad, R, diag, mdin):
    """¿El OBB (inflado a Ld×Wd) choca con algún vehículo RESERVADO en el paso
    fino kk? Para kk más allá del final de una reserva, esta sigue aparcada."""
    if R == 0:
        return False
    vr = 0.5 * diag + mdin
    cA_ok = False
    for j in range(R):
        lj = rlen[j]
        idx = roff[j] + (kk if kk < lj else lj - 1)
        ox = rx[idx, 0]; oy = rx[idx, 1]
        r_sum = vr + rrad[j]
        dx = x - ox
        if dx > r_sum or dx < -r_sum:
            continue
        dy = y - oy
        if dy > r_sum or dy < -r_sum:
            continue
        if dx * dx + dy * dy > r_sum * r_sum:
            continue
        if not cA_ok:
            cA = _k_corners(x, y, th, Ld, Wd)
            aA = _k_axes(cA)
            cA_ok = True
        oth = rx[idx, 2]
        rl = rlw[j, 0]; rw = rlw[j, 1]
        cB = _k_corners(ox, oy, oth, rl, rw)
        aB = _k_axes(cB)
        if _k_sat(cA, aA, cB, aB):
            return True
    return False


@njit(cache=True)
def _k_expand(x0, y0, th0, v0, k, acc, dl, subpasos, h, L, amax,
              vmaxc, vrev, veh_len, veh_wid, margin, mdin, diag,
              worldW, worldH, oc, oax, obb, K, blc, blax, blbb, B,
              rx, roff, rlen, rlw, rrad, R, feas, osub, ov):
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
            # La rapidez limita el VOLANTE (acoplamiento giro–velocidad): a más
            # velocidad, menos ángulo de dirección efectivo. El coche opera
            # DENTRO de ese límite; NO se rebaja la velocidad por la curva —la
            # única frenada es la realista de aproximación a la meta—.
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
                           rx, roff, rlen, rlw, rrad, R, diag, mdin):
                ok = 0
                break
            osub[a, sp, 0] = x; osub[a, sp, 1] = y; osub[a, sp, 2] = th
        feas[a] = ok
        ov[a] = v


@njit(cache=True)
def _k_tiro(x, y, th, v, k, gx, gy, gth, con_ang, ang_tol,
            L, dmax, ld, vmaxc, amax, goal_tol, v_tol, h,
            veh_len, veh_wid, margin, mdin, diag, worldW, worldH,
            oc, oax, obb, K, blc, blax, blbb, B, rx, roff, rlen, rlw, rrad, R):
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
            s = 0.85 * d
            smax = 4.0 * L
            if s > smax:
                s = smax
            tx = gx - s * cos(gth); ty = gy - s * sin(gth)
        else:
            tx = gx; ty = gy
        v_ant = v
        # FRENADA FÍSICA (v² = 2·a·d): la velocidad admisible en cada instante es
        # la máxima desde la que aún puede detenerse en la meta con la
        # deceleración disponible, medida sobre la distancia de RUTA restante
        # (recta hasta el borde de la tolerancia + arco de corrección del rumbo
        # si se exige ángulo de llegada). Lejos de la meta no limita (el remate
        # cruza a vmax) y al acercarse impone la parábola de frenado exacta del
        # vehículo: ni se pasa de largo (el umbral crece con v²/a, no con el
        # tamaño del coche) ni recorre los últimos metros a paso de tortuga (la
        # velocidad ya no queda congelada a la de entrada). El factor 0.85 deja
        # margen para que el seguimiento (acotado por a_max) sea alcanzable, y
        # el suelo v_tol/2 garantiza entrar en la meta sin estancarse.
        da_err = abs(atan2(sin(th - gth), cos(th - gth))) if con_ang == 1 else 0.0
        R_min = L / max(1e-3, tan(dmax))
        d_freno = (d - goal_tol) + da_err * R_min
        if d_freno < 0.0:
            d_freno = 0.0
        v_des = sqrt(0.25 * v_tol * v_tol + 2.0 * 0.85 * amax * d_freno)
        # La maniobra final (con ángulo de llegada) va TODA a velocidad reducida:
        # el crucero se topa a V_MAN_FRAC·vmax en vez de a vmax, para que el
        # remate NO vuelva a acelerar tras el tramo de reorientación. La parábola
        # de frenado sigue mandando al acercarse (frena hasta parar).
        v_top = V_MAN_FRAC * vmaxc if con_ang == 1 else vmaxc
        if v_top < 0.4:
            v_top = 0.4
        if v_des > v_top:
            v_des = v_top
        dv = max(-amax * h, min(amax * h, v_des - v))
        v = min(vmaxc, max(0.0, v + dv))
        alpha = atan2(ty - y, tx - x) - th
        alpha = atan2(sin(alpha), cos(alpha))
        # Lookahead acotado: nunca mayor que 'ld', para que la corrección de rumbo
        # sea firme al principio y el tramo restante quede recto. Cerca del punto
        # perseguido se contrae a la distancia real para homing preciso.
        dt_ = hypot(tx - x, ty - y)
        # Lookahead que se ACORTA con la velocidad: despacio (maniobra final) el
        # seguimiento es más firme → el piloto pide más volante y la trayectoria
        # de llegada se CIERRA; rápido, lookahead largo → llegada suave. Así la
        # bajada de velocidad sí aprovecha el mayor volante disponible al ralentí.
        ld_v = ld * (0.5 + 0.5 * (v / vmaxc))
        den = dt_ if dt_ < ld_v else ld_v
        if den < 1e-3:
            den = 1e-3
        # Tope de dirección reducido con la rapidez (giro–velocidad): a más
        # velocidad, menos volante disponible. El coche gira DENTRO de ese tope;
        # no se rebaja la velocidad por la curvatura. La única frenada es la
        # realista de aproximación (v_des, arriba): entra al giro con la
        # velocidad que trae y va frenando, así el arco se cierra al ralentizar.
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
                       rx, roff, rlen, rlw, rrad, R, diag, mdin):
            return -1, out
        out[n, 0] = x; out[n, 1] = y; out[n, 2] = th
        n += 1
        kk += 1
        if v < 1e-3 and d > goal_tol:
            return -1, out
    return -1, out


@njit(cache=True)
def _k_aparca(px, py, pth, k0, kfin, Ld, Wd,
              rx, roff, rlen, rlw, rrad, R, diag, mdin):
    """La plaza final debe quedar libre para siempre: comprueba la pose de
    aparcamiento contra todas las reservas desde k0 hasta que todas paran."""
    for kk in range(k0, kfin + 1):
        if _k_hits_dyn(px, py, pth, kk, Ld, Wd, rx, roff, rlen, rlw, rrad, R, diag, mdin):
            return False
    return True


@njit(cache=True)
def _k_maniobra(x, y, th, v0, k, gx, gy, gth, ang_tol,
                L, dmax, ld, vmaxc, amax, goal_tol, v_tol, h,
                veh_len, veh_wid, margin, mdin, diag, worldW, worldH,
                oc, oax, obb, K, blc, blax, blbb, B,
                rx, roff, rlen, rlw, rrad, R, res_maxlen):
    """Conector REVERSIBLE para asentar el ÁNGULO DE LLEGADA en sitio apretado.

    Cuando la búsqueda se atasca intentando encarar la meta con la orientación
    exigida, este conector prueba una primera MANIOBRA —adelante o marcha atrás,
    girando a un lado u otro y con distintas longitudes— que puede ALEJARSE de la
    meta, y desde la pose resultante remata hacia adelante con _k_tiro. Es la
    maniobra clásica de 'tirar un poco y rectificar' (o su versión en retroceso).

    SENTIDO Y VELOCIDAD DE ENTRADA ('v0'): una PARADA solo tiene sentido cuando se
    CAMBIA de sentido. Por eso:
      · Tramo en el MISMO sentido que v0: ARRASTRA la velocidad de entrada (no
        frena en seco) y enlaza directo con el remate. Sin paradas intermedias.
      · Tramo que INVIERTE el sentido: la propia maniobra antepone un PREFIJO de
        frenado físico en recto (a_max) hasta v=0 y solo entonces invierte. Se
        acepta CUALQUIER velocidad de entrada: el coche ya no necesita llegar
        parado para poder plantearse ir marcha atrás (era eso lo que hacía a la
        búsqueda detenerse "por si acaso" cerca de la meta), y la inversión
        siempre pasa por velocidad nula (nunca salta de -v a +v).

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
    # Crucero de la maniobra final DELIBERADAMENTE bajo (misma constante que el
    # remate): toda la maniobra despacio, con más volante disponible para el giro.
    v_man = V_MAN_FRAC * vmaxc
    if v_man < 0.4:
        v_man = 0.4
    va0 = v0 if v0 >= 0.0 else -v0            # rapidez de entrada
    # Sentido de entrada: +1 avanzando, -1 retrocediendo, 0 casi parado.
    sv0 = 0.0
    if v0 > v_tol:
        sv0 = 1.0
    elif v0 < -v_tol:
        sv0 = -1.0
    best_n = -1
    best_total = 1e18
    # Segmento = [prefijo de frenado si hay inversión] + rampa + cola de frenado.
    seg = np.empty((tail_max + seg_max + tail_max, 3))
    for di in range(2):
        dirf = dirs[di]
        adelante = dirf > 0.0
        for si in range(5):
            frac = steers[si]
            x1 = x; y1 = y; th1 = th
            kk = k
            # SENTIDO: si el tramo va en el MISMO sentido que el coche, arrastra
            # la velocidad (no frena en seco). Si INVIERTE el sentido, primero un
            # PREFIJO de frenado físico en recto hasta v=0 (la inversión pasa por
            # velocidad nula: nada de saltar de -v a +v de golpe).
            nbrk = 0
            if sv0 != 0.0 and ((sv0 > 0.0) != adelante):
                vb = va0
                brk_ok = True
                while vb > 1e-3 and nbrk < tail_max:
                    vb -= amax * h
                    if vb < 0.0:
                        vb = 0.0
                    vvb = sv0 * vb
                    x1 = x1 + cos(th1) * vvb * h
                    y1 = y1 + sin(th1) * vvb * h
                    kk += 1
                    if not _k_free_sb(x1, y1, th1, Lm, Wm, worldW, worldH,
                                      oc, oax, obb, K, blc, blax, blbb, B, diag, margin):
                        brk_ok = False
                        break
                    if _k_hits_dyn(x1, y1, th1, kk, Ld, Wd,
                                   rx, roff, rlen, rlw, rrad, R, diag, mdin):
                        brk_ok = False
                        break
                    seg[nbrk, 0] = x1; seg[nbrk, 1] = y1; seg[nbrk, 2] = th1
                    nbrk += 1
                if not brk_ok:
                    continue
                vmag = 0.0
            else:
                vmag = va0
            for step in range(seg_max):
                # Aproxima la velocidad de crucero de maniobra respetando a_max en
                # ambos sentidos: si se entra por encima, frena suave (no en seco);
                # si por debajo, acelera. Con v0=0 (marcha atrás) es la rampa de
                # siempre.
                if vmag > v_man:
                    vmag = vmag - amax * h
                    if vmag < v_man:
                        vmag = v_man
                else:
                    vmag = min(v_man, vmag + amax * h)
                vv = dirf * vmag
                # El volante disponible se reduce con la velocidad; a este crucero
                # bajo queda casi entero. No se rebaja la velocidad por la curva.
                fdel = tan(frac * dmax * (1.0 - STEER_SPEED_C * (vmag / vmaxc)))
                th1 = th1 + (1.0 / L) * fdel * (vv * h)
                x1 = x1 + cos(th1) * vv * h
                y1 = y1 + sin(th1) * vv * h
                kk += 1
                if not _k_free_sb(x1, y1, th1, Lm, Wm, worldW, worldH,
                                  oc, oax, obb, K, blc, blax, blbb, B, diag, margin):
                    break                    # este tramo choca: se abandona
                if _k_hits_dyn(x1, y1, th1, kk, Ld, Wd,
                               rx, roff, rlen, rlw, rrad, R, diag, mdin):
                    break
                seg[nbrk + step, 0] = x1; seg[nbrk + step, 1] = y1
                seg[nbrk + step, 2] = th1
                ncore = nbrk + step + 1
                if (step & 1) != 0:          # prueba a rematar cada 2 subpasos
                    continue
                if adelante:
                    # HACIA DELANTE: sin cambio de sentido → NADA de cola de frenado
                    # ni parada intermedia. El remate _k_tiro enlaza con la velocidad
                    # ACTUAL y frena de forma continua hasta la meta.
                    xt = x1; yt = y1; tht = th1; kt = kk
                    nseg = ncore
                    v_rem = vmag
                else:
                    # MARCHA ATRÁS: cola de frenado hasta v=0 (respetando a_max) antes
                    # de rematar hacia delante, para que la inversión pase por
                    # velocidad nula. La pose final del tramo queda DETENIDA.
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
                                       rx, roff, rlen, rlw, rrad, R, diag, mdin):
                            tail_ok = False; break
                        seg[ncore + ntail, 0] = xt
                        seg[ncore + ntail, 1] = yt
                        seg[ncore + ntail, 2] = tht
                        ntail += 1
                    if not tail_ok:
                        continue
                    nseg = ncore + ntail
                    v_rem = 0.0
                # Remate: hacia delante arranca con la velocidad arrastrada; en
                # retroceso, desde el coche DETENIDO.
                nt, pt = _k_tiro(xt, yt, tht, v_rem, kt, gx, gy, gth, 1, ang_tol,
                                 L, dmax, ld, vmaxc, amax, goal_tol, v_tol, h,
                                 veh_len, veh_wid, margin, mdin, diag, worldW, worldH,
                                 oc, oax, obb, K, blc, blax, blbb, B,
                                 rx, roff, rlen, rlw, rrad, R)
                if nt <= 0:
                    continue
                kfin = kt + nt
                kmax = kfin if kfin > res_maxlen else res_maxlen
                if not _k_aparca(pt[nt - 1, 0], pt[nt - 1, 1], pt[nt - 1, 2],
                                 kfin, kmax, Ld, Wd,
                                 rx, roff, rlen, rlw, rrad, R, diag, mdin):
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
def _k_seg(x, y, th, k, dirf, frac, npasos, L, dmax, vmaxc, amax, h,
           Lm, Wm, Ld, Wd, worldW, worldH, oc, oax, obb, K,
           blc, blax, blbb, B, rx, roff, rlen, rlw, rrad, R,
           diag, margin, mdin, out, base):
    """Tramo de maniobra con sentido (adelante/atrás) y giro CONSTANTES que
    arranca y TERMINA PARADO: rampa de aceleración de 'npasos' subpasos + cola
    de frenado hasta v=0 (la inversión de sentido siempre pasa por velocidad
    nula). Escribe las poses en out[base:] y devuelve (n, x, y, th, k) con la
    pose final; n = -1 si el tramo choca o se sale del mundo."""
    # Crucero bajo de maniobra (misma constante que la de una fase y el remate):
    # toda despacio, con más volante disponible. Sin rebaja por curvatura; el tope
    # de dirección ya se reduce con la velocidad más abajo.
    v_man = V_MAN_FRAC * vmaxc
    if v_man < 0.4:
        v_man = 0.4
    tope_cola = amax * h * 60.0        # la cola de frenado debe caber en 60 subpasos
    if v_man > tope_cola:
        v_man = tope_cola
    n = 0
    vmag = 0.0
    for fase in range(2):
        pasos = npasos if fase == 0 else 70
        for _ in range(pasos):
            if fase == 0:
                vmag = min(v_man, vmag + amax * h)
            else:
                vmag -= amax * h
                if vmag <= 1e-3:
                    return n, x, y, th, k
            vv = dirf * vmag
            fdel = tan(frac * dmax * (1.0 - STEER_SPEED_C * (vmag / vmaxc)))
            th = th + (1.0 / L) * fdel * (vv * h)
            x = x + cos(th) * vv * h
            y = y + sin(th) * vv * h
            k += 1
            if not _k_free_sb(x, y, th, Lm, Wm, worldW, worldH,
                              oc, oax, obb, K, blc, blax, blbb, B, diag, margin):
                return -1, x, y, th, k
            if _k_hits_dyn(x, y, th, k, Ld, Wd,
                           rx, roff, rlen, rlw, rrad, R, diag, mdin):
                return -1, x, y, th, k
            out[base + n, 0] = x; out[base + n, 1] = y; out[base + n, 2] = th
            n += 1
    return n, x, y, th, k


@njit(cache=True)
def _k_maniobra2(x, y, th, k, gx, gy, gth, ang_tol,
                 L, dmax, ld, vmaxc, amax, goal_tol, v_tol, h,
                 veh_len, veh_wid, margin, mdin, diag, worldW, worldH,
                 oc, oax, obb, K, blc, blax, blbb, B,
                 rx, roff, rlen, rlw, rrad, R, res_maxlen):
    """Conector de DOS FASES (el 'cambio de sentido doble' de una maniobra de
    tres puntos): tramo 1 adelante o atrás + tramo 2 en el sentido OPUESTO +
    remate hacia delante con _k_tiro. Cubre el caso que la maniobra de una fase
    no resuelve: el vehículo 'pasado' de su plaza que debe salir, invertir el
    rumbo por tramos y entrar encarado con el ángulo exigido (p. ej. arrancar a
    3 m de la meta mirando hacia ella pero teniendo que acabar mirando hacia
    fuera). Cada combinación se valida entera (mundo, obstáculos, reservas y
    aparcamiento definitivo) y gana la maniobra completa más corta. El remate
    solo se intenta si la pose tras el tramo 2 ENCARA la zanahoria del eje de
    llegada (filtro barato que descarta la mayoría de combinaciones)."""
    out = np.empty((1400, 3))
    seg1 = np.empty((120, 3))
    seg2 = np.empty((120, 3))
    dirs = (1.0, -1.0)
    steers = (-1.0, -0.5, 0.0, 0.5, 1.0)
    largos = (10, 22, 34)
    best_n = -1
    best_total = 1e18
    Ld = veh_len + 2.0 * mdin
    Wd = veh_wid + 2.0 * mdin
    Lm = veh_len + 2.0 * margin
    Wm = veh_wid + 2.0 * margin
    for di in range(2):
        d1 = dirs[di]
        for s1 in range(5):
            f1 = steers[s1]
            for l1 in range(3):
                n1, x1, y1, th1, k1 = _k_seg(
                    x, y, th, k, d1, f1, largos[l1], L, dmax, vmaxc, amax, h,
                    Lm, Wm, Ld, Wd, worldW, worldH, oc, oax, obb, K,
                    blc, blax, blbb, B, rx, roff, rlen, rlw, rrad, R,
                    diag, margin, mdin, seg1, 0)
                if n1 < 0:
                    continue
                for s2 in range(5):
                    f2 = steers[s2]
                    for l2 in range(3):
                        n2, x2, y2, th2, k2 = _k_seg(
                            x1, y1, th1, k1, -d1, f2, largos[l2], L, dmax,
                            vmaxc, amax, h, Lm, Wm, Ld, Wd, worldW, worldH,
                            oc, oax, obb, K, blc, blax, blbb, B,
                            rx, roff, rlen, rlw, rrad, R,
                            diag, margin, mdin, seg2, 0)
                        if n2 < 0:
                            continue
                        d2 = hypot(gx - x2, gy - y2)
                        if d2 < 1e-6:
                            continue
                        s = 0.85 * d2
                        smax = 4.0 * L
                        if s > smax:
                            s = smax
                        tx = gx - s * cos(gth); ty = gy - s * sin(gth)
                        alpha = atan2(ty - y2, tx - x2) - th2
                        alpha = atan2(sin(alpha), cos(alpha))
                        if alpha > 0.9 or alpha < -0.9:
                            continue
                        nt, pt = _k_tiro(x2, y2, th2, 0.0, k2, gx, gy, gth,
                                         1, ang_tol, L, dmax, ld, vmaxc, amax,
                                         goal_tol, v_tol, h, veh_len, veh_wid,
                                         margin, mdin, diag, worldW, worldH,
                                         oc, oax, obb, K, blc, blax, blbb, B,
                                         rx, roff, rlen, rlw, rrad, R)
                        if nt <= 0:
                            continue
                        kfin = k2 + nt
                        kmax = kfin if kfin > res_maxlen else res_maxlen
                        if not _k_aparca(pt[nt - 1, 0], pt[nt - 1, 1],
                                         pt[nt - 1, 2], kfin, kmax, Ld, Wd,
                                         rx, roff, rlen, rlw, rrad, R,
                                         diag, mdin):
                            continue
                        total = n1 + n2 + nt
                        if total < best_total:
                            best_total = total
                            m = 0
                            for q in range(n1):
                                out[m, 0] = seg1[q, 0]; out[m, 1] = seg1[q, 1]
                                out[m, 2] = seg1[q, 2]; m += 1
                            for q in range(n2):
                                out[m, 0] = seg2[q, 0]; out[m, 1] = seg2[q, 1]
                                out[m, 2] = seg2[q, 2]; m += 1
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
_EMPTY_RAD = np.empty(0)



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

    def desde_poligonos(self, polys, nombre="importado"):
        """Construye el mapa a partir de polígonos (cajas ORIENTADAS de 4 esquinas)
        en metros. A diferencia de `desde_rects`, admite obstáculos rotados, así que
        `self.rects` queda a None y la persistencia usa la rama de `obstaculos`."""
        obs = [[(float(px), float(py)) for px, py in pol] for pol in polys]
        self.origen = nombre
        self.rects = None
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


# --------------------------------------------------------------------------- #
# Imagen → POLÍGONOS de forma libre (contorno real + relleno)
#
# En vez de forzar rectángulos, se detecta el CONTORNO real de cada mancha de
# obstáculo y se rellena en triángulos. Así una forma diagonal, en L o curva se
# reproduce con fidelidad (sus aristas son las líneas de colisión) y su interior
# queda cubierto de obstáculo (los vehículos no pueden aparecer dentro). Método:
#   1. Rejilla de ocupación (negro = obstáculo), fina.
#   2. Se extraen las aristas de frontera (lados de celda ocupada con vecino libre)
#      y se encadenan en lazos cerrados → el contorno exacto de cada forma.
#   3. Douglas-Peucker simplifica cada lazo: las "escaleras" de una pared diagonal
#      colapsan a una recta diagonal limpia; se conservan los vértices reales.
#   4. Cada lazo se triangula (ear clipping) para rellenar el interior. El motor
#      SAT es de 4 esquinas, así que cada triángulo se emite como cuadrilátero
#      DEGENERADO [A, B, C, C] (el eje repetido es (0,0), que el SAT ya ignora).
# Nota: los huecos interiores de una forma se rellenan como sólido (macizo).
# --------------------------------------------------------------------------- #
IMG_POLY_NX = 480        # rejilla de muestreo (a mayor nº, más fidelidad de contorno)
IMG_POLY_NY = 288        # mantiene la proporción 40:24 del mundo
POLY_TOL_CELDAS = 1.5    # tolerancia Douglas-Peucker en nº de celdas (suaviza escaleras)
POLY_ENDEREZAR_DEG = 8.0  # aristas a < este ángulo de la horizontal/vertical se enderezan


def _cruz(a, b, c):
    """Producto cruzado (b-a)×(c-a): >0 giro a la izquierda (CCW)."""
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _area_con_signo(pts):
    """Área con signo (shoelace); >0 si el polígono está en orden CCW."""
    s = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return 0.5 * s


def _contornos_grid(grid):
    """Lazos de frontera (listas de vértices en coordenadas de esquina de celda)
    de las manchas ocupadas. Cada lado de una celda ocupada cuyo vecino está libre
    es una arista dirigida (obstáculo a la izquierda); encadenándolas por extremos
    salen los contornos exteriores (CCW) y de los huecos (CW)."""
    nx = len(grid); ny = len(grid[0]) if nx else 0

    def occ(x, y):
        return 0 <= x < nx and 0 <= y < ny and grid[x][y]

    sig = {}                                   # vértice inicio → lista de vértices fin
    for ix in range(nx):
        col = grid[ix]
        for iy in range(ny):
            if not col[iy]:
                continue
            c00 = (ix, iy); c10 = (ix + 1, iy)
            c11 = (ix + 1, iy + 1); c01 = (ix, iy + 1)
            if not occ(ix, iy - 1): sig.setdefault(c00, []).append(c10)   # lado inferior
            if not occ(ix + 1, iy): sig.setdefault(c10, []).append(c11)   # lado derecho
            if not occ(ix, iy + 1): sig.setdefault(c11, []).append(c01)   # lado superior
            if not occ(ix - 1, iy): sig.setdefault(c01, []).append(c00)   # lado izquierdo

    lazos = []
    for inicio in list(sig.keys()):
        while sig.get(inicio):
            lazo = [inicio]; cur = inicio; cerrado = False
            while True:
                salientes = sig.get(cur)
                if not salientes:
                    break
                nxt = salientes.pop()
                if not sig[cur]:
                    del sig[cur]
                if nxt == inicio:
                    cerrado = True
                    break
                lazo.append(nxt); cur = nxt
            if cerrado and len(lazo) >= 3:
                lazos.append(lazo)
    return lazos


def _limpiar_colineales(pts):
    """Quita vértices duplicados y colineales de un polígono cerrado."""
    q = []
    for p in pts:
        if not q or abs(p[0] - q[-1][0]) > 1e-9 or abs(p[1] - q[-1][1]) > 1e-9:
            q.append(p)
    if len(q) > 1 and abs(q[0][0] - q[-1][0]) < 1e-9 and abs(q[0][1] - q[-1][1]) < 1e-9:
        q.pop()
    n = len(q)
    if n < 3:
        return q
    out = []
    for i in range(n):
        a = q[(i - 1) % n]; b = q[i]; c = q[(i + 1) % n]
        if abs(_cruz(a, b, c)) > 1e-9:
            out.append(b)
    return out


def _dist_pt_seg(p, a, b):
    """Distancia del punto p al segmento ab."""
    ax, ay = a; bx, by = b; px, py = p
    dx = bx - ax; dy = by - ay
    L2 = dx * dx + dy * dy
    if L2 < 1e-18:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / L2
    t = 0.0 if t < 0 else (1.0 if t > 1 else t)
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _dp_abierto(pts, tol):
    """Douglas-Peucker sobre una polilínea (extremos fijos). Iterativo (sin
    recursión) para soportar lazos con muchos puntos."""
    n = len(pts)
    if n < 3:
        return pts[:]
    keep = [False] * n
    keep[0] = keep[n - 1] = True
    pila = [(0, n - 1)]
    while pila:
        i, j = pila.pop()
        if j <= i + 1:
            continue
        a = pts[i]; b = pts[j]
        dmax = -1.0; idx = -1
        for k in range(i + 1, j):
            d = _dist_pt_seg(pts[k], a, b)
            if d > dmax:
                dmax = d; idx = k
        if dmax > tol and idx != -1:
            keep[idx] = True
            pila.append((i, idx)); pila.append((idx, j))
    return [pts[k] for k in range(n) if keep[k]]


def _dp_cerrado(pts, tol):
    """Douglas-Peucker sobre un polígono cerrado: se parte en el vértice más lejano
    al primero y se simplifican las dos cadenas."""
    n = len(pts)
    if n < 4:
        return pts
    p0 = pts[0]
    lejos = max(range(n), key=lambda i: (pts[i][0] - p0[0]) ** 2 + (pts[i][1] - p0[1]) ** 2)
    if lejos == 0:
        return pts
    ra = _dp_abierto(pts[0:lejos + 1], tol)
    rb = _dp_abierto(pts[lejos:] + [pts[0]], tol)
    return ra[:-1] + rb[:-1]


def _enderezar_ejes(pts, tol_deg):
    """Ajusta las aristas casi horizontales/verticales a exactamente horizontal o
    vertical (dentro de `tol_deg` grados). Corrige paredes que salen 'tuertas' y
    mantiene los ángulos rectos (cuadrados) intactos, sin tocar las diagonales
    reales (que superan la tolerancia)."""
    if tol_deg <= 0 or len(pts) < 3:
        return pts
    tan_tol = math.tan(math.radians(tol_deg))
    p = [list(v) for v in pts]                   # copia mutable; a/b comparten vértice
    n = len(p)
    for _ in range(2):                           # un par de pasadas para converger
        for i in range(n):
            a = p[i]; b = p[(i + 1) % n]
            dx = b[0] - a[0]; dy = b[1] - a[1]
            adx = abs(dx); ady = abs(dy)
            if adx < 1e-9 and ady < 1e-9:
                continue
            if ady <= adx * tan_tol:             # casi horizontal → misma y
                ym = 0.5 * (a[1] + b[1]); a[1] = ym; b[1] = ym
            elif adx <= ady * tan_tol:           # casi vertical → misma x
                xm = 0.5 * (a[0] + b[0]); a[0] = xm; b[0] = xm
    return [tuple(v) for v in p]


# --------------------------------------------------------------------------- #
# Triangulación de polígonos CON HUECOS (port de 'earcut' de Mapbox)
#
# Ear clipping robusto con soporte de agujeros: enlaza cada hueco al contorno
# exterior con un "puente" correcto (rayo hacia la izquierda desde el vértice más
# a la izquierda del hueco) y triangula, con reintentos que curan intersecciones
# locales. Es el estándar probado; sustituye a un bridging casero que fallaba en
# formas con varios huecos (metía triángulos dentro de un hueco).
# --------------------------------------------------------------------------- #
class _EcN:
    __slots__ = ("i", "x", "y", "prev", "next", "steiner")

    def __init__(self, i, x, y):
        self.i = i; self.x = x; self.y = y
        self.prev = None; self.next = None
        self.steiner = False


def _ec_area(p, q, r):
    return (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)


def _ec_iguales(p1, p2):
    return p1.x == p2.x and p1.y == p2.y


def _ec_signo(x):
    return 1 if x > 0 else (-1 if x < 0 else 0)


def _ec_area_anillo(anillo):
    s = 0.0; n = len(anillo); j = n - 1
    for i in range(n):
        s += (anillo[j][0] - anillo[i][0]) * (anillo[i][1] + anillo[j][1])
        j = i
    return s


def _ec_insertar(i, x, y, ultimo):
    p = _EcN(i, x, y)
    if ultimo is None:
        p.prev = p; p.next = p
    else:
        p.next = ultimo.next; p.prev = ultimo
        ultimo.next.prev = p; ultimo.next = p
    return p


def _ec_quitar(p):
    p.next.prev = p.prev
    p.prev.next = p.next


def _ec_anillo(anillo, sentido_horario, contador):
    """Construye una lista doblemente enlazada del anillo con la orientación pedida."""
    s = _ec_area_anillo(anillo)
    ultimo = None
    it = anillo if (s > 0) == sentido_horario else reversed(anillo)
    for (x, y) in it:
        ultimo = _ec_insertar(contador[0], x, y, ultimo)
        contador[0] += 1
    if ultimo is not None and _ec_iguales(ultimo, ultimo.next):
        _ec_quitar(ultimo)
        ultimo = ultimo.next
    return ultimo


def _ec_filtrar(inicio, fin=None):
    if inicio is None:
        return inicio
    if fin is None:
        fin = inicio
    p = inicio
    while True:
        otra = False
        if not p.steiner and (_ec_iguales(p, p.next) or _ec_area(p.prev, p, p.next) == 0):
            _ec_quitar(p)
            p = fin = p.prev
            if p is p.next:
                break
            otra = True
        else:
            p = p.next
        if not (otra or p is not fin):
            break
    return fin


def _ec_en_triangulo(ax, ay, bx, by, cx, cy, px, py):
    return ((cx - px) * (ay - py) - (ax - px) * (cy - py) >= 0 and
            (ax - px) * (by - py) - (bx - px) * (ay - py) >= 0 and
            (bx - px) * (cy - py) - (cx - px) * (by - py) >= 0)


def _ec_es_oreja(oreja):
    a = oreja.prev; b = oreja; c = oreja.next
    if _ec_area(a, b, c) >= 0:
        return False                              # reflejo → no es oreja
    p = oreja.next.next
    while p is not oreja.prev:
        if (_ec_en_triangulo(a.x, a.y, b.x, b.y, c.x, c.y, p.x, p.y) and
                _ec_area(p.prev, p, p.next) >= 0):
            return False
        p = p.next
    return True


def _ec_en_segmento(p, q, r):
    return (q.x <= max(p.x, r.x) and q.x >= min(p.x, r.x) and
            q.y <= max(p.y, r.y) and q.y >= min(p.y, r.y))


def _ec_intersecta(p1, q1, p2, q2):
    o1 = _ec_signo(_ec_area(p1, q1, p2)); o2 = _ec_signo(_ec_area(p1, q1, q2))
    o3 = _ec_signo(_ec_area(p2, q2, p1)); o4 = _ec_signo(_ec_area(p2, q2, q1))
    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and _ec_en_segmento(p1, p2, q1):
        return True
    if o2 == 0 and _ec_en_segmento(p1, q2, q1):
        return True
    if o3 == 0 and _ec_en_segmento(p2, p1, q2):
        return True
    if o4 == 0 and _ec_en_segmento(p2, q1, q2):
        return True
    return False


def _ec_intersecta_poligono(a, b):
    p = a
    while True:
        if (p.i != a.i and p.next.i != a.i and p.i != b.i and p.next.i != b.i and
                _ec_intersecta(p, p.next, a, b)):
            return True
        p = p.next
        if p is a:
            return False


def _ec_localmente_dentro(a, b):
    if _ec_area(a.prev, a, a.next) < 0:
        return _ec_area(a, b, a.next) >= 0 and _ec_area(a, a.prev, b) >= 0
    return _ec_area(a, b, a.prev) < 0 or _ec_area(a, a.next, b) < 0


def _ec_medio_dentro(a, b):
    p = a; dentro = False
    px = (a.x + b.x) / 2.0; py = (a.y + b.y) / 2.0
    while True:
        if ((p.y > py) != (p.next.y > py) and p.next.y != p.y and
                px < (p.next.x - p.x) * (py - p.y) / (p.next.y - p.y) + p.x):
            dentro = not dentro
        p = p.next
        if p is a:
            break
    return dentro


def _ec_diagonal_valida(a, b):
    return (a.next.i != b.i and a.prev.i != b.i and not _ec_intersecta_poligono(a, b) and
            ((_ec_localmente_dentro(a, b) and _ec_localmente_dentro(b, a) and
              _ec_medio_dentro(a, b) and
              (_ec_area(a.prev, a, b.prev) != 0 or _ec_area(a, b.prev, b) != 0)) or
             (_ec_iguales(a, b) and _ec_area(a.prev, a, a.next) > 0 and
              _ec_area(b.prev, b, b.next) > 0)))


def _ec_partir_poligono(a, b):
    a2 = _EcN(a.i, a.x, a.y); b2 = _EcN(b.i, b.x, b.y)
    an = a.next; bp = b.prev
    a.next = b; b.prev = a
    a2.next = an; an.prev = a2
    b2.next = a2; a2.prev = b2
    bp.next = b2; b2.prev = bp
    return b2


def _ec_curar(inicio, triangulos):
    p = inicio
    while True:
        a = p.prev; b = p.next.next
        if (not _ec_iguales(a, b) and _ec_intersecta(a, p, p.next, b) and
                _ec_localmente_dentro(a, b) and _ec_localmente_dentro(b, a)):
            triangulos.append(((a.x, a.y), (p.x, p.y), (b.x, b.y)))
            _ec_quitar(p); _ec_quitar(p.next)
            p = inicio = b
        p = p.next
        if p is inicio:
            break
    return _ec_filtrar(p)


def _ec_partir_earcut(inicio, triangulos):
    a = inicio
    while True:
        b = a.next.next
        while b is not a.prev:
            if a.i != b.i and _ec_diagonal_valida(a, b):
                c = _ec_partir_poligono(a, b)
                a = _ec_filtrar(a, a.next)
                c = _ec_filtrar(c, c.next)
                _ec_earcut_linked(a, triangulos, 0)
                _ec_earcut_linked(c, triangulos, 0)
                return
            b = b.next
        a = a.next
        if a is inicio:
            break


def _ec_earcut_linked(oreja, triangulos, pasada):
    if oreja is None:
        return
    stop = oreja
    while oreja.prev is not oreja.next:
        prev = oreja.prev; nxt = oreja.next
        if _ec_es_oreja(oreja):
            triangulos.append(((prev.x, prev.y), (oreja.x, oreja.y), (nxt.x, nxt.y)))
            _ec_quitar(oreja)
            oreja = nxt.next; stop = nxt.next
            continue
        oreja = nxt
        if oreja is stop:
            if pasada == 0:
                _ec_earcut_linked(_ec_filtrar(oreja), triangulos, 1)
            elif pasada == 1:
                oreja = _ec_curar(_ec_filtrar(oreja), triangulos)
                _ec_earcut_linked(oreja, triangulos, 2)
            elif pasada == 2:
                _ec_partir_earcut(oreja, triangulos)
            break


def _ec_mas_izquierda(inicio):
    p = inicio; izq = inicio
    while True:
        if p.x < izq.x or (p.x == izq.x and p.y < izq.y):
            izq = p
        p = p.next
        if p is inicio:
            break
    return izq


def _ec_sector_contiene(m, p):
    return _ec_area(m.prev, m, p.prev) < 0 and _ec_area(p.next, m, m.next) < 0


def _ec_buscar_puente(hueco, exterior):
    p = exterior
    hx = hueco.x; hy = hueco.y
    qx = -1e18; m = None
    while True:                                   # rayo hacia la izquierda desde el hueco
        if hy <= p.y and hy >= p.next.y and p.next.y != p.y:
            x = p.x + (hy - p.y) * (p.next.x - p.x) / (p.next.y - p.y)
            if x <= hx and x > qx:
                qx = x
                m = p if p.x < p.next.x else p.next
                if x == hx:
                    return m
        p = p.next
        if p is exterior:
            break
    if m is None:
        return None
    stop = m; mx = m.x; my = m.y
    tan_min = 1e18
    p = m
    while True:
        if (hx >= p.x >= mx and hx != p.x and
                _ec_en_triangulo(hx if hy < my else qx, hy, mx, my,
                                 qx if hy < my else hx, hy, p.x, p.y)):
            tan = abs(hy - p.y) / (hx - p.x)
            if (_ec_localmente_dentro(p, hueco) and
                    (tan < tan_min or (tan == tan_min and
                     (p.x > m.x or (p.x == m.x and _ec_sector_contiene(m, p)))))):
                m = p; tan_min = tan
        p = p.next
        if p is stop:
            break
    return m


def _ec_eliminar_huecos(huecos, exterior, contador):
    cola = []
    for anillo in huecos:
        lista = _ec_anillo(anillo, False, contador)
        if lista is None:
            continue
        if lista is lista.next:
            lista.steiner = True
        cola.append(_ec_mas_izquierda(lista))
    cola.sort(key=lambda n: (n.x, n.y))
    for h in cola:
        puente = _ec_buscar_puente(h, exterior)
        if puente is None:
            continue
        inverso = _ec_partir_poligono(puente, h)
        _ec_filtrar(inverso, inverso.next)
        exterior = _ec_filtrar(puente, puente.next)
    return exterior


def _triangular_con_huecos(exterior, huecos):
    """Triangula el contorno exterior descontando los huecos. Devuelve lista de
    triángulos (cada uno, 3 vértices (x, y)). Rellena el interior menos los huecos."""
    contador = [0]
    nodo = _ec_anillo(exterior, True, contador)
    if nodo is None or nodo.next is nodo.prev:
        return []
    if huecos:
        nodo = _ec_eliminar_huecos(huecos, nodo, contador)
    triangulos = []
    _ec_earcut_linked(nodo, triangulos, 0)
    return triangulos


def _punto_en_poligono(p, poly):
    """Test punto-en-polígono por lanzamiento de rayo (par/impar)."""
    x, y = p
    dentro = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]; xj, yj = poly[j]
        if (yi > y) != (yj > y):
            xc = (xj - xi) * (y - yi) / (yj - yi + 1e-18) + xi
            if x < xc:
                dentro = not dentro
        j = i
    return dentro


def imagen_a_poligonos(ruta, nx=IMG_POLY_NX, ny=IMG_POLY_NY, umbral=IMG_UMBRAL,
                       tol=None):
    """Imagen → lista de polígonos (de 4 esquinas; triángulos como cuadrilátero
    degenerado) que reproducen el contorno real de cada forma y rellenan su interior.
    Sustituye a imagen_a_rects para admitir obstáculos de forma libre.
    Preserva los HUECOS: el espacio libre rodeado de obstáculo (habitaciones, etc.)
    no se rellena, porque cada hueco se descuenta de su forma exterior."""
    grid = imagen_a_grid(ruta, nx, ny, umbral)
    gx = len(grid); gy = len(grid[0]) if gx else 0
    if gx == 0 or gy == 0:
        return []
    res_x = W / gx; res_y = H / gy
    if tol is None:
        tol = POLY_TOL_CELDAS * max(res_x, res_y)

    # Lazos → metros y simplificados. Los de área positiva son contornos exteriores
    # (obstáculo macizo); los de área negativa son huecos (espacio libre interior).
    exteriores = []
    huecos = []
    for lazo in _contornos_grid(grid):
        pts = _limpiar_colineales([(vx * res_x, vy * res_y) for vx, vy in lazo])
        if len(pts) < 3:
            continue
        pts = _dp_cerrado(pts, tol)
        pts = _enderezar_ejes(pts, POLY_ENDEREZAR_DEG)
        pts = _limpiar_colineales(pts)
        if len(pts) < 3:
            continue
        if _area_con_signo(pts) > 0:
            exteriores.append(pts)               # contorno exterior (CCW)
        else:
            huecos.append(pts)                   # hueco (CW), para descontar con earcut

    # Cada hueco se asigna al contorno exterior más pequeño que lo contiene.
    asignados = [[] for _ in exteriores]
    for h in huecos:
        cont = [(abs(_area_con_signo(o)), oi)
                for oi, o in enumerate(exteriores) if _punto_en_poligono(h[0], o)]
        if cont:
            asignados[min(cont)[1]].append(h)

    salida = []
    for oi, outer in enumerate(exteriores):
        for a, b, c in _triangular_con_huecos(outer, asignados[oi]):
            salida.append([a, b, c, c])         # triángulo → cuadrilátero degenerado
    return salida


def guardar_mapa(nombre, rects=None, polys=None):
    """Guarda el mapa en `mapas/<nombre>.json` para recargarlo en otra ejecución.
    Admite dos formatos: rectángulos alineados (`rects`, [x, y, w, h]) o cajas
    ORIENTADAS/polígonos (`polys`, listas de 4 esquinas). Incluye el tamaño de
    mundo para poder reescalar al cargar."""
    os.makedirs(MAPAS_DIR, exist_ok=True)
    nombre = "".join(c for c in nombre if c.isalnum() or c in "-_ ").strip()
    if not nombre:
        nombre = "mapa"
    ruta = os.path.join(MAPAS_DIR, nombre + ".json")
    datos = {"version": 1, "mundo": [W, H]}
    if polys is not None:
        datos["obstaculos"] = [[[float(px), float(py)] for px, py in pol]
                               for pol in polys]
    else:
        datos["rects"] = [list(map(float, r)) for r in (rects or [])]
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(datos, f)
    return ruta


def cargar_mapa(ruta):
    """Carga un mapa guardado de tipo `rects` (alineado). Si el mundo del fichero
    tenía otro tamaño, los rectángulos se reescalan al mundo actual."""
    with open(ruta, "r", encoding="utf-8") as f:
        datos = json.load(f)
    w0, h0 = datos.get("mundo", [W, H])
    fx, fy = W / w0, H / h0
    return [(rx * fx, ry * fy, rw * fx, rh * fy)
            for rx, ry, rw, rh in datos["rects"]]


def cargar_mapa_en(env, ruta, nombre=None):
    """Carga un mapa directamente en el entorno, detectando el formato: cajas
    orientadas (`obstaculos`) o rectángulos alineados (`rects`, mapas antiguos).
    Reescala si el mundo guardado tenía otro tamaño. Devuelve el nº de obstáculos."""
    with open(ruta, "r", encoding="utf-8") as f:
        datos = json.load(f)
    if nombre is None:
        nombre = os.path.splitext(os.path.basename(ruta))[0]
    w0, h0 = datos.get("mundo", [W, H])
    fx, fy = W / w0, H / h0
    if datos.get("obstaculos") is not None:
        polys = [[(px * fx, py * fy) for px, py in pol]
                 for pol in datos["obstaculos"]]
        env.desde_poligonos(polys, nombre)
        return len(polys)
    rects = [(rx * fx, ry * fy, rw * fx, rh * fy)
             for rx, ry, rw, rh in datos["rects"]]
    env.desde_rects(rects, nombre)
    return len(rects)


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
                 meta_th=None, grupo=1, prioridad=1, vid=None, v_inicial=0.0):
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
        self.v_inicial = v_inicial    # velocidad de arranque (m/s)
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
        self.goal_tol = 1.6           # tolerancia GRUESA: por defecto; se ajusta en planificar al lado corto del vehículo
        self.goal_tol_fin = 0.20      # tolerancia FINA: el piloto conduce hasta aquí
        #                              (para no quedarse parado a un paso de la meta)
        self.v_tol = 0.45
        self.ang_tol_meta = math.radians(3)    # tolerancia del ÁNGULO DE LLEGADA
        self.k_max = 1600
        self.dist_tiro = 10.0         # por defecto; se ajusta en planificar al lado mayor del vehículo
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
        # Techo de NODOS CREADOS (no solo expandidos). Acota la memoria del
        # grafo de reconstrucción: cada nodo guarda un tramo (4 poses), así que
        # este límite es lo que evita que un vehículo irresoluble haga crecer la
        # memoria sin freno (y tumbe el servidor). Es el "número absurdamente
        # alto de nodos" a partir del cual se da por imposible.
        self.max_nodos = 3_000_000
        self.peso_h = 1.6
        self.dir_fracs = []
        self.configurar_calidad(3)
        self.deadline = None
        self.tick = None
        self.motivo = ""
        self.expandidos = 0                  # diagnóstico del último planificar()
        self.nodos = 0
        self.relajado = False                # ¿llegó con el ángulo relajado?
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
        # El PESO del heurístico se sube respecto a versiones anteriores para que
        # la búsqueda encuentre ruta en MUCHOS menos nodos (más rápida y con
        # menos memoria: clave para no agotar el tiempo/memoria del servidor y
        # para que, con el techo alto de nodos, todo vehículo resoluble llegue
        # sin dejar coches «aparcados»). La suavidad y la forma del giro ya NO
        # dependen del peso, sino del acoplamiento giro–velocidad, así que subirlo
        # no empeora los giros; solo hace las rutas algo menos óptimas (un poco
        # más largas), cosa asumible.
        tabla = {
            # nivel: (n_dir, ang_res, peso_h, max_nodos, ang_tol_max,
            #         rel_ini_g, rel_span1_g, rel_span2_g, plazo_mejora_g)
            # peso_h: peso del heurístico (fijo por nivel). ang_tol_max: tope del
            # ángulo de la etapa 1 (rad ≈ 26°,21°,17°,13°,7°). rel_span1_g:
            # expansiones-en-meta para pasar de ang_tol a ang_tol_max (etapa 1,
            # "casi bien encarada"). rel_span2_g: las que abarca la etapa 2,
            # ajustadas para que el ángulo crezca al DOBLE de ritmo que en la
            # etapa 1 hasta el ángulo libre (±180°). En calidad 5 el ritmo es el
            # mismo (como si fuera a ±180°) pero el ángulo se corta en ±90°.
            1: (17, 12, 1.9, 1200000, 0.4538,  84_000, 132_000,    442_000,  30_000),
            2: (23, 11, 1.6, 1650000, 0.3665,  98_000, 154_000,    680_000,  35_000),
            3: (31, 10, 1.3, 2400000, 0.2967, 112_000, 176_000,  1_025_000,  40_000),
            4: (41,  8, 1.1, 3600000, 0.2269, 140_000, 221_000,  1_845_000,  50_000),
            5: (55,  6, 1.1, 5400000, 0.12,   350_000, 350_000,  7_818_000, 120_000),
        }
        (n_dir, ang, peso, mx, a_max, r_ini_g, r_span1_g,
         r_span2_g, plazo_mejora_g) = tabla.get(int(nivel), tabla[3])
        half = n_dir // 2
        fracs = [0.0]
        for i in range(1, half + 1):
            f = (i / half) ** 1.3
            fracs += [f, -f]
        self.dir_fracs = sorted(fracs)
        self.res_ang = math.radians(ang)
        self.peso_h = peso
        self.max_exp = mx
        if mx * 2 > self.max_nodos:
            self.max_nodos = mx * 2
        self.ang_tol_max = a_max
        # Relajación del ángulo por esfuerzo EN ZONA DE META (expansiones con
        # d_goal ≤ dist_tiro, NADA de tiempo): sin relajar hasta rel_ini_g;
        # etapa 1 (hasta ang_tol_max) dura rel_span1_g expansiones más; etapa
        # 2 (hasta ángulo libre) rel_span2_g expansiones más todavía.
        self.rel_ini_g = r_ini_g
        self.rel_span1_g = r_span1_g
        self.rel_span2_g = r_span2_g
        # Tope final del ángulo tras la etapa 2: libre (±180°), salvo calidad 5,
        # que se corta en ±90°.
        self.ang_tol_final = math.radians(90) if int(nivel) == 5 else math.pi
        # Presupuesto de MEJORA del ángulo (búsqueda anytime), en EXPANSIONES
        # sin mejorar (no segundos): tras hallar una llegada solo válida por
        # relajación, cuántos nodos más se exploran buscando una mejor
        # encarada antes de conformarse con la candidata.
        self.plazo_mejora = plazo_mejora_g


    def _clave(self, x, y, th, v, k):
        # Campos de 8 bits SIN SOLAPE (ix|iy|ith|iv|k). La velocidad se desplaza
        # +64 antes de enmascarar: con marcha atrás v es NEGATIVA y el antiguo
        # `int(...) & 0xFF` producía 250-255, que invadía los bits de la
        # orientación → estados de retroceso aliasados con rumbos distintos y
        # podados como dominados (la búsqueda "agotaba" espacio que en realidad
        # no había explorado).
        ix = int(x / self.res_pos) & 0xFF
        iy = int(y / self.res_pos) & 0xFF
        ith = int((th % (2 * math.pi)) / self.res_ang) & 0xFF
        iv = (int(round(v / self.res_v)) + 64) & 0xFF
        # NOTA: k entra ENTERO en la clave (sin saturarlo aunque el mundo quede
        # estático). Se probó saturarlo para podar entre rebanadas temporales y
        # produce falsos "sin_ruta": al quedar fijada la primera pose continua
        # que entra en cada celda, un vehículo grande en pasillos estrechos no
        # puede refinar la pose en visitas posteriores y la frontera se agota.
        return (ix << 38) | (iy << 30) | (ith << 22) | (iv << 14) | k

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
            return _EMPTY_XY, _EMPTY_OFF, _EMPTY_OFF, _EMPTY_LW, _EMPTY_RAD, 0
        T = sum(len(t) for t, _, _ in items)
        rx = np.empty((T, 3)); off = np.empty(R, np.int64)
        rlen = np.empty(R, np.int64); rlw = np.empty((R, 2)); rrad = np.empty(R)
        p = 0
        for j, (traj, l, w) in enumerate(items):
            off[j] = p; rlen[j] = len(traj); rlw[j, 0] = l; rlw[j, 1] = w
            rrad[j] = 0.5 * math.hypot(l, w)
            for pose in traj:
                rx[p, 0] = pose[0]; rx[p, 1] = pose[1]; rx[p, 2] = pose[2]
                p += 1
        return rx, off, rlen, rlw, rrad, R

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
    def planificar(self, veh, reservas):
        """Devuelve la trayectoria [(x,y,th), ...] (una pose por paso fino de DT,
        con la velocidad ya incorporada) o None si no halla solución. El estado
        incluye la velocidad (x,y,θ,v,k) y las acciones eligen la aceleración
        (acotada) y la dirección, de modo que el planificador puede acelerar o
        frenar en cualquier momento si eso da una ruta mejor.

        Si veh.meta_th no es None, la llegada solo se acepta con la orientación
        final dentro de ang_tol_meta alrededor de ese ángulo."""
        self._len, self._wid = veh.length, veh.width
        self.goal_tol = min(veh.length, veh.width)
        self.dist_tiro = 1.0 * max(veh.length, veh.width)
        self.diag = veh.diag
        L = veh.wheelbase
        dmax = veh.delta_max
        r_min = L / max(1e-3, math.tan(dmax))   # radio de giro mínimo (para el heurístico)
        sx, sy, sth = veh.inicio
        gx, gy = veh.meta
        # Velocidad de arranque elegida por el usuario (0 en modo aleatorio).
        v_ini = veh.v_inicial
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
        rx, roff, rlen, rlw, rrad, R = self._pack_reservas(reservas)

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
        # Acota la velocidad de arranque al rango físico del vehículo (por si un
        # Vehiculo se creó a mano fuera del validador de especificaciones).
        v_ini = min(self.v_max_c, max(-self.v_rev, v_ini))
        clave0 = self._clave(sx, sy, sth, v_ini, 0)
        nid = 0
        padre_id = {}
        arista_id = {}
        delta_prev = {0: 0.0}
        acc_prev = {0: 0.0}
        h0 = self._h_time(sx, sy)
        abierto = [(self.peso_h * h0, 0, sx, sy, sth, v_ini, 0, 0.0)]
        mejor_g = {clave0: 0.0}
        expand = 0
        self._exp_ult_maniobra = 0
        self._last_tick = time.perf_counter()
        exp_goal = 0                      # expansiones DENTRO de la zona de meta
        self.motivo = "limite"
        self.relajado = False

        # BÚSQUEDA "ANYTIME" con DOS niveles de candidata (ambas se van
        # afinando y solo cierran la búsqueda como último recurso, tras
        # 'plazo_mejora' expansiones SIN mejorar, o al agotar plazo/nodos):
        #
        #  · mejor_traj — llegada PRECISA (el conector _k_tiro paró en la meta,
        #    a goal_tol_fin) pero con el ángulo solo válido por RELAJACIÓN. Una
        #    llegada precisa Y con el ángulo exacto se devuelve al instante; una
        #    relajada se guarda y se sigue buscando un ángulo mejor.
        #
        #  · backup_traj — parada de RESPALDO dentro de la tolerancia GRUESA
        #    (goal_tol, hasta 1.6 m de la meta). ANTES esto cerraba la búsqueda
        #    de inmediato → el vehículo se detenía a más de un metro del punto
        #    final «sin sentido». Ahora es solo una red de seguridad: se guarda
        #    la parada más CERCANA a la meta, pero la búsqueda sigue intentando
        #    una llegada precisa con _k_tiro; el respaldo únicamente se usa si
        #    no aparece nada mejor. mejor_traj (precisa) siempre gana al backup.
        mejor_traj = None
        mejor_err = 1e18
        backup_traj = None
        backup_d = 1e18
        expand_cand = 0                   # última expansión que MEJORÓ algo

        def _acepta(traj):
            """Llegada PRECISA (vía _k_tiro/maniobra). Ángulo exacto → cierra al
            instante; ángulo relajado → candidata (devuelve None y sigue)."""
            nonlocal mejor_traj, mejor_err, expand_cand
            if not con_ang:
                self.motivo = "ok"
                self.relajado = False
                return traj
            err = abs(ang_norm(traj[-1][2] - gth))
            if err <= ang_tol * 1.01:
                self.motivo = "ok"
                self.relajado = False
                return traj
            if err < mejor_err:
                mejor_traj = traj
                mejor_err = err
                expand_cand = expand
            return None

        def _cerrar_fallback():
            """Devuelve la mejor llegada disponible: la precisa (relajada de
            ángulo) si la hay, y si no la parada de respaldo más cercana."""
            if mejor_traj is not None:
                self.motivo = "ok"
                self.relajado = True
                return mejor_traj
            if backup_traj is not None:
                self.motivo = "ok"
                self.relajado = True
                return backup_traj
            return None

        while abierto and expand < self.max_exp and nid < self.max_nodos:
            _f, cid, x, y, th, v, k, g = heapq.heappop(abierto)
            expand += 1

            # Cierre por ESTANCAMIENTO: hay candidata (precisa o de respaldo) y
            # llevamos 'plazo_mejora' expansiones sin acercarnos más ni mejorar
            # el ángulo → nos quedamos con la mejor hallada.
            if ((mejor_traj is not None or backup_traj is not None)
                    and expand - expand_cand > self.plazo_mejora):
                return _cerrar_fallback()

            if (expand & 255) == 0:
                self.expandidos = expand
                self.nodos = nid
                now = time.perf_counter()
                if self.deadline is not None and now > self.deadline:
                    fb = _cerrar_fallback()
                    if fb is not None:
                        return fb
                    self.motivo = "limite"
                    return None
                if self.tick is not None and now - self._last_tick > 0.01:
                    self._last_tick = now
                    self.tick(expand)

            d_goal = math.hypot(x - gx, y - gy)
            if d_goal <= self.dist_tiro:
                exp_goal += 1

            # RELAJACIÓN PROGRESIVA DEL ÁNGULO DE LLEGADA (red de seguridad en
            # DOS ETAPAS: primero hasta el tope del nivel —llegadas "casi bien
            # encaradas"— y, si ni así aparece solución, hasta ángulo LIBRE).
            # Prioridad del usuario: llegar SIEMPRE (y pronto) > llegar con el
            # ángulo exacto. El avance se mide SOLO por EXPANSIONES DENTRO DE
            # LA ZONA DE LA META (d_goal ≤ dist_tiro), sin relojes de tiempo:
            # así una ruta larga no llega con la tolerancia ya regalada solo
            # por el viaje, y el resultado no depende de la velocidad de la
            # máquina.
            if con_ang:
                # Etapa 1 (hasta ang_tol_max) y etapa 2 (al DOBLE de ritmo hasta
                # el ángulo libre), cada una con su presupuesto de expansiones-en-
                # meta. El presupuesto de la etapa 2 (rel_span2_g) ya está fijado
                # para que su pendiente sea el doble que la de la etapa 1.
                fr1 = (exp_goal - self.rel_ini_g) / self.rel_span1_g
                if fr1 < 0.0:
                    fr1 = 0.0
                elif fr1 > 1.0:
                    fr1 = 1.0
                stage1_end = self.rel_ini_g + self.rel_span1_g
                fr2 = (exp_goal - stage1_end) / self.rel_span2_g
                if fr2 < 0.0:
                    fr2 = 0.0
                elif fr2 > 1.0:
                    fr2 = 1.0
                max_tol = max(ang_tol, self.ang_tol_max)
                ang_tol_din = (ang_tol + (max_tol - ang_tol) * fr1
                               + (math.pi - max_tol) * fr2)
                # Tope final del nivel: libre (±180°) en calidades 1-4, ±90° en
                # calidad 5.
                if ang_tol_din > self.ang_tol_final:
                    ang_tol_din = self.ang_tol_final
                if mejor_traj is not None:
                    # Ya hay llegada candidata: solo interesan llegadas
                    # CLARAMENTE mejores (85% de su error, sin bajar de la
                    # tolerancia original).
                    tol_mejora = 0.85 * mejor_err
                    if tol_mejora < ang_tol:
                        tol_mejora = ang_tol
                    if ang_tol_din > tol_mejora:
                        ang_tol_din = tol_mejora
                con_ang_din = 1
                peso_din = self.peso_h
            else:
                ang_tol_din = ang_tol
                con_ang_din = 0
                peso_din = self.peso_h

            # LLEGADA EN MARCHA ATRÁS (etapa 2, en reverso): si el coche viene EN
            # RETROCESO, cerca de la meta y YA encarado con el rumbo correcto
            # (sentido incluido: ang_tol estricto), se cierra frenando el retroceso
            # en recto hasta parar. Así, si la maniobra que hizo lo trae pasando
            # por la meta bien orientado, se PARA ahí: esa es la última maniobra,
            # no vuelve a salir hacia delante y reentrar. Se prueba ANTES que el
            # remate hacia delante (que, al apuntar la zanahoria, lo sacaría fuera).
            if (con_ang_din and v < -self.v_tol and d_goal <= self.dist_tiro
                    and abs(ang_norm(th - gth)) <= ang_tol):
                bx = x; by = y; bth = th; bv = v; bk = k
                Lm = self._len + 2.0 * self.margen; Wm = self._wid + 2.0 * self.margen
                Ld = self._len + 2.0 * self.margen_din; Wd = self._wid + 2.0 * self.margen_din
                btail = []
                ok_tail = True
                while bv < -1e-3 and len(btail) < 120:
                    bv = min(0.0, bv + self.a_max * h)
                    bx += math.cos(bth) * bv * h
                    by += math.sin(bth) * bv * h
                    bk += 1
                    if not _k_free_sb(bx, by, bth, Lm, Wm, W, H,
                                      oc, oax, obb, K, blc, blax, blbb, B, self.diag, self.margen):
                        ok_tail = False; break
                    if _k_hits_dyn(bx, by, bth, bk, Ld, Wd,
                                   rx, roff, rlen, rlw, rrad, R, self.diag, self.margen_din):
                        ok_tail = False; break
                    btail.append((bx, by, bth))
                if (ok_tail and btail
                        and math.hypot(bx - gx, by - gy) <= self.goal_tol_fin
                        and abs(ang_norm(bth - gth)) <= ang_tol):
                    kfin_b = bk if bk > res_maxlen else res_maxlen
                    if _k_aparca(bx, by, bth, bk, kfin_b, Ld, Wd,
                                 rx, roff, rlen, rlw, rrad, R, self.diag, self.margen_din):
                        base = self._reconstruir(padre_id, arista_id, cid, (sx, sy, sth))
                        res = _acepta(base + btail)
                        if res is not None:
                            return res

            if d_goal <= self.dist_tiro:
                # Se intenta PRIMERO conducir hasta el punto exacto (tolerancia
                # fina) con el piloto: así no se acepta un nodo que se queda parado a
                # un paso de la meta. Remata solo si ya apunta bien (o está muy
                # cerca), para no trazar un arco amplio desde una orientación mala.
                # Con ángulo de llegada, el "apuntar bien" se mide hacia la
                # zanahoria del eje de llegada, no hacia la meta.
                if con_ang_din:
                    s0 = min(0.62 * d_goal, 4.0 * L)
                    txa = gx - s0 * math.cos(gth); tya = gy - s0 * math.sin(gth)
                else:
                    txa = gx; tya = gy
                alpha0 = math.atan2(tya - y, txa - x) - th
                alpha0 = math.atan2(math.sin(alpha0), math.cos(alpha0))
                # Solo si NO va marcha atrás: el remate conduce hacia DELANTE y
                # arrancarlo con v<0 invertiría el sentido de golpe (sin frenar).
                # La llegada/inversión en retroceso va por sus propios conectores.
                if v >= -self.v_tol and (abs(alpha0) <= self.ang_tiro
                                         or (not con_ang_din and d_goal <= self.goal_tol * 1.5)):
                    n, poses = _k_tiro(x, y, th, v, k, gx, gy, gth, con_ang_din, ang_tol_din,
                                       L, dmax, self.ld_tiro,
                                       self.v_max_c, self.a_max, self.goal_tol_fin,
                                       self.v_tol, h, self._len, self._wid,
                                       self.margen, self.margen_din, self.diag,
                                       W, H, oc, oax, obb, K, blc, blax, blbb, B,
                                       rx, roff, rlen, rlw, rrad, R)
                    if n > 0:
                        tiro = [(poses[i, 0], poses[i, 1], poses[i, 2]) for i in range(n)]
                        kfin_t = k + n
                        Ld = self._len + 2.0 * self.margen_din
                        Wd = self._wid + 2.0 * self.margen_din
                        fx, fy, fth = tiro[-1]
                        kmax = kfin_t if kfin_t > res_maxlen else res_maxlen
                        if _k_aparca(fx, fy, fth, kfin_t, kmax, Ld, Wd,
                                     rx, roff, rlen, rlw, rrad, R, self.diag, self.margen_din):
                            base = self._reconstruir(padre_id, arista_id, cid,
                                                     (sx, sy, sth))
                            res = _acepta(base + tiro)
                            if res is not None:
                                return res

            # Respaldo (RED DE SEGURIDAD, no cierre): nodo dentro de la
            # tolerancia gruesa y casi parado. Si el remate _k_tiro no logró
            # llegar al punto exacto, esta parada aproximada se GUARDA como
            # candidata de último recurso —la más cercana a la meta— pero la
            # búsqueda NO se detiene: sigue intentando una llegada precisa. Así
            # el vehículo ya no se queda quieto a más de un metro del destino
            # «sin sentido»; solo caería en el respaldo si de verdad no existe
            # forma de rematar en el punto exacto dentro del presupuesto.
            if (d_goal <= self.goal_tol and abs(v) <= self.v_tol
                    and d_goal < backup_d):
                ang_ok = True
                if con_ang_din:
                    tol_resp = 1.5 * ang_tol_din
                    if tol_resp > self.ang_tol_final:
                        tol_resp = self.ang_tol_final   # tope final del nivel
                    ang_ok = abs(ang_norm(th - gth)) <= tol_resp
                if ang_ok:
                    Ld = self._len + 2.0 * self.margen_din
                    Wd = self._wid + 2.0 * self.margen_din
                    kfin = k if k > res_maxlen else res_maxlen
                    if _k_aparca(x, y, th, k, kfin, Ld, Wd,
                                 rx, roff, rlen, rlw, rrad, R, self.diag, self.margen_din):
                        backup_traj = self._reconstruir(padre_id, arista_id, cid,
                                                        (sx, sy, sth))
                        backup_d = d_goal
                        expand_cand = expand      # progreso: respaldo más cercano

            # Conector REVERSIBLE (adelante-y-atrás para cuadrar el ángulo): solo
            # cuando la búsqueda ya se atasca con el ángulo de llegada exigido y el
            # coche está cerca de la meta. Se prueba de forma espaciada para no
            # encarecer el caso normal. Si encuentra maniobra, cierra la ruta.
            #
            # Se le pasa la velocidad ACTUAL 'v': el tramo en el MISMO sentido la
            # arrastra (sin parar), y el que INVIERTE el sentido antepone su propio
            # prefijo de frenado físico hasta v=0. Ningún caso exige llegar parado:
            # la búsqueda ya no tiene motivo para detenerse «por si acaso».
            if (con_ang_din and d_goal <= self.dist_tiro
                    and expand > self.umbral_maniobra
                    and expand - self._exp_ult_maniobra >= self.paso_maniobra):
                self._exp_ult_maniobra = expand
                nm, pm = _k_maniobra(x, y, th, v, k, gx, gy, gth, ang_tol_din,
                                     L, dmax, self.ld_tiro, self.v_max_c, self.a_max,
                                     self.goal_tol_fin, self.v_tol, h,
                                     self._len, self._wid, self.margen, self.margen_din,
                                     self.diag, W, H, oc, oax, obb, K,
                                     blc, blax, blbb, B, rx, roff, rlen, rlw, rrad, R,
                                     res_maxlen)
                if nm <= 0 and abs(v) <= self.v_tol:
                    # La maniobra de UNA fase no basta (p. ej. vehículo 'pasado'
                    # de su plaza que necesita un cambio de sentido doble): se
                    # intenta la de DOS fases (inversión doble → arranca PARADA,
                    # por eso solo si el coche ya llega casi detenido) antes de
                    # seguir quemando nodos.
                    nm, pm = _k_maniobra2(x, y, th, k, gx, gy, gth, ang_tol_din,
                                          L, dmax, self.ld_tiro, self.v_max_c,
                                          self.a_max, self.goal_tol_fin,
                                          self.v_tol, h, self._len, self._wid,
                                          self.margen, self.margen_din,
                                          self.diag, W, H, oc, oax, obb, K,
                                          blc, blax, blbb, B,
                                          rx, roff, rlen, rlw, rrad, R,
                                          res_maxlen)
                if nm > 0:
                    maniobra = [(pm[i, 0], pm[i, 1], pm[i, 2]) for i in range(nm)]
                    base = self._reconstruir(padre_id, arista_id, cid, (sx, sy, sth))
                    res = _acepta(base + maniobra)
                    if res is not None:
                        return res

            if k >= self.k_max:
                continue

            dprev = delta_prev.get(cid, 0.0)
            aprev = acc_prev.get(cid, 0.0)
            _k_expand(x, y, th, v, k, acc, dl, ns, h, L, self.a_max,
                      self.v_max_c, self.v_rev, self._len, self._wid,
                      self.margen, self.margen_din, self.diag, W, H,
                      oc, oax, obb, K, blc, blax, blbb, B,
                      rx, roff, rlen, rlw, rrad, R, feas, osub, ov)

            for ai in range(A):
                if not feas[ai]:
                    continue
                delta = dl[ai]
                nv = ov[ai]
                nx = osub[ai, ns - 1, 0]
                ny = osub[ai, ns - 1, 1]
                nth = osub[ai, ns - 1, 2]
                nk = k + ns
                # PROHIBICIÓN DURA: la búsqueda NO puede generar un nodo DETENIDO
                # (|v|<v_tol) dentro de la zona de la meta si el rumbo NO coincide
                # con el de llegada (sentido incluido). Así el coche NUNCA se para
                # al pasar por la meta mal orientado —sigue de largo y reorienta en
                # movimiento, o por su conector, que frena él mismo—. Solo puede
                # llegar a pararse si YA está bien orientado, y esa parada es la
                # definitiva (la que remata/aparca). Ningún conector necesita que
                # la búsqueda se detenga antes, así que esto no bloquea maniobras.
                if con_ang and abs(nv) < self.v_tol:
                    if (math.hypot(nx - gx, ny - gy) <= self.dist_tiro
                            and abs(ang_norm(nth - gth)) > ang_tol_din):
                        continue
                # COSTE = TIEMPO. Una ruta más larga acumula más pasos y sale peor
                # por sí sola; no se penaliza girar salvo desempates ε.
                ng = g + self.dt
                if nv < 0:
                    ng += 0.6 * self.dt               # marcha atrás: maniobra indeseada
                if abs(nv) < 1e-3 and acc[ai] <= 0.0:
                    ng += 0.20 * self.dt              # pararse sin motivo cuesta tiempo
                if con_ang:
                    d_goal_next = math.hypot(nx - gx, ny - gy)
                    av_n = abs(nv)
                    # PARÓN cerca de la meta: SIEMPRE artefacto del pozo del
                    # heurístico. Ya NINGÚN conector necesita llegar parado —la
                    # maniobra antepone su propio frenado antes de invertir el
                    # sentido—, así que detenerse «por si luego hay que ir marcha
                    # atrás» no aporta nada: se encarece SIN mirar el ángulo. El
                    # coche que pasa recto junto a la meta sigue recto; si toca
                    # invertir, el conector frena él mismo (decisión, no espera).
                    if d_goal_next <= self.dist_tiro and av_n < self.v_tol:
                        ng += 3.0 * self.dt * (self.v_tol - av_n) / self.v_tol
                    # ENVOLVENTE DE APROXIMACIÓN (frenado progresivo de "etapa 1"):
                    # llegar a la zona de maniobra (dist_tiro) ya a velocidad de
                    # maniobra. Fuera de la zona la envolvente crece con la
                    # parábola física v² = v_man² + 2·a·d_extra, así el descenso es
                    # gradual, empieza donde toca según v y a_max, y NO depende
                    # del ángulo. Evita entrar tan rápido que no pueda parar y
                    # tenga que dar una vuelta para reintentar.
                    v_man_c = V_MAN_FRAC * self.v_max_c
                    d_ext = d_goal_next - self.dist_tiro
                    if d_ext < 0.0:
                        d_ext = 0.0
                    v_env = math.sqrt(v_man_c * v_man_c + 2.0 * self.a_max * d_ext)
                    if av_n > v_env:
                        ng += 2.0 * self.dt * (av_n - v_env) / self.v_max_c
                ng += 0.10 * self.dt * abs(delta) / dmax          # ε: prefiere ir recto
                # ε: sin temblor de volante. Reforzado: mantener el MISMO ángulo
                # de dirección a lo largo de la curva sale mucho más barato que
                # ir alternándolo, así la curvatura es estable y el giro sale
                # limpio (sin volantazos en mitad del arco).
                ng += 0.30 * self.dt * abs(delta - dprev)
                # ε: sin temblor de acelerador (jerk). Sin él, alternar
                # acelerar/frenar sale gratis (el coste es solo tiempo) y el
                # planificador encadena frenadas y arranques sin sentido al
                # girar —bastaba con reducir un poco la velocidad—. Reforzado
                # para eliminar el "frena y vuelve a arrancar" al entrar en las
                # curvas. |Δacc|/a_max ∈ {0,1,2}; una inversión +a→−a cuesta
                # 0.80·dt: disuade el vaivén con firmeza, pero sigue permitiendo
                # frenar de verdad cuando hace falta.
                ng += 0.40 * self.dt * abs(acc[ai] - aprev) / self.a_max
                key = self._clave(nx, ny, nth, nv, nk)
                prev_g = mejor_g.get(key)
                if prev_g is None or ng < prev_g:
                    mejor_g[key] = ng
                    nid += 1
                    padre_id[nid] = cid
                    arista_id[nid] = osub[ai].copy()
                    delta_prev[nid] = delta
                    acc_prev[nid] = acc[ai]
                    hh = self._h_time(nx, ny)
                    if con_ang:
                        dx_g = gx - nx; dy_g = gy - ny
                        d_g = math.hypot(dx_g, dy_g)
                        if d_g > 0.1:
                            ang_to_g = math.atan2(dy_g, dx_g)
                            err_app = abs(ang_norm(gth - ang_to_g))
                            err_th = abs(ang_norm(nth - gth))
                            # CIERRE DEL POZO: el error de RUMBO respecto al ángulo
                            # de llegada cuesta una reorientación que NO se abarata
                            # al acercarse a la meta (arreglar el rumbo exige al
                            # menos un arco de radio r_min, y para acabar EN la meta
                            # suele hacer falta salir y volver). Antes este término
                            # iba escalado por len/v_max (~medio segundo) y se
                            # desvanecía: un nodo que pasaba junto a la meta MAL
                            # ENCARADO le parecía «casi hecho» a la búsqueda (h≈0),
                            # que entonces se metía en él frenando de paso. Ahora un
                            # rumbo equivocado mantiene h alto —no hay incentivo en
                            # reducir la velocidad si no se va a acabar—; solo el
                            # nodo YA encarado (err_th≈0) tiene h≈0 y «va a acabar».
                            hh += 1.6 * err_th * (r_min / self.v_max_c)
                            hh += 0.35 * err_app * (self._len / self.v_max_c)
                    heapq.heappush(abierto,
                                   (ng + peso_din * hh, nid, nx, ny, nth, nv, nk, ng))

        # Búsqueda agotada: si hay alguna candidata (precisa relajada o parada
        # de respaldo), vale la mejor de ellas.
        self.expandidos = expand
        self.nodos = nid
        fb = _cerrar_fallback()
        if fb is not None:
            return fb
        # Frontera vacía → no existe ruta (definitivo). Si quedaban nodos, se
        # alcanzó el techo de nodos/expansiones → resultado inconcluso.
        if not abierto:
            self.motivo = "sin_ruta"          # sin salida: algo lo bloquea
        elif nid >= self.max_nodos:
            self.motivo = "techo_nodos"       # agotó el presupuesto de nodos
        else:
            self.motivo = "limite"            # agotó expansiones/tiempo
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
    rrad = np.array([1.5])
    acc = np.array([1.0, 0.0]); dl = np.array([0.1, -0.1])
    feas = np.empty(2, np.int8); osub = np.empty((2, 4, 3)); ov = np.empty(2)
    _k_expand(2.0, 2.0, 0.0, 0.0, 0, acc, dl, 4, 0.1, 0.7, 1.0,
              2.5, 1.25, 1.3, 0.7, 0.1, 0.15, 1.48, W, H,
              oc, oax, obb, 1, eb, eb, ebb, 0, rx, roff, rlen, rlw, rrad, 1, feas, osub, ov)
    _k_tiro(2.0, 2.0, 0.0, 0.0, 0, 6.0, 6.0, 0.0, 1, 0.2,
            0.7, 0.6, 2.0, 2.5, 1.0, 1.6, 0.45, 0.1,
            1.3, 0.7, 0.1, 0.15, 1.48, W, H, oc, oax, obb, 1,
            eb, eb, ebb, 0, rx, roff, rlen, rlw, rrad, 1)
    _k_aparca(2.0, 2.0, 0.0, 0, 1, 1.6, 1.0, rx, roff, rlen, rlw, rrad, 1, 1.48, 0.15)
    _k_maniobra(2.0, 2.0, 0.0, 0.0, 0, 6.0, 6.0, 0.0, 0.2,
                0.7, 0.6, 2.0, 2.5, 1.0, 0.2, 0.45, 0.1,
                1.3, 0.7, 0.1, 0.15, 1.48, W, H, oc, oax, obb, 1,
                eb, eb, ebb, 0, rx, roff, rlen, rlw, rrad, 1, 1)
    _k_maniobra2(2.0, 2.0, 0.0, 0, 6.0, 6.0, 0.0, 0.2,
                 0.7, 0.6, 2.0, 2.5, 1.0, 0.2, 0.45, 0.1,
                 1.3, 0.7, 0.1, 0.15, 1.48, W, H, oc, oax, obb, 1,
                 eb, eb, ebb, 0, rx, roff, rlen, rlw, rrad, 1, 1)
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
# PARADA POR HORNADAS: al explorar órdenes en paralelo, en vez de cortar cada
# vehículo por un plazo de tiempo (que causaba fallos «por reloj»), se les deja
# buscar hasta su tope de nodos y se procesan los órdenes en HORNADAS de tantos
# como núcleos (todas a la vez). Un contador ACUMULADO cuenta los vehículos que
# han LLEGADO sumando todas las hornadas vistas. Tras despachar k hornadas de n
# órdenes con m vehículos cada uno, el umbral es FRAC · k · n · m; en cuanto el
# acumulado lo alcanza, se corta la hornada en curso (sus vehículos aún sin
# planificar cuentan como fallo) y se pasa a la siguiente. Al correr una hornada
# entera a la vez (tamaño = núcleos), el umbral siempre es alcanzable dentro de
# ella: no hay órdenes en cola «sin contar» como cuando había más órdenes que
# núcleos. Así se reduce el fallo por falta de tiempo sin explorar eternamente.
FRAC_PARADA_GLOBAL = 0.95


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


def total_combinaciones(especs, modo):
    """Número TEÓRICO de órdenes de planificación distintos que un modo PODRÍA
    llegar a explorar, SIN el tope de max_cand (es el «de X» que se muestra junto
    a «Órdenes a explorar»):

      · "secuencial"  → 1  (un único orden determinista, no explora).
      · "global"      → n!  (todos permutan entre sí).
      · "prioridades" → producto de los factoriales del tamaño de cada bucket
                        (grupo, prioridad): dentro de cada bucket se permuta, pero
                        los buckets van en orden fijo.

    'especs' es una lista de diccionarios (con claves opcionales 'grupo' y
    'prioridad', por defecto 1) o de objetos con esos atributos. Devuelve 0 si la
    lista está vacía."""
    n = len(especs)
    if n == 0:
        return 0
    if modo == "secuencial":
        return 1
    if modo == "global":
        return math.factorial(n)

    # "prioridades"/grupos: agrupa por (grupo, prioridad) y multiplica factoriales.
    def _clave(e):
        if isinstance(e, dict):
            return (e.get("grupo", 1), e.get("prioridad", 1))
        return (getattr(e, "grupo", 1), getattr(e, "prioridad", 1))

    buckets = {}
    for e in especs:
        k = _clave(e)
        buckets[k] = buckets.get(k, 0) + 1
    total = 1
    for tam in buckets.values():
        total *= math.factorial(tam)
    return total


def bloqueos_metas(vehiculos, indices, excepto, pose0=None):
    """OBB cuadrados en las metas ajenas, para que nadie planifique aparcar
    donde otro vehículo debe aparcar.

    'pose0' es la pose INICIAL (x, y, th) del vehículo que va a planificar: un
    bloqueo que solape esa pose se OMITE. Sin esto, un vehículo que arranca
    pegado a la meta de otro queda atrapado dentro de un obstáculo invisible
    (todas sus acciones 'colisionan' desde el primer nodo) y la búsqueda
    devuelve 'sin_ruta' al instante aunque tenga espacio de sobra. La
    protección real de la plaza ajena la siguen dando las reservas dinámicas
    del otro vehículo y la comprobación de aparcamiento (_k_aparca)."""
    veh0 = vehiculos[excepto] if excepto is not None else None
    cA = aA = None
    if pose0 is not None and veh0 is not None:
        # OBB del vehículo en su pose inicial, con el mismo inflado (margen)
        # que usará la búsqueda al validar contra los bloqueos.
        cA = _k_corners(pose0[0], pose0[1], pose0[2],
                        veh0.length + 0.30, veh0.width + 0.30)
        aA = _k_axes(cA)
    bloq = []
    for j in indices:
        if j == excepto:
            continue
        veh = vehiculos[j]
        mx, my = veh.meta
        poly = obb_corners(mx, my, 0.0, veh.diag, veh.diag)
        if cA is not None:
            cB = np.array(poly)
            aB = _k_axes(cB)
            if _k_sat(cA, aA, cB, aB):
                continue                # atraparía al vehículo en su inicio
        bloq.append((poly, (mx, my, veh.diag / 2)))
    return bloq


def _cuenta_arribo(arribos):
    """Suma un vehículo LLEGADO al contador ACUMULADO compartido entre procesos.
    Quién decide la parada (comparar con el umbral de la hornada) es el proceso
    principal, no el worker."""
    if arribos is None:
        return
    with arribos.get_lock():
        arribos.value += 1


def planificar_orden(planificador, vehiculos, orden, reservas_base,
                     tick_veh=None, deadline_dur=None,
                     arribos=None, parar=None):
    """Planifica cooperativamente los vehículos de 'orden' (índices) sobre las
    reservas base.

    'deadline_dur', si se indica, es el presupuesto de tiempo (segundos) POR
    VEHÍCULO: se renueva justo antes de planificar cada uno. Sin esto, un
    'planificador.deadline' fijado por el llamador ANTES de la llamada queda
    compartido entre todos los vehículos del orden, de modo que los primeros
    (más fáciles, o simplemente los primeros en la cola) pueden agotar el
    plazo entero y dejar a los últimos sin apenas tiempo para buscar —el
    síntoma es que, con varios vehículos, solo los primeros consiguen ruta y
    el resto se queda "aparcado" aunque exista solución.

    'arribos'/'parar' implementan la PARADA POR HORNADAS (ver FRAC_PARADA_GLOBAL):
    'arribos' es un contador ACUMULADO de vehículos llegados que este orden va
    sumando; 'parar' es la bandera que el proceso principal pone a 1 al alcanzar
    el umbral de la hornada. Con 'parar' a 1, los vehículos que aún no se hayan
    planificado en este orden cuentan como fallo (no se les llega a buscar ruta).

    Devuelve (trayectorias, motivos, coste): dicts idx → traj / motivo, y el
    coste total de flota (suma de duraciones REALES de los vehículos que
    llegaron). Los fallos NO se penalizan en el coste: la selección ordena por
    nº de fallos primero y el coste solo desempata entre órdenes con los
    mismos fallos, así que 'coste' es tiempo real puro."""
    reservas = reservas_base.copia()
    trays = {}
    motivos = {}
    coste = 0.0
    for pos, idx in enumerate(orden):
        veh = vehiculos[idx]
        # Parada global: si otro proceso ya alcanzó el 90 % de llegadas, los
        # vehículos que quedan por planificar en este orden cuentan como fallo.
        if parar is not None and parar.value:
            trays[idx] = None
            motivos[idx] = "parada_global"
            continue
        if tick_veh is not None:
            tick_veh(pos, idx)
        planificador.bloqueos = bloqueos_metas(
            vehiculos, orden, excepto=idx, pose0=veh.inicio)
        if deadline_dur is not None:
            planificador.deadline = time.perf_counter() + deadline_dur
        traj = planificador.planificar(veh, reservas)
        motivos[idx] = planificador.motivo
        if traj is not None and len(traj) >= 2:
            trays[idx] = traj
            reservas.add(traj, veh.length, veh.width)
            coste += (len(traj) - 1) * DT
            _cuenta_arribo(arribos)
        elif traj:
            trays[idx] = [traj[0], traj[0]]
            reservas.add(trays[idx], veh.length, veh.width)
            _cuenta_arribo(arribos)
        else:
            trays[idx] = None
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


def _nodos_corto(n):
    """Formato compacto de un recuento de nodos para la barra de estado, para que
    quepan todos los núcleos: 1234→'1.2k', 45000→'45k', 3400000→'3.4M'."""
    if n < 1000:
        return str(n)
    if n < 999_500:
        return f"{n / 1000:.1f}k" if n < 9950 else f"{n / 1000:.0f}k"
    return f"{n / 1_000_000:.1f}M" if n < 9_950_000 else f"{n / 1_000_000:.0f}M"


# Estado propio de cada proceso worker: se construye UNA vez (en el inicializador)
# y se reutiliza para todos los órdenes que le toquen a ese proceso.
_WK = {}


def _wk_init(entorno, vehiculos, reservas_base, calidad, cola, contador,
             arribos, parar):
    """Inicializador de cada proceso: compila los núcleos (caché en disco), crea
    su propio planificador y le asigna un identificador de núcleo correlativo.
    'arribos'/'parar' son el estado compartido de la parada por hornadas."""
    warmup()
    pl = Planificador(entorno)
    pl.configurar_calidad(calidad)
    with contador.get_lock():
        wid = contador.value
        contador.value += 1
    _WK.update(pl=pl, vehiculos=vehiculos, reservas=reservas_base,
               cola=cola, wid=wid, ult=0.0,
               arribos=arribos, parar=parar)


def _wk_eval(args):
    """Evalúa UN orden candidato en el proceso worker y devuelve su resultado.
    Publica el progreso (nodos del vehículo en curso) en la cola compartida."""
    orden, max_exp, deadline_dur = args
    pl = _WK["pl"]
    cola = _WK["cola"]
    wid = _WK["wid"]
    parar = _WK["parar"]
    if max_exp is not None:              # None → se respeta el tope de calidad
        pl.max_exp = max_exp

    def tick(expand):
        # Parada global: en cuanto otro proceso alcanza el 90 % de llegadas,
        # abortamos la búsqueda en curso adelantando el deadline al pasado (el
        # bucle A* lo comprueba cada 256 expansiones y devuelve al instante).
        if parar is not None and parar.value:
            pl.deadline = time.perf_counter() - 1.0
        ahora = time.perf_counter()
        if ahora - _WK["ult"] >= 0.08:          # limita el tráfico entre procesos
            _WK["ult"] = ahora
            try:
                cola.put_nowait((wid, expand))
            except Exception:
                pass

    pl.tick = tick
    try:
        # 'deadline_dur' es POR VEHÍCULO (planificar_orden lo renueva al empezar
        # cada uno): antes se fijaba aquí un único plazo para el orden entero y
        # los primeros vehículos agotaban el tiempo de los últimos.
        trays, motivos, coste = planificar_orden(
            pl, _WK["vehiculos"], orden, _WK["reservas"],
            deadline_dur=deadline_dur,
            arribos=_WK["arribos"], parar=parar)
    finally:
        pl.tick = None
    fallos = sum(1 for i in orden if trays[i] is None)
    try:
        cola.put_nowait((wid, 0))               # ese núcleo queda libre
    except Exception:
        pass
    return orden, fallos, coste, trays, motivos


def evaluar_ordenes_hornadas(env, vehiculos, ordenes, reservas_base,
                             calidad, max_exp=None, progreso=None,
                             sigue_vivo=None):
    """Evalúa los órdenes candidatos POR HORNADAS de tantos órdenes como núcleos
    utilizables (una orden por núcleo, todas a la vez). Cada hornada se detiene en
    cuanto el contador ACUMULADO de vehículos llegados alcanza FRAC_PARADA_GLOBAL ·
    (vehículos vistos hasta esa hornada); los rezagados de la hornada en curso se
    cortan (cuentan como fallo) y se pasa a la siguiente. La usan TANTO el
    escritorio (tkinter) COMO el servidor web, cada uno con sus 'callbacks'.

    'max_exp': tope de expansiones por vehículo (None → el del nivel de calidad).
    'progreso(texto)': callback opcional de avance (barra de estado / web).
    'sigue_vivo()': predicado opcional; si devuelve False se aborta y retorna None
    (p. ej. ventana cerrada). Devuelve el mejor (fallos, coste, trays, motivos,
    orden), o None."""
    ctx = multiprocessing.get_context("spawn")
    nucleos = nucleos_disponibles()
    workers = max(1, min(len(ordenes), nucleos))
    m = len(ordenes[0]) if ordenes else 0     # vehículos por orden (igual en todas)
    cola = ctx.Queue()
    contador = ctx.Value("i", 0)
    # Estado compartido de la PARADA POR HORNADAS: 'arribos' cuenta —ACUMULADO
    # entre hornadas— los vehículos llegados; el proceso principal lo compara
    # con el umbral de la hornada y pone 'parar' a 1 para cortar los rezagados.
    arribos = ctx.Value("i", 0)
    parar = ctx.Value("i", 0)
    nodos = {}                     # wid -> últimos nodos reportados
    mejor = None
    clave_mejor = None             # (fallos, coste, idx): desempata por índice
    # Hornadas de 'workers' órdenes: dentro de cada una TODAS corren a la vez,
    # así que el umbral acumulado siempre es alcanzable dentro de la hornada.
    hornadas = [ordenes[i:i + workers]
                for i in range(0, len(ordenes), workers)]
    total_veh = len(ordenes) * m
    ex = ProcessPoolExecutor(
        max_workers=workers, mp_context=ctx, initializer=_wk_init,
        initargs=(env, vehiculos, reservas_base, calidad,
                  cola, contador, arribos, parar))
    try:
        oi_base = 0                # índice global del primer orden de la hornada
        vistos = 0                 # órdenes despachados hasta ahora (acumulado)
        for hj, hornada in enumerate(hornadas):
            parar.value = 0                       # nueva hornada: sin corte aún
            vistos += len(hornada)
            # Umbral ACUMULADO: 95 % de todos los vehículos vistos hasta aquí.
            umbral = max(1, math.ceil(FRAC_PARADA_GLOBAL * vistos * m))
            # dur=None: no se corta cada vehículo por reloj (eso causaba fallos
            # «por tiempo»); busca hasta su tope de nodos y es el umbral de
            # llegadas quien acota el tiempo de la hornada.
            fut_idx = {ex.submit(_wk_eval, (orden, max_exp, None)): oi_base + k
                       for k, orden in enumerate(hornada)}
            oi_base += len(hornada)
            pendientes = set(fut_idx)
            while pendientes:
                if sigue_vivo is not None and not sigue_vivo():
                    return None
                while True:        # vaciar la cola de progreso
                    try:
                        wid, expand = cola.get_nowait()
                    except queue.Empty:
                        break
                    nodos[wid] = expand
                # Umbral acumulado alcanzado → cortar los rezagados de la hornada.
                if not parar.value and arribos.value >= umbral:
                    parar.value = 1
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
                if progreso is not None:
                    detalle = " ".join(f"N{w + 1}:{_nodos_corto(nodos.get(w, 0))}"
                                       for w in range(workers))
                    progreso(
                        f"Hornada {hj + 1}/{len(hornadas)} · "
                        f"{min(arribos.value, total_veh)}/{total_veh} llegadas "
                        f"(corta en {umbral}) · {detalle}")
                time.sleep(0.02)
    finally:
        try:
            ex.shutdown(wait=False, cancel_futures=True)
        except TypeError:          # cancel_futures: Python < 3.9
            ex.shutdown(wait=False)
        cola.close()
    return mejor


# --------------------------------------------------------------------------- #
# Entrada de texto: lista de diccionarios de características
# --------------------------------------------------------------------------- #
# Claves admitidas por vehículo (los ángulos SIEMPRE en grados en el texto):
#
#   id              identificador del vehículo (entero o texto; si falta: su nº).
#   inicio          (x, y) punto inicial en metros
#   giro_inicial    orientación inicial en grados (si falta: mirando a la meta)
#   meta            (x, y) punto destino en metros
#   angulo_llegada  ángulo exigido al llegar, en grados; "libre" o None → sin
#                   exigencia (si falta: la dirección inicio→meta)
#   largo, ancho    dimensiones del vehículo en metros
#   v_max           velocidad máxima (m/s)
#   v_inicial       velocidad de arranque (m/s; si falta: 0). Positiva = avanzando,
#                   negativa = marcha atrás. Acotada a |v_inicial| ≤ v_max (atrás,
#                   a −0.5·v_max). En modo aleatorio siempre arranca de 0.
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
#     "v_inicial": 1.5, "a_max": 1.2, "giro_max": 33, "grupo": 1, "prioridad": 1},
#    {"id": 2, "inicio": (36, 3), "meta": (4, 20), "grupo": 2, "prioridad": 1}]
CLAVES_VEH = {"id", "inicio", "giro_inicial", "meta", "angulo_llegada",
              "largo", "ancho", "v_max", "v_inicial", "a_max", "giro_max",
              "grupo", "prioridad"}


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
        if "v_inicial" in d:
            e["v_inicial"] = _como_num(d["v_inicial"], "v_inicial", -12.0, 12.0)
            # La velocidad inicial no puede superar (en magnitud) la v_max del
            # vehículo; hacia atrás se limita a la marcha atrás máxima (0.5·v_max).
            vmx = e.get("v_max", VEH_VMAX)
            if e["v_inicial"] > vmx:
                raise ValueError(f"Vehículo {e['id']!r}: la v_inicial "
                                 f"({e['v_inicial']:g}) no puede superar la v_max "
                                 f"({vmx:g}).")
            if e["v_inicial"] < -0.5 * vmx:
                raise ValueError(f"Vehículo {e['id']!r}: la v_inicial marcha atrás "
                                 f"({e['v_inicial']:g}) no puede superar la marcha "
                                 f"atrás máxima (−0.5·v_max = {-0.5 * vmx:g}).")
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
                    vid=e["id"],
                    v_inicial=e.get("v_inicial", 0.0))


def especs_a_texto(especs):
    """Serializa una lista de diccionarios de especificación a texto editable
    (un vehículo por línea), reutilizable tal cual como entrada del programa."""
    return "[\n" + "".join(" " + repr(e) + ",\n" for e in especs) + "]\n"


TEXTO_EJEMPLO = (
    "[\n"
    ' {"id": 1, "inicio": (3, 3),  "giro_inicial": 0,  "meta": (36, 20),\n'
    '  "angulo_llegada": 90, "largo": 1.6, "ancho": 0.8, "v_max": 3.0,\n'
    '  "v_inicial": 1.5, "a_max": 1.2, "giro_max": 33, "grupo": 1, "prioridad": 1},\n'
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

# Plazos POR VEHÍCULO (se renuevan al empezar cada vehículo, vía deadline_dur):
# acotan el tiempo del caso difícil para que ningún vehículo cuelgue la
# aplicación minutos enteros, sin que los primeros del orden puedan comerse el
# presupuesto de los últimos (el bug que describe el docstring de
# planificar_orden).
PLAZO_VEH_CAND = 5.0         # s/vehículo al COMPARAR órdenes candidatos
#                              (fase de exploración de órdenes, interactividad
#                              de la UI; no participa en la decisión de
#                              relajar el ángulo ni en la de rendirse, que son
#                              puramente por NODOS: max_exp/max_nodos y los
#                              umbrales de relajación en configurar_calidad).


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
        # "Órdenes a explorar (máx)" + total de combinaciones posibles ("de X").
        # La casilla se desplaza a la izquierda y a su derecha, FUERA de ella, se
        # muestra el número teórico de órdenes que el modo podría llegar a probar.
        ttk.Label(panel, text="Órdenes a explorar (máx):").grid(
            row=fila[0], column=0, sticky="w", pady=1)
        sub = ttk.Frame(panel)
        sub.grid(row=fila[0], column=1, sticky="e", pady=1)
        self.e_cand = ttk.Entry(sub, width=5)
        self.e_cand.insert(0, "12")
        self.e_cand.pack(side="left")
        self.lbl_combos = ttk.Label(sub, text="de —")
        self.lbl_combos.pack(side="left", padx=(4, 0))
        fila[0] += 1

        sep()
        # Botón principal destacado.
        ttk.Button(panel, text="▶  Calcular y simular", style="Destacado.TButton",
                   command=self._seguro(self.calcular_y_simular)).grid(
            row=fila[0], column=0, columnspan=2, sticky="ew", pady=1)
        fila[0] += 1
        for txt, cmd in (("↺  Reproducir de nuevo", self.reproducir),
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
        ttk.Label(caja, text="Vehículos manuales "
                             "(lista de diccionarios; ángulos en grados):").grid(
            row=0, column=0, sticky="w")
        self.texto = tk.Text(caja, height=8, width=int(W * SCALE / 8),
                             font=("Menlo", 10), undo=True)
        self.texto.grid(row=1, column=0, sticky="ew")
        self.texto.insert("1.0", TEXTO_EJEMPLO)

        # El total de combinaciones («de X») se recalcula al cambiar de modo de
        # optimización, al editar los vehículos y al (re)generarlos. Cambiar el
        # modo a ALEATORIO o el nº de vehículos regenera las instrucciones para
        # que las preferencias queden definidas y el total se pueda calcular.
        self._ultimo_num = self.e_num.get()
        self.opt.trace_add("write", self._actualizar_combos)
        self.modo.trace_add("write", self._modo_cambia)
        self.e_num.bind("<Return>", self._num_cambia)
        self.e_num.bind("<FocusOut>", self._num_cambia)
        self.texto.bind("<KeyRelease>", self._actualizar_combos)
        self._actualizar_combos()

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

    def _actualizar_combos(self, *_):
        """Recalcula el total de órdenes posibles («de X») a partir de las
        instrucciones actuales de la caja de texto y del modo de optimización
        elegido. Si el texto aún no es válido, muestra «de —»."""
        if not hasattr(self, "lbl_combos"):
            return
        try:
            especs = parsear_especificaciones(self.texto.get("1.0", "end"))
            total = total_combinaciones(especs, self.opt.get())
            self.lbl_combos.config(text=f"de {total:,}")
        except Exception:
            self.lbl_combos.config(text="de —")

    def _modo_cambia(self, *_):
        """Al pasar a ALEATORIO se regeneran las instrucciones (para que las
        preferencias queden definidas y el total sea calculable); en manual solo
        se refresca el total con lo que ya hay en el texto."""
        if self.modo.get() == "aleatorio":
            self._seguro(self.generar_posiciones)()
        else:
            self._actualizar_combos()

    def _num_cambia(self, *_):
        """Cambiar el nº de vehículos (en modo aleatorio) regenera las
        instrucciones; solo si el valor realmente cambió, para no rehacer la
        flota al mover el foco sin editar."""
        actual = self.e_num.get()
        if actual == getattr(self, "_ultimo_num", None):
            return
        self._ultimo_num = actual
        if self.modo.get() == "aleatorio":
            self._seguro(self.generar_posiciones)()
        else:
            self._actualizar_combos()

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
            polys = imagen_a_poligonos(ruta)
        except Exception as e:  # noqa: BLE001
            self._ocupado = False
            messagebox.showerror("Imagen no válida",
                                 f"No se pudo leer/convertir la imagen:\n{e}\n\n"
                                 "Sin Pillow instalado solo se admiten PNG/GIF/PPM "
                                 "(pip install pillow para el resto).")
            return
        self._ocupado = False
        if len(polys) == 0:
            messagebox.showwarning("Mapa vacío",
                                   "La imagen no contiene píxeles oscuros: el mapa "
                                   "queda completamente libre.")
        nombre = os.path.splitext(os.path.basename(ruta))[0]
        self.env.desde_poligonos(polys, nombre)
        self._mapa_cambiado(f"Mapa importado de «{nombre}» "
                            f"({len(polys)} bloques). Genera vehículos.")
        if messagebox.askyesno("Guardar mapa",
                               "¿Guardar este mapa en la carpeta «mapas/» para "
                               "poder reutilizarlo en otras ejecuciones?"):
            nom = simpledialog.askstring("Nombre del mapa",
                                         "Nombre con el que guardarlo:",
                                         initialvalue=nombre)
            if nom:
                ruta_j = guardar_mapa(nom, polys=polys)
                self.estado.set(f"Mapa guardado en {ruta_j}")

    def guardar_mapa_actual(self):
        from tkinter import messagebox, simpledialog
        if self.env.origen == "aleatorio" or (self.env.rects is None
                                              and not self.env.obstaculos):
            messagebox.showinfo("Solo mapas importados",
                                "Se guardan los mapas creados desde una imagen. "
                                "Importa una imagen primero.")
            return
        nom = simpledialog.askstring("Nombre del mapa",
                                     "Nombre con el que guardarlo:",
                                     initialvalue=self.env.origen)
        if nom:
            if self.env.rects is not None:       # mapa antiguo alineado a ejes
                ruta_j = guardar_mapa(nom, self.env.rects)
            else:                                 # cajas orientadas
                ruta_j = guardar_mapa(nom, polys=self.env.obstaculos)
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
            n = cargar_mapa_en(self.env, ruta)
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Mapa corrupto", f"No se pudo cargar:\n{e}")
            return
        nombre = os.path.splitext(os.path.basename(ruta))[0]
        self._mapa_cambiado(f"Mapa «{nombre}» cargado "
                            f"({n} bloques). Genera vehículos.")

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
            self._actualizar_combos()
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
        self._ultimo_num = self.e_num.get()
        self._actualizar_combos()
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
            largo = random.uniform(0.7, 1.2)
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
                "v_max": round(random.uniform(1.8, 3), 2),
                "a_max": round(random.uniform(0.6, 1), 2),
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
                indices, base,
                reubicable=(self.modo.get() == "aleatorio"))
            self._aplicar_resultado(indices, trays, motivos)
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

    def _planificar_flota(self, indices, reservas_base, reubicable):
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
                    mejor = self._evaluar_ordenes_paralelo(
                        ordenes, reservas_base, cap_prev)
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
                        # Presupuesto por candidato: cap de calidad + plazo POR
                        # VEHÍCULO, para que explorar muchos órdenes siga siendo
                        # interactivo y ningún vehículo se coma el plazo ajeno.
                        pl.max_exp = cap_prev
                        dur_veh = PLAZO_VEH_CAND
                    else:
                        # Orden DEFINITIVO: sin plazo de tiempo, solo nodos
                        # (cap_completo). El "cuándo se rinde" es max_exp/
                        # max_nodos y la relajación del ángulo por nodos-en-
                        # meta; nada de segundos aquí.
                        pl.max_exp = CAP_COMPLETO
                        pl.deadline = None
                        dur_veh = None
                    trays, motivos, coste = planificar_orden(
                        pl, self.vehiculos, orden, reservas_base,
                        tick_veh=tick_veh, deadline_dur=dur_veh)
                    fallos = sum(1 for i in orden if trays[i] is None)
                    if mejor is None or (fallos, coste) < (mejor[0], mejor[1]):
                        mejor = (fallos, coste, trays, motivos, orden)
                    if fallos == 0 and not explorar:
                        break

            fallos, _, trays, motivos, orden = mejor
            # ---- rescate de los vehículos sin ruta en el mejor orden ---- #
            if fallos:
                pl.max_exp = CAP_COMPLETO
                pl.deadline = None            # solo nodos, nada de segundos
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
                    pl.bloqueos = bloqueos_metas(
                        self.vehiculos, orden, excepto=i, pose0=veh.inicio)
                    pl.deadline = None            # solo nodos
                    traj = pl.planificar(veh, reservas)
                    motivos[i] = pl.motivo
                    intentos = 0
                    while (pl.motivo == "sin_ruta" and reubicable
                           and intentos < 12):
                        if not self._reubicar(veh):
                            break
                        pl.bloqueos = bloqueos_metas(self.vehiculos, orden,
                                                     excepto=i, pose0=veh.inicio)
                        pl.deadline = None            # solo nodos
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

    def _evaluar_ordenes_paralelo(self, ordenes, reservas_base, max_exp):
        """Envoltorio de la evaluación POR HORNADAS (evaluar_ordenes_hornadas)
        con los 'callbacks' de la interfaz de escritorio: refresca la barra de
        estado y aborta si se cierra la ventana. Devuelve el mejor (fallos,
        coste, trays, motivos, orden) o None."""
        calidad = int(round(float(self.calidad.get())))

        def progreso(texto):
            self._plan_msg = texto
            self.estado.set(texto)
            self.root.update()

        return evaluar_ordenes_hornadas(
            self.env, self.vehiculos, ordenes, reservas_base,
            calidad, max_exp=max_exp, progreso=progreso,
            sigue_vivo=self.root.winfo_exists)

    def _aplicar_resultado(self, indices, trays, motivos):
        """Vuelca las trayectorias en los vehículos, reconstruye los fotogramas e
        informa del resultado."""
        from tkinter import messagebox
        if not self.root.winfo_exists():
            return
        ok = 0; sin_ruta = 0; inconcluso = 0
        for i in indices:
            veh = self.vehiculos[i]
            traj = trays.get(i)
            if traj is None:
                veh.mision_ok = False
                veh.traj = []
                if motivos.get(i) == "sin_ruta":
                    sin_ruta += 1
                else:
                    inconcluso += 1
                continue
            veh.mision_ok = True
            ok += 1
            veh.traj = traj
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
        self.frame = 0
        self.reproduciendo = True
        self._anim()

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
                self.estado.set("Reproducción finalizada.")
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
                ax = mx + 0.7 * math.cos(veh.meta_th)
                ay = my + 0.7 * math.sin(veh.meta_th)
                c.create_line(mx * SCALE, my * SCALE, ax * SCALE, ay * SCALE,
                              fill=col, width=1, arrow="last",
                              arrowshape=(6, 7, 2))
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
