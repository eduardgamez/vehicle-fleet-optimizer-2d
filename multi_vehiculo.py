#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulador físico de enrutamiento multi-vehículo en ESPACIO CONTINUO.

Modelo:

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

Arquitectura:

  · PLANIFICACIÓN — Hybrid A* cooperativo sobre el estado continuo (x, y, θ, v)
    con primitivas de arco del modelo de bicicleta. Cada vehículo evita los
    obstáculos fijos, las metas ajenas y las trayectorias temporales de los
    vehículos ya planificados (obstáculos móviles en el tiempo). El coste es el
    TIEMPO, con una heurística Dijkstra-con-obstáculos y una conexión analítica
    (pure-pursuit) para rematar la llegada.

  · EJECUCIÓN — se reproducen las trayectorias resultantes, ya libres de colisión.

Rendimiento:

  El núcleo numérico (geometría OBB/SAT, integración del modelo de bicicleta y
  los chequeos de colisión) es lo que domina el cálculo: en el perfilado se lleva
  la inmensa mayoría del tiempo. Aquí se compila a CÓDIGO NATIVO en el arranque
  con Numba, operando sobre arrays NumPy empaquetados, en lugar de interpretarse
  en Python. Numba genera instrucciones para la microarquitectura concreta del
  equipo donde se ejecuta, así que se adapta al hardware (Apple Silicon, x86, …).
  El resultado son planificaciones varias veces más rápidas, y las rutas son las
  mismas: la búsqueda A* conserva su orden de expansión y sus desempates.

  Requiere:   numpy, numba   (pip install numpy numba)
  Ejecutar:   python3 multi_vehiculo_opt.py
"""

import math
import time
import heapq
import random

from math import cos, sin, tan, hypot, sqrt, atan2

import numpy as np
from numba import njit


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

VEH_LEN = 1.3       # largo del vehículo (m)
VEH_WID = 0.7       # ancho del vehículo (m)
VEH_VMAX = 2.5      # velocidad máxima (m/s)


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


# =========================================================================== #
# NÚCLEOS NUMÉRICOS (compilados a código nativo por Numba)
#
# Toda la geometría de colisión y la integración del modelo de bicicleta operan
# sobre arrays NumPy empaquetados. La evaluación de las acciones de cada nodo del
# A* se reparte entre los núcleos de la CPU (prange).
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
def _k_expand(x0, y0, th0, v0, k, acc, dtan, subpasos, h, L,
              vmaxc, vrev, veh_len, veh_wid, margin, mdin, diag,
              worldW, worldH, oc, oax, obb, K, blc, blax, blbb, B,
              rx, roff, rlen, rlw, R, feas, osub, ov):
    """Evalúa TODAS las acciones (accel, delta) de un nodo. Para cada acción integra
    los 'subpasos' del modelo de bicicleta y valida en cada subpaso los obstáculos
    fijos, las metas bloqueadas y las reservas dinámicas. Rellena, por acción a:
    feas[a] = 1/0,  osub[a] = subposes,  ov[a] = velocidad final."""
    A = acc.shape[0]
    Lm = veh_len + 2.0 * margin
    Wm = veh_wid + 2.0 * margin
    Ld = veh_len + 2.0 * mdin
    Wd = veh_wid + 2.0 * mdin
    for a in range(A):
        x = x0; y = y0; th = th0; v = v0
        ok = 1
        ac = acc[a]; dtn = dtan[a]
        for sp in range(subpasos):
            v = min(vmaxc, max(-vrev, v + ac * h))
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
def _k_tiro(x, y, th, v, k, gx, gy, L, dmax, ld, vmaxc, amax, goal_tol, v_tol, h,
            veh_len, veh_wid, margin, mdin, diag, worldW, worldH,
            oc, oax, obb, K, blc, blax, blbb, B, rx, roff, rlen, rlw, R):
    """Conexión analítica (pure-pursuit con frenado hasta detenerse en la meta).
    Devuelve (n, poses): n>=0 → éxito con n poses en poses[:n]; n<0 → sin remate.
    Un tope de longitud descarta los rizos que se enroscan sin encarar la meta.

    'ld' es la distancia de anticipación (lookahead) del pure-pursuit, acotada:
    corta → corrige el rumbo con firmeza al inicio del remate y luego avanza en
    recta hasta la meta (en vez de un arco amplio y poco pronunciado)."""
    out = np.empty((600, 3))
    n = 0
    Lm = veh_len + 2.0 * margin; Wm = veh_wid + 2.0 * margin
    Ld = veh_len + 2.0 * mdin; Wd = veh_wid + 2.0 * mdin
    d0 = hypot(gx - x, gy - y)
    largo = 0.0
    largo_max = 1.5 * d0 + 2.0 * goal_tol
    kk = k
    for _ in range(600):
        d = hypot(gx - x, gy - y)
        if d <= goal_tol and abs(v) <= v_tol:
            return n, out
        dd = d if d > 0.0 else 0.0
        v_des = min(vmaxc, sqrt(2.0 * amax * dd))
        dv = max(-amax * h, min(amax * h, v_des - v))
        v = min(vmaxc, max(0.0, v + dv))
        alpha = atan2(gy - y, gx - x) - th
        alpha = atan2(sin(alpha), cos(alpha))
        # Lookahead acotado: nunca mayor que 'ld', para que la corrección de rumbo
        # sea firme al principio y el tramo restante quede recto. Cerca de la meta
        # se contrae a la distancia real para homing preciso.
        den = d if d < ld else ld
        if den < 1e-3:
            den = 1e-3
        delta = max(-dmax, min(dmax, atan2(2.0 * L * sin(alpha), den)))
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

        self.obstaculos = obs
        if obs:
            self.np_c, self.np_ax, self.np_bb = _pack_polys(obs)
        else:
            self.np_c, self.np_ax, self.np_bb = _EMPTY_C, _EMPTY_C, _EMPTY_BB

    def libre(self, x, y, theta, length, width, margen=0.0):
        """¿Cabe el vehículo (OBB) sin tocar bordes ni obstáculos?"""
        Lm = length + 2.0 * margen
        Wm = width + 2.0 * margen
        return bool(_k_free_sb(x, y, theta, Lm, Wm, W, H,
                               self.np_c, self.np_ax, self.np_bb, self.np_c.shape[0],
                               _EMPTY_C, _EMPTY_C, _EMPTY_BB, 0, 1.0, 0.0))


# --------------------------------------------------------------------------- #
# Vehículo  (su trayectoria es una pose por paso fino de DT segundos)
# --------------------------------------------------------------------------- #
class Vehiculo:
    def __init__(self, idx, inicio, meta, length, width, v_max):
        self.idx = idx
        self.inicio = inicio          # (x, y, theta)
        self.meta = meta              # (x, y)
        self.length = length
        self.width = width
        self.wheelbase = 0.55 * length
        self.delta_max = math.radians(35)     # ángulo de dirección máximo
        self.v_max = v_max
        self.a_max = 1.0              # aceleración/frenado máximos (m/s²)
        self.traj = []                # [(x, y, theta), ...]
        self.dt_plan = DT

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
        self.goal_tol = 1.6
        self.v_tol = 0.45
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

    def configurar_calidad(self, nivel):
        """Equilibrio TIEMPO ↔ CALIDAD de ruta. A mayor nivel: más ángulos de giro
        (más densos cerca de 0 para corregir el rumbo con finura), resolución
        angular más fina y más expansiones permitidas.

        El 3er valor es el PESO del heurístico (A* ponderado): >1 acelera pero
        aleja del óptimo → rutas más largas y curvas. Antes iba de 2.3 a 1.9 (muy
        inflado incluso en calidad máxima), lo que hacía que SIEMPRE salieran arcos
        suaves poco pronunciados en vez de la ruta más corta. Ahora baja de verdad
        con la calidad: el óptimo es «recto cuando conviene, giro firme cuando hace
        falta», y esa forma emerge sola al acercar el peso a 1. Medido en mapas con
        obstáculos y reservas, peso≈1.5 (nivel 3) no pierde robustez; por debajo de
        ~1.2 los casos difíciles se disparan en tiempo, reservado a la calidad alta."""
        tabla = {
            1: (9,  14, 2.0,  300000),
            2: (13, 13, 1.7,  500000),
            3: (17, 12, 1.5,  800000),
            4: (27, 10, 1.3, 1400000),
            5: (41,  8, 1.15, 2400000),
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
        calcula una vez por mapa y se reutiliza para todos los vehículos."""
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
        frenar en cualquier momento si eso da una ruta mejor."""
        self._len, self._wid = veh.length, veh.width
        self.diag = veh.diag
        L = veh.wheelbase
        dmax = veh.delta_max
        sx, sy, sth = veh.inicio
        gx, gy = veh.meta
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
        dtan = np.tan(dl)

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
                if self.tick is not None and now - self._last_tick > 0.15:
                    self._last_tick = now
                    self.tick(expand)

            d_goal = math.hypot(x - gx, y - gy)
            if d_goal <= self.goal_tol and abs(v) <= self.v_tol:
                # Solo acepta si la plaza de aparcamiento queda libre para siempre.
                Ld = self._len + 2.0 * self.margen_din
                Wd = self._wid + 2.0 * self.margen_din
                kfin = k if k > res_maxlen else res_maxlen
                if _k_aparca(x, y, th, k, kfin, Ld, Wd,
                             rx, roff, rlen, rlw, R, self.diag, self.margen_din):
                    self.motivo = "ok"
                    return self._reconstruir(padre_id, arista_id, cid, (sx, sy, sth))

            elif d_goal <= self.dist_tiro:
                # Remata solo si ya apunta bien a la meta (o está muy cerca), para
                # no trazar un arco amplio desde una orientación mala.
                alpha0 = math.atan2(gy - y, gx - x) - th
                alpha0 = math.atan2(math.sin(alpha0), math.cos(alpha0))
                if abs(alpha0) <= self.ang_tiro or d_goal <= self.goal_tol * 1.5:
                    n, poses = _k_tiro(x, y, th, v, k, gx, gy, L, dmax, self.ld_tiro,
                                       self.v_max_c, self.a_max, self.goal_tol,
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

            if k >= self.k_max:
                continue

            dprev = delta_prev.get(cid, 0.0)
            _k_expand(x, y, th, v, k, acc, dtan, ns, h, L,
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
    acc = np.array([1.0, 0.0]); dtan = np.tan(np.array([0.1, -0.1]))
    feas = np.empty(2, np.int8); osub = np.empty((2, 4, 3)); ov = np.empty(2)
    _k_expand(2.0, 2.0, 0.0, 0.0, 0, acc, dtan, 4, 0.1, 0.7,
              2.5, 1.25, 1.3, 0.7, 0.1, 0.15, 1.48, W, H,
              oc, oax, obb, 1, eb, eb, ebb, 0, rx, roff, rlen, rlw, 1, feas, osub, ov)
    _k_tiro(2.0, 2.0, 0.0, 0.0, 0, 6.0, 6.0, 0.7, 0.6, 2.0, 2.5, 1.0, 1.6, 0.45, 0.1,
            1.3, 0.7, 0.1, 0.15, 1.48, W, H, oc, oax, obb, 1,
            eb, eb, ebb, 0, rx, roff, rlen, rlw, 1)
    _k_aparca(2.0, 2.0, 0.0, 0, 1, 1.6, 1.0, rx, roff, rlen, rlw, 1, 1.48, 0.15)
    _k_occ(4, 4, 0.5, 0.45, W, H, oc, oax, obb, 1)


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
class App:
    def __init__(self, root):
        import tkinter as tk
        self.tk = tk
        self.root = root
        root.title("Simulador multi-vehículo · espacio continuo + SAT  "
                   "[núcleo nativo Numba]")
        root.resizable(False, False)

        self.env = Entorno()
        self.planificador = Planificador(self.env)
        self.vehiculos = []
        self.inicios = []
        self.metas = []
        self.frames = []
        self.frame = 0
        self.anim_id = None
        self.reproduciendo = False
        self._warmed = False

        self.modo_manual = False
        self.colocando_inicio = True
        self.pend_inicio = None
        self._ocupado = False
        self._plan_msg = ""

        self._construir_ui()
        self.env.generar(0.0)
        self._dibujar_estatico()

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

    def _construir_ui(self):
        tk = self.tk
        from tkinter import ttk
        cont = ttk.Frame(self.root, padding=8)
        cont.grid(row=0, column=0)

        panel = ttk.Frame(cont, padding=(0, 0, 12, 0))
        panel.grid(row=0, column=0, sticky="n")

        ttk.Label(panel, text="Parámetros", font=("", 13, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 6))
        self.e_num = self._campo(panel, 1, "Nº de vehículos:", "4")
        self.e_dens = self._campo(panel, 2, "Densidad obstáculos (0-1):", "0.0")

        self.calidad = tk.IntVar(value=3)
        self.calidad_txt = tk.StringVar()
        ttk.Label(panel, textvariable=self.calidad_txt, width=38,
                  anchor="w").grid(
            row=3, column=0, columnspan=2, sticky="w", pady=(6, 0))
        tk.Scale(panel, from_=1, to=5, resolution=1, tickinterval=1,
                 orient="horizontal", variable=self.calidad, showvalue=False,
                 command=self._calidad_cambia).grid(
            row=4, column=0, columnspan=2, sticky="ew")
        self._calidad_cambia()

        ttk.Separator(panel, orient="horizontal").grid(
            row=6, column=0, columnspan=2, sticky="ew", pady=8)
        ttk.Label(panel, text="Posiciones", font=("", 11, "bold")).grid(
            row=7, column=0, columnspan=2, sticky="w")
        self.modo = tk.StringVar(value="aleatorio")
        ttk.Radiobutton(panel, text="Aleatorias", variable=self.modo,
                        value="aleatorio").grid(row=8, column=0, columnspan=2, sticky="w")
        ttk.Radiobutton(panel, text="Manuales (clic: inicio → destino)",
                        variable=self.modo, value="manual").grid(
            row=9, column=0, columnspan=2, sticky="w")
        ttk.Button(panel, text="Generar posiciones",
                   command=self._seguro(self.generar_posiciones)).grid(
            row=10, column=0, columnspan=2, sticky="ew", pady=(6, 2))
        ttk.Button(panel, text="Nuevo mapa de obstáculos",
                   command=self._seguro(self.nuevo_mapa)).grid(
            row=11, column=0, columnspan=2, sticky="ew", pady=2)

        ttk.Separator(panel, orient="horizontal").grid(
            row=12, column=0, columnspan=2, sticky="ew", pady=8)
        ttk.Label(panel, text="Velocidad de reproducción").grid(
            row=13, column=0, columnspan=2, sticky="w")
        self.vel = tk.IntVar(value=40)
        ttk.Scale(panel, from_=160, to=8, variable=self.vel,
                  orient="horizontal").grid(row=14, column=0, columnspan=2, sticky="ew")

        ttk.Separator(panel, orient="horizontal").grid(
            row=15, column=0, columnspan=2, sticky="ew", pady=8)
        ttk.Button(panel, text="▶  Calcular y simular",
                   command=self._seguro(self.calcular_y_simular)).grid(
            row=16, column=0, columnspan=2, sticky="ew", pady=2)
        ttk.Button(panel, text="↺  Reproducir de nuevo",
                   command=self._seguro(self.reproducir)).grid(
            row=17, column=0, columnspan=2, sticky="ew", pady=2)
        ttk.Button(panel, text="⏸  Pausar / reanudar",
                   command=self._seguro(self.pausar)).grid(
            row=18, column=0, columnspan=2, sticky="ew", pady=2)
        ttk.Button(panel, text="⟲  Reiniciar",
                   command=self._seguro(self.reiniciar)).grid(
            row=19, column=0, columnspan=2, sticky="ew", pady=2)
        ttk.Button(panel, text="✕  Salir",
                   command=self.root.destroy).grid(
            row=20, column=0, columnspan=2, sticky="ew", pady=(2, 0))

        self.canvas = tk.Canvas(cont, width=int(W * SCALE), height=int(H * SCALE),
                                bg=COL_FONDO, highlightthickness=2,
                                highlightbackground=COL_BORDE)
        self.canvas.grid(row=0, column=1)
        self.canvas.bind("<Button-1>", self._seguro(self.click_mapa))

        self.estado = tk.StringVar(value="Listo. Ajusta parámetros y genera posiciones.")
        ttk.Label(cont, textvariable=self.estado, relief="sunken",
                  anchor="w", padding=4, width=1).grid(
            row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))

    def _campo(self, panel, fila, etiqueta, valor):
        from tkinter import ttk
        ttk.Label(panel, text=etiqueta).grid(row=fila, column=0, sticky="w", pady=1)
        e = ttk.Entry(panel, width=7)
        e.insert(0, valor)
        e.grid(row=fila, column=1, sticky="e", pady=1)
        return e

    def _calidad_cambia(self, *_):
        nivel = int(round(float(self.calidad.get())))
        self.calidad_txt.set(f"Calidad de ruta (1 rápida ⟷ 5 máxima):  {nivel}")
        self.planificador.configurar_calidad(nivel)

    def _params(self):
        from tkinter import messagebox
        length, width, vmax = VEH_LEN, VEH_WID, VEH_VMAX
        try:
            n = int(self.e_num.get())
            dens = float(self.e_dens.get())
        except ValueError:
            messagebox.showerror("Error", "Los parámetros deben ser numéricos.")
            return None
        if not (1 <= n <= len(PALETA)):
            messagebox.showerror("Error", f"Nº de vehículos entre 1 y {len(PALETA)}.")
            return None
        return n, length, width, vmax, max(0.0, min(1.0, dens))

    def nuevo_mapa(self):
        p = self._params()
        dens = p[4] if p else 0.0
        self._detener()
        self.env.generar(dens)
        self.inicios, self.metas, self.frames, self.vehiculos = [], [], [], []
        self.frame = 0
        self.estado.set("Nuevo mapa generado. Genera posiciones.")
        self._dibujar_estatico()

    def generar_posiciones(self):
        p = self._params()
        if not p:
            return
        n, length, width, vmax, _ = p
        self._detener()
        self.frames = []
        self.frame = 0

        if self.modo.get() == "manual":
            self.modo_manual = True
            self.colocando_inicio = True
            self.pend_inicio = None
            self.inicios, self.metas, self.vehiculos = [], [], []
            self._cfg = (n, length, width, vmax)
            self.estado.set("MANUAL · Clic para el INICIO del vehículo 1.")
            self._dibujar_estatico()
            return

        self.modo_manual = False
        if not self._aleatorias(n, length, width):
            from tkinter import messagebox
            messagebox.showwarning("Sin espacio",
                "No se hallaron posiciones libres. Reduce vehículos/tamaño/obstáculos.")
            return
        self._crear_vehiculos(length, width, vmax)
        self.estado.set(f"{n} vehículos colocados al azar. Pulsa «Calcular y simular».")
        self._dibujar_estatico()

    def _aleatorias(self, n, length, width):
        self.inicios, self.metas = [], []
        sep = length * 1.6
        for _ in range(n):
            ini = self._muestrear(length, width, [p[:2] for p in self.inicios], sep)
            met = self._muestrear(length, width, self.metas, sep)
            if ini is None or met is None:
                return False
            ix, iy, ith = ini
            th_meta = math.atan2(met[1] - iy, met[0] - ix)
            if self.env.libre(ix, iy, th_meta, length, width, margen=0.3):
                ith = th_meta
            self.inicios.append((ix, iy, ith))
            self.metas.append((met[0], met[1]))
        return True

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

    def _reubicar(self, veh):
        sep = veh.length * 1.6
        otros_ini = [self.inicios[k][:2] for k in range(len(self.inicios)) if k != veh.idx]
        otros_meta = [self.metas[k] for k in range(len(self.metas)) if k != veh.idx]
        ini = self._muestrear(veh.length, veh.width, otros_ini, sep)
        met = self._muestrear(veh.length, veh.width, otros_meta, sep)
        if ini is None or met is None:
            return False
        ix, iy, ith = ini
        thm = math.atan2(met[1] - iy, met[0] - ix)
        if self.env.libre(ix, iy, thm, veh.length, veh.width, margen=0.3):
            ith = thm
        veh.inicio = (ix, iy, ith)
        veh.meta = (met[0], met[1])
        self.inicios[veh.idx] = veh.inicio
        self.metas[veh.idx] = veh.meta
        return True

    def _crear_vehiculos(self, length, width, vmax, n=None):
        if n is None:
            n = len(self.inicios)
        self.vehiculos = [
            Vehiculo(i, self.inicios[i], self.metas[i], length, width, vmax)
            for i in range(n)]

    def click_mapa(self, ev):
        if not self.modo_manual:
            return
        n, length, width, vmax = self._cfg
        x, y = ev.x / SCALE, ev.y / SCALE

        if self.colocando_inicio:
            if not self.env.libre(x, y, 0.0, length, width, margen=0.2):
                self.estado.set("Inicio inválido (fuera o sobre obstáculo). Otro punto.")
                return
            self.pend_inicio = (x, y)
            self.colocando_inicio = False
            self.estado.set(f"MANUAL · Clic para el DESTINO del vehículo {len(self.metas) + 1}.")
            self._marca(x, y, PALETA[len(self.metas) % len(PALETA)])
        else:
            if not self.env.libre(x, y, 0.0, length, width, margen=0.2):
                self.estado.set("Destino inválido. Elige otro punto.")
                return
            ix, iy = self.pend_inicio
            self.inicios.append((ix, iy, math.atan2(y - iy, x - ix)))
            self.metas.append((x, y))
            self.colocando_inicio = True
            if len(self.metas) >= n:
                self.modo_manual = False
                self._crear_vehiculos(length, width, vmax)
                self.estado.set(f"{n} vehículos colocados. Pulsa «Calcular y simular».")
                self._dibujar_estatico()
            else:
                self.estado.set(f"MANUAL · Clic para el INICIO del vehículo {len(self.metas) + 1}.")
                self._dibujar_estatico()

    def _marca(self, x, y, col):
        r = 4
        self.canvas.create_oval(x * SCALE - r, y * SCALE - r,
                                x * SCALE + r, y * SCALE + r, fill=col, outline="")

    def calcular_y_simular(self):
        p = self._params()
        if not p:
            return
        n, length, width, vmax, _ = p
        if len(self.inicios) < n or len(self.metas) < n:
            from tkinter import messagebox
            messagebox.showinfo("Faltan posiciones",
                                "Genera o coloca primero todas las posiciones.")
            return

        self._detener()
        self._crear_vehiculos(length, width, vmax, n)
        self._ocupado = True
        try:
            if not self._warmed:
                self.estado.set("Compilando núcleos nativos (solo la 1ª vez)…")
                self.root.update()
                warmup()
                self._warmed = True
            self._planificar_todo(n)
        finally:
            self._ocupado = False

    def _planificar_todo(self, n):
        from tkinter import messagebox
        self.estado.set("Planificando de forma cooperativa (Hybrid A* espacio-tiempo)…")
        self.root.update()
        if not self.root.winfo_exists():
            return
        aleatorio = self.modo.get() == "aleatorio"

        CAP_COMPLETO = 6_000_000
        reservas = Reservas()
        planificados = []
        sin_ruta = 0
        inconcluso = 0
        self.planificador.tick = self._tick_plan
        self.planificador.deadline = None
        cap_prev = self.planificador.max_exp
        self.planificador.max_exp = CAP_COMPLETO
        try:
            for i, veh in enumerate(self.vehiculos):
                self._plan_msg = f"Planificando vehículo {i + 1}/{n}…"
                self.estado.set(self._plan_msg)
                self.root.update()
                if not self.root.winfo_exists():
                    return
                motivo, traj = self._planificar_veh(veh, reservas, i, aleatorio)
                if motivo != "ok":
                    veh.traj = []
                    if motivo == "sin_ruta":
                        sin_ruta += 1
                    else:
                        inconcluso += 1
                else:
                    veh.traj = traj
                    veh.dt_plan = DT
                    reservas.add(traj, veh.length, veh.width)
                    planificados.append(veh)
                if not self.root.winfo_exists():
                    return
        finally:
            self.planificador.tick = None
            self.planificador.deadline = None
            self.planificador.max_exp = cap_prev

        self.vehiculos = planificados
        self._dibujar_estatico()
        if not planificados:
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
        self.estado.set(f"{len(planificados)} rutas sin colisiones "
                        f"({len(self.frames)} fotogramas).{aviso}  Reproduciendo…")
        self.reproducir()

    def _planificar_veh(self, veh, reservas, i, aleatorio):
        """Planifica UN vehículo hasta una conclusión real. Devuelve (motivo, traj):
        'ok' con ruta; 'sin_ruta' (no existe, definitivo); 'limite' (techo de
        expansiones, inconcluso). En modo aleatorio, ante un 'sin_ruta' prueba a
        reubicar los extremos."""
        self.planificador.bloqueos = self._bloqueos_metas(excepto=i)
        traj = self.planificador.planificar(veh, reservas)
        motivo = self.planificador.motivo
        intentos = 0
        while (motivo == "sin_ruta" and aleatorio and intentos < 12):
            if not self._reubicar(veh):
                break
            self.planificador.bloqueos = self._bloqueos_metas(excepto=i)
            traj = self.planificador.planificar(veh, reservas)
            motivo = self.planificador.motivo
            intentos += 1
        if traj is not None and len(traj) >= 2:
            return "ok", traj
        if traj:
            return "ok", [traj[0], traj[0]]
        return (motivo if motivo != "ok" else "sin_ruta"), None

    def _tick_plan(self, expand):
        self.estado.set(f"{self._plan_msg}  ({expand:,} nodos explorados)")
        self.root.update()

    def _bloqueos_metas(self, excepto):
        """OBB cuadrados en las metas ajenas, para que nadie planifique aparcar
        donde otro vehículo debe aparcar."""
        bloq = []
        for j, veh in enumerate(self.vehiculos):
            if j == excepto:
                continue
            mx, my = veh.meta
            poly = obb_corners(mx, my, 0.0, veh.diag, veh.diag)
            bloq.append((poly, (mx, my, veh.diag / 2)))
        return bloq

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
            self.anim_id = self.root.after(int(self.vel.get()), self._anim)
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
        for i, m in enumerate(self.metas):
            col = PALETA[i % len(PALETA)]
            c.create_oval(m[0] * SCALE - 7, m[1] * SCALE - 7,
                          m[0] * SCALE + 7, m[1] * SCALE + 7, outline=col, width=2)
            c.create_text(m[0] * SCALE, m[1] * SCALE, text="◎", fill=col,
                          font=("", 13, "bold"))
        for i, ini in enumerate(self.inicios):
            col = PALETA[i % len(PALETA)]
            poly = obb_corners(ini[0], ini[1], ini[2], VEH_LEN, VEH_WID)
            c.create_polygon(self._poly_px(poly), outline=col, fill="",
                             width=1, dash=(3, 3))

    def _rutas(self):
        for veh in self.vehiculos:
            if veh.traj:
                col = PALETA[veh.idx % len(PALETA)]
                pts = []
                for x, y, _ in veh.traj:
                    pts.extend((x * SCALE, y * SCALE))
                self.canvas.create_line(pts, fill=col, width=1, smooth=True)

    def _dibujar_estatico(self):
        self._fondo()
        for veh in self.vehiculos:
            self._dibujar_coche(veh.inicio[0], veh.inicio[1], veh.inicio[2],
                                veh.length, veh.width,
                                PALETA[veh.idx % len(PALETA)], veh.idx + 1)

    def _dibujar_frame(self):
        self._fondo()
        self._rutas()
        fr = self.frames[self.frame]
        for veh in self.vehiculos:
            x, y, th = fr[veh.idx]
            self._dibujar_coche(x, y, th, veh.length, veh.width,
                                PALETA[veh.idx % len(PALETA)], veh.idx + 1)
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
