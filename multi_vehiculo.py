#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulador físico de enrutamiento multi-vehículo en ESPACIO CONTINUO.

Enfoque (sin cuadrículas ni movimientos discretos):

  1. Espacio continuo: las posiciones son coordenadas reales (x, y) en metros.

  2. Cinemática realista (modelo de bicicleta):
         x'  = v · cos(θ)
         y'  = v · sin(θ)
         θ'  = (v / L) · tan(δ)        L = batalla (wheelbase)
         v'  = a                       a = aceleración (acotada)

  3. Giro no holonómico: el ángulo de dirección δ está acotado (|δ| ≤ δ_max),
     lo que impone un radio de giro mínimo  R_min = L / tan(δ_max).  Si v = 0
     entonces θ' = 0  →  el coche NO puede girar sobre su eje en estático.

  4. Colisiones reales: los vehículos son rectángulos ORIENTADOS (OBB) y la
     detección usa el Teorema de los Ejes Separadores (SAT), tanto contra los
     obstáculos fijos como entre los propios vehículos (dinámicos).

Arquitectura:

  · PLANIFICACIÓN — Hybrid A* sobre el estado continuo (x, y, θ) con primitivas
    de arco del modelo de bicicleta (avance y marcha atrás). Guarda la pose de
    cada subpaso, de modo que la ruta resultante ES una trayectoria de bicicleta
    cinemáticamente factible (respeta el radio de giro). Penaliza giro, reversa y
    cambios de sentido (cúspides), y usa expansión analítica para converger.

  · EJECUCIÓN — todos los vehículos avanzan a la vez SOBRE su trayectoria,
    parametrizada por longitud de arco. Lo único que se controla es la VELOCIDAD
    (con aceleración acotada): cada vehículo acelera, frena ante su meta y ante
    las cúspides, y cede el paso (frena) si su avance inminente chocaría —por
    SAT— con un vehículo prioritario. Quien llega a su meta se detiene y queda
    como obstáculo para los demás. Así no hay colisiones y la llegada es fiable.

Requiere solo la librería estándar (tkinter). Ejecutar con un Python con Tk:
    /Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12 multi_vehiculo.py
"""

import math
import time
import heapq
import random
import tkinter as tk
from tkinter import ttk, messagebox

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

# Dimensiones y velocidad de los vehículos (fijas; ya no editables en el panel)
VEH_LEN = 1.3       # largo (m)
VEH_WID = 0.7       # ancho (m)
VEH_VMAX = 2.5      # velocidad máxima (m/s)


# --------------------------------------------------------------------------- #
# Geometría: rectángulos orientados (OBB) y SAT
# --------------------------------------------------------------------------- #
def obb_corners(x, y, theta, length, width):
    """Esquinas (en orden) de un rectángulo centrado en (x,y) orientado θ."""
    hl, hw = length / 2.0, width / 2.0
    c, s = math.cos(theta), math.sin(theta)
    pts = []
    for lx, ly in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw)):
        pts.append((x + lx * c - ly * s, y + lx * s + ly * c))
    return pts


def _axes(poly):
    """Normales a las aristas del polígono (ejes candidatos del SAT)."""
    ejes = []
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        ex, ey = x2 - x1, y2 - y1
        nx, ny = -ey, ex
        d = math.hypot(nx, ny)
        if d > 1e-12:
            ejes.append((nx / d, ny / d))
    return ejes


def _proyecta(poly, eje):
    ax, ay = eje
    ds = [px * ax + py * ay for px, py in poly]
    return min(ds), max(ds)


def sat_colision(p1, p2):
    """True si dos polígonos convexos se solapan (Separating Axis Theorem)."""
    for eje in _axes(p1) + _axes(p2):
        a0, a1 = _proyecta(p1, eje)
        b0, b1 = _proyecta(p2, eje)
        if a1 < b0 or b1 < a0:      # existe eje separador → no colisionan
            return False
    return True


def _radio(poly, cx, cy):
    return max(math.hypot(px - cx, py - cy) for px, py in poly)


# --------------------------------------------------------------------------- #
# Obstáculos (polígonos convexos) y entorno
# --------------------------------------------------------------------------- #
def rect_poly(x, y, w, h):
    return obb_corners(x + w / 2.0, y + h / 2.0, 0.0, w, h)


class Entorno:
    def __init__(self):
        self.obstaculos = []        # lista de polígonos
        self.obst_bb = []           # (cx, cy, radio) para descarte rápido

    def generar(self, densidad=0.0):
        """Genera (aleatoriamente) un mapa tipo CIUDAD: manzanas rectangulares
        separadas por calles de anchura variable —algunas estrechas—, con un
        anillo perimetral libre. 'densidad' añade tabiques/obstáculos extra que
        complican aún más el trazado. Cada llamada produce un mapa distinto."""
        borde = 2.0                              # anillo perimetral libre
        # anchuras de calle posibles (incluye estrechas)
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
                if random.random() < 0.82:       # algunos huecos = plazas
                    # manzana, a veces partida para crear callejones internos
                    if bw > 5.0 and random.random() < 0.4:
                        hueco = random.uniform(1.0, bw - 3.5)
                        obs.append(rect_poly(x, y, hueco, bh))
                        obs.append(rect_poly(x + hueco + 2.3, y,
                                             bw - hueco - 2.3, bh))
                    else:
                        obs.append(rect_poly(x, y, bw, bh))
                y += bh + random.choice(calles)
            x += bw + random.choice(calles)

        # un par de obstáculos rotados (evidencian el SAT continuo)
        for _ in range(2):
            obs.append(obb_corners(random.uniform(borde + 3, W - borde - 3),
                                   random.uniform(borde + 3, H - borde - 3),
                                   random.uniform(0, math.pi),
                                   random.uniform(2.0, 4.0), 1.6))

        # tabiques extra según densidad (estrechan más las calles)
        for _ in range(int(densidad * 25)):
            obs.append(obb_corners(random.uniform(borde, W - borde),
                                   random.uniform(borde, H - borde),
                                   random.uniform(0, math.pi),
                                   random.uniform(1.5, 3.0),
                                   random.uniform(1.0, 1.8)))

        self.obstaculos = obs
        self._precalcular()

    def _precalcular(self):
        self.obst_bb = []
        for poly in self.obstaculos:
            cx = sum(p[0] for p in poly) / len(poly)
            cy = sum(p[1] for p in poly) / len(poly)
            self.obst_bb.append((cx, cy, _radio(poly, cx, cy)))

    def libre(self, x, y, theta, length, width, margen=0.0):
        """¿Cabe el vehículo (OBB) sin tocar bordes ni obstáculos?"""
        L = length + 2 * margen
        Wd = width + 2 * margen
        corners = obb_corners(x, y, theta, L, Wd)
        for px, py in corners:                       # límites del mundo
            if px < 0 or px > W or py < 0 or py > H:
                return False
        veh_r = math.hypot(L, Wd) / 2.0
        for (ocx, ocy, orad), poly in zip(self.obst_bb, self.obstaculos):
            if math.hypot(ocx - x, ocy - y) > orad + veh_r:
                continue
            if sat_colision(corners, poly):
                return False
        return True



# --------------------------------------------------------------------------- #
# Vehículo  (trayectoria temporal: una pose por paso de planificación)
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
        self.a_max = 1.0              # aceleración/frenado máximos (m/s²): baja
        # traj: una pose por PASO FINO (de DT segundos), ya con la velocidad
        # incorporada por el planificador (el espaciado entre poses ES v·DT).
        self.traj = []                # [(x, y, theta), ...]
        self.dt_plan = DT             # duración de cada paso de la trayectoria

    @property
    def radio_giro_min(self):
        return self.wheelbase / math.tan(self.delta_max)

    @property
    def diag(self):
        return math.hypot(self.length, self.width)

    def pose_en_tiempo(self, t):
        """Pose en el instante t (s). Como cada paso de 'traj' dura DT, basta
        indexar. Tras el final, se queda en su meta (vehículo aparcado)."""
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
# Reservas espacio-tiempo: trayectorias de los vehículos ya planificados
# --------------------------------------------------------------------------- #
class Reservas:
    """Guarda las trayectorias temporales ya comprometidas. Permite consultar
    si un OBB choca (SAT) con algún vehículo reservado en un paso dado. Para
    pasos posteriores al final de una trayectoria, el vehículo sigue aparcado
    en su meta (su última pose)."""

    def __init__(self):
        self.items = []               # (traj, length, width)

    def add(self, traj, length, width):
        self.items.append((traj, length, width))

    def choca(self, corners, k, vx, vy, vr, margen):
        for traj, length, width in self.items:
            ox, oy, oth = traj[k] if k < len(traj) else traj[-1]
            if math.hypot(vx - ox, vy - oy) > vr + math.hypot(length, width) / 2 + margen:
                continue
            oc = obb_corners(ox, oy, oth, length + 2 * margen, width + 2 * margen)
            if sat_colision(corners, oc):
                return True
        return False


# --------------------------------------------------------------------------- #
# Hybrid A* COOPERATIVO en espacio-tiempo (x, y, θ, t)
# --------------------------------------------------------------------------- #
#
# Cada vehículo se planifica respetando: (a) los obstáculos fijos, (b) las metas
# ocupadas por otros (bloqueos), y (c) las TRAYECTORIAS de los vehículos de mayor
# prioridad como obstáculos MÓVILES en el tiempo. Las acciones incluyen avanzar a
# distintas velocidades, ESPERAR y dar marcha atrás: así la coordinación (ceder
# el paso, rodear, temporizar) queda integrada en el propio plan, sin frenazos
# reactivos. La ejecución se limita a reproducir las trayectorias resultantes.
# --------------------------------------------------------------------------- #
class Planificador:
    def __init__(self, entorno):
        self.env = entorno
        self.res_pos = 0.7
        # Heurístico holonómico CON obstáculos (Dijkstra en rejilla): guía la
        # búsqueda rodeando los obstáculos desde el inicio → rutas más directas.
        self.h_res = 0.5             # tamaño de celda de la rejilla del heurístico
        self._occ_sig = None         # firma de la rejilla de ocupación cacheada
        self.res_v = 0.5             # discretización de la velocidad (m/s)
        self.dt = 0.4                 # duración de cada acción macro (s)
        self.subpasos = 4             # → TAU = dt/subpasos = 0.1 s = DT
        self.goal_tol = 1.6
        self.v_tol = 0.45             # velocidad para considerar "detenido"
        self.k_max = 1600             # horizonte temporal (pasos finos)
        # Conexión analítica (pure-pursuit) para rematar la llegada. Solo se
        # dispara si el vehículo ya está BIEN ALINEADO con la meta (ang_tiro) o
        # muy cerca; si llega mal orientado, la búsqueda sigue girando para
        # alinearse y luego remata recto (evita el arco único y AMPLIO hacia la
        # meta, que no es la ruta más rápida).
        self.dist_tiro = 10.0
        self.ang_tiro = math.radians(30)
        self.margen = 0.10            # holgura contra obstáculos fijos
        self.margen_din = 0.15        # holgura contra otros vehículos (pasos ceñidos)
        self.bloqueos = []            # metas ajenas (estáticas) [(poly, bb)]
        # se fijan por vehículo en planificar():
        self.v_max_c = 5.0            # velocidad máxima (nunca superada)
        self.a_max = 1.0              # aceleración/frenado máximos (acotados)
        self.v_rev = 2.5              # velocidad máxima de marcha atrás
        # calidad de la búsqueda (nº de primitivas de giro, resolución angular,
        # heurístico y tope de expansiones). La fija configurar_calidad().
        self.res_ang = math.radians(10)
        self.max_exp = 600000
        self.peso_h = 1.6
        self.dir_fracs = []
        self.configurar_calidad(3)
        # límites de RESPUESTA (para que nunca se cuelgue con muchos vehículos):
        self.deadline = None         # perf_counter límite; None = sin límite
        self.tick = None             # callback(expand) periódico (refresca UI)

    def configurar_calidad(self, nivel):
        """Ajusta el equilibrio TIEMPO ↔ CALIDAD de ruta. A mayor nivel: más
        ángulos de giro posibles (más densos cerca de 0 para corregir el rumbo
        con finura), resolución angular más fina, heurístico menos goloso (rutas
        más cortas) y más expansiones permitidas. Cuesta más tiempo de cálculo."""
        # Nota: 'peso_h' se mantiene ALTO (búsqueda golosa = rápida) en todos los
        # niveles. Bajarlo no acorta las rutas (comprobado) y sí ralentiza mucho la
        # búsqueda, llegando a FALLAR en casos difíciles. La calidad sube por más
        # primitivas de giro y resolución angular más fina, no por menos golosidad.
        tabla = {
            1: (9,  14, 2.3, 250000),   # rápido
            2: (13, 13, 2.2, 400000),
            3: (17, 12, 2.1, 600000),   # predeterminado (alto, pero ágil)
            4: (27, 10, 2.0, 1000000),
            5: (41,  8, 1.9, 1700000),  # máxima calidad (lento)
        }
        n_dir, ang, peso, mx = tabla.get(int(nivel), tabla[3])
        half = n_dir // 2
        # fracciones de dmax simétricas, con espaciado más fino cerca de 0
        fracs = [0.0]
        for i in range(1, half + 1):
            f = (i / half) ** 1.3
            fracs += [f, -f]
        self.dir_fracs = sorted(fracs)
        self.res_ang = math.radians(ang)
        self.peso_h = peso
        self.max_exp = mx

    # ---- utilidades de ocupación estática (obstáculos + metas ajenas) ---- #
    def _clave(self, x, y, th, v, k):
        return (int(x / self.res_pos), int(y / self.res_pos),
                int((th % (2 * math.pi)) / self.res_ang),
                int(round(v / self.res_v)), k)

    def _libre(self, x, y, th):
        if not self.env.libre(x, y, th, self._len, self._wid, self.margen):
            return False
        if self.bloqueos:
            corners = obb_corners(x, y, th, self._len + 2 * self.margen,
                                  self._wid + 2 * self.margen)
            vr = self.diag / 2 + self.margen
            for poly, (cx, cy, rr) in self.bloqueos:
                if math.hypot(cx - x, cy - y) > rr + vr:
                    continue
                if sat_colision(corners, poly):
                    return False
        return True

    def _mover(self, x, y, th, v, accel, delta, L):
        """Integra una acción macro (modelo de bicicleta) partiendo de la
        velocidad 'v' (con signo) y aplicando la aceleración 'accel' (acotada)
        y la dirección 'delta'. La velocidad evoluciona en cada SUBPASO sin
        superar nunca v_max (avance) ni v_rev (retroceso). Devuelve
        (lista de poses por subpaso a TAU=DT, velocidad final) o (None, None)
        si algún subpaso choca con un obstáculo fijo."""
        h = self.dt / self.subpasos
        subs = []
        for _ in range(self.subpasos):
            v = min(self.v_max_c, max(-self.v_rev, v + accel * h))
            th = th + (1.0 / L) * math.tan(delta) * (v * h)
            x = x + math.cos(th) * v * h
            y = y + math.sin(th) * v * h
            if not self._libre(x, y, th):
                return None, None
            subs.append((x, y, th))
        return subs, v

    def _choca_din(self, x, y, th, k, reservas):
        corners = obb_corners(x, y, th, self._len + 2 * self.margen_din,
                              self._wid + 2 * self.margen_din)
        return reservas.choca(corners, k, x, y, self.diag / 2 + self.margen_din,
                              0.0)

    def _subs_libres_din(self, subs, k0, reservas):
        """¿Todos los subpasos están libres de los vehículos reservados (en su
        instante fino correspondiente)?"""
        for off, (px, py, pth) in enumerate(subs, start=1):
            if self._choca_din(px, py, pth, k0 + off, reservas):
                return False
        return True

    def _aparcamiento_libre(self, pose, k0, reservas):
        """Al llegar a la meta el vehículo se queda APARCADO ahí para siempre.
        Hay que garantizar que esa plaza no la pise NINGÚN vehículo reservado en
        NINGÚN instante futuro (incluidos los de mayor prioridad, ya planificados,
        que podrían pasar por ahí más tarde). Comprueba la pose final contra todas
        las reservas desde k0 hasta que todas están paradas."""
        x, y, th = pose
        kfin = k0
        for traj, _, _ in reservas.items:
            if len(traj) > kfin:
                kfin = len(traj)
        for k in range(k0, kfin + 1):
            if self._choca_din(x, y, th, k, reservas):
                return False
        return True

    def _tiro_directo(self, x, y, th, v, k, gx, gy, L, dmax, reservas):
        """Conexión analítica hacia la meta (pure pursuit) con FRENADO: regula
        la velocidad para acercarse a v_max en tramo libre y decelerar —con
        aceleración acotada— hasta detenerse justo en la meta. Comprueba la
        estática y las reservas dinámicas en CADA paso fino. Devuelve la lista
        densa de poses (una por DT) o None."""
        h = self.dt / self.subpasos      # = DT
        poses = []
        kk = k
        d0 = math.hypot(gx - x, gy - y)  # distancia inicial a la meta
        largo = 0.0                      # longitud recorrida por el remate
        # Tope anti-BUCLE: un remate directo recorre ~d0 (con una curva suave si
        # venía algo desviado). Si el pure-pursuit se enrosca para reencarar la
        # meta (porque apunta hacia otro lado), su longitud se dispara muy por
        # encima de d0 → lo descartamos y la búsqueda sigue hasta encarar bien,
        # en vez de dejar ese rizo circular feo justo antes de aparcar.
        largo_max = 1.5 * d0 + 2.0 * self.goal_tol
        for _ in range(600):
            d = math.hypot(gx - x, gy - y)
            if d <= self.goal_tol and abs(v) <= self.v_tol:
                return poses
            # velocidad deseada: la máxima que aún permite frenar a tiempo
            v_des = min(self.v_max_c, math.sqrt(2.0 * self.a_max * max(d, 0.0)))
            dv = max(-self.a_max * h, min(self.a_max * h, v_des - v))
            v = min(self.v_max_c, max(0.0, v + dv))     # aproxima siempre de frente
            alpha = math.atan2(gy - y, gx - x) - th
            alpha = math.atan2(math.sin(alpha), math.cos(alpha))
            delta = max(-dmax, min(dmax, math.atan2(2.0 * L * math.sin(alpha), max(d, 1e-3))))
            th = th + (1.0 / L) * math.tan(delta) * (v * h)
            x = x + math.cos(th) * v * h
            y = y + math.sin(th) * v * h
            largo += abs(v) * h
            if largo > largo_max:                        # se está enroscando
                return None
            if not self._libre(x, y, th):
                return None
            if self._choca_din(x, y, th, kk + 1, reservas):
                return None
            poses.append((x, y, th))
            kk += 1
            if v < 1e-3 and d > self.goal_tol:           # se paró sin llegar
                return None
        return None

    # ------------------- heurístico con obstáculos ------------------------ #
    def _asegurar_ocupacion(self):
        """Rejilla de ocupación (celda libre / bloqueada) del mapa, INFLADA por
        el semiancho del vehículo para que su CENTRO no pegue con los muros.
        Depende solo de los obstáculos (fijos) y del tamaño del vehículo, así que
        se calcula una vez por mapa y se reutiliza para todos los vehículos."""
        infl = 0.5 * self._wid + self.margen
        sig = (id(self.env.obstaculos), round(infl, 3), self.h_res)
        if self._occ_sig == sig:
            return
        res = self.h_res
        nx = int(W / res) + 1
        ny = int(H / res) + 1
        occ = [[self.env.libre(ix * res, iy * res, 0.0, 0.0, 0.0, infl)
                for iy in range(ny)] for ix in range(nx)]
        self._occ = occ
        self._occ_nx, self._occ_ny = nx, ny
        self._occ_sig = sig

    def _construir_heuristica(self, gx, gy):
        """Campo de distancias desde la meta por Dijkstra 8-conexo sobre la
        rejilla libre. Es la longitud del camino MÁS CORTO que rodea obstáculos
        (holonómico); como el coche real (no holonómico) no puede ser más corto,
        orienta bien la búsqueda hacia rutas directas. Cachea (gx, gy) para el
        respaldo euclídeo cuando una pose cae fuera de la rejilla o en celda
        bloqueada."""
        self._asegurar_ocupacion()
        res = self.h_res
        nx, ny, occ = self._occ_nx, self._occ_ny, self._occ
        INF = float("inf")
        dist = [[INF] * ny for _ in range(nx)]
        gi = min(nx - 1, max(0, int(round(gx / res))))
        gj = min(ny - 1, max(0, int(round(gy / res))))
        if not occ[gi][gj]:                       # meta en celda inflada-bloqueada:
            bd = INF                              # usa la celda libre más cercana
            for ix in range(nx):
                for iy in range(ny):
                    if occ[ix][iy]:
                        dd = (ix * res - gx) ** 2 + (iy * res - gy) ** 2
                        if dd < bd:
                            bd, gi, gj = dd, ix, iy
        vecinos = ((1, 0, res), (-1, 0, res), (0, 1, res), (0, -1, res),
                   (1, 1, res * 1.41421356), (1, -1, res * 1.41421356),
                   (-1, 1, res * 1.41421356), (-1, -1, res * 1.41421356))
        dist[gi][gj] = 0.0
        pq = [(0.0, gi, gj)]
        while pq:
            d, ix, iy = heapq.heappop(pq)
            if d > dist[ix][iy]:
                continue
            for dx, dy, c in vecinos:
                jx, jy = ix + dx, iy + dy
                if 0 <= jx < nx and 0 <= jy < ny and occ[jx][jy]:
                    nd = d + c
                    if nd < dist[jx][jy]:
                        dist[jx][jy] = nd
                        heapq.heappush(pq, (nd, jx, jy))
        self._hdist = dist
        self._h_gx, self._h_gy = gx, gy

    def _h_time(self, x, y):
        """Tiempo estimado a la meta = distancia (rejilla con obstáculos) / v_max.
        Respaldo euclídeo si la pose queda fuera de la rejilla o en celda sin
        distancia calculada (aislada)."""
        res = self.h_res
        ix = int(round(x / res))
        iy = int(round(y / res))
        if 0 <= ix < self._occ_nx and 0 <= iy < self._occ_ny:
            d = self._hdist[ix][iy]
            if d < float("inf"):
                return d / self.v_max_c
        return math.hypot(x - self._h_gx, y - self._h_gy) / self.v_max_c

    def planificar(self, veh, reservas):
        """Devuelve la trayectoria [(x,y,th), ...] (una pose por PASO FINO de DT,
        ya con la velocidad incorporada) o None si no halla solución coordinada.

        El estado incluye la VELOCIDAD (x, y, θ, v, k) y las acciones eligen la
        ACELERACIÓN (acotada) además de la dirección. Así el planificador puede
        acelerar o frenar EN CUALQUIER MOMENTO —p. ej. decelerar antes de un giro
        y volver a acelerar al salir— si eso da una ruta mejor. Parte del reposo
        (v=0) y la conexión analítica termina frenando hasta detenerse."""
        self._len, self._wid = veh.length, veh.width
        self.diag = veh.diag
        L = veh.wheelbase
        dmax = veh.delta_max
        sx, sy, sth = veh.inicio
        gx, gy = veh.meta
        self.v_max_c = veh.v_max
        self.a_max = veh.a_max
        self.v_rev = 0.5 * veh.v_max

        # Heurístico holonómico CON obstáculos (rejilla Dijkstra desde la meta):
        # guía la búsqueda rodeando los obstáculos desde el principio → rutas más
        # directas (evita el "ir recto y descubrir el obstáculo tarde").
        self._construir_heuristica(gx, gy)

        # repertorio de acciones: (aceleración, dirección). La aceleración está
        # acotada a ±a_max (o 0 = mantener velocidad); la dirección, a ±dmax.
        # Direcciones FINAS cerca de 0 para poder corregir el rumbo suavemente
        # desde el principio (evita el "ir recto y girar tarde").
        a = self.a_max
        deltas = [f * dmax for f in self.dir_fracs]
        acciones = [(av, d) for av in (a, 0.0, -a) for d in deltas]

        # El grafo de reconstrucción se indexa por ID ÚNICO de nodo (no por la
        # clave discretizada). Así cada tramo arranca EXACTAMENTE donde acaba el
        # de su padre → trayectoria continua, sin saltos de bucket ("teletransporte").
        # La clave discretizada se usa solo para PODAR estados dominados.
        clave0 = self._clave(sx, sy, sth, 0.0, 0)
        nid = 0
        padre_id = {}                     # nid -> nid del padre
        arista_id = {}                    # nid -> poses densas del tramo hacia nid
        delta_prev = {0: 0.0}             # nid -> dirección con la que se llegó
        h0 = self._h_time(sx, sy)
        # cola: (f, nid, x, y, theta, v, k_fino, g)
        abierto = [(self.peso_h * h0, 0, sx, sy, sth, 0.0, 0, 0.0)]
        mejor_g = {clave0: 0.0}
        expand = 0
        ns = self.subpasos
        self._last_tick = time.perf_counter()
        # motivo de parada; se afina al terminar. "limite" es el valor pesimista
        # por defecto (si salimos por techo de expansiones o plazo opcional).
        self.motivo = "limite"

        while abierto and expand < self.max_exp:
            _f, cid, x, y, th, v, k, g = heapq.heappop(abierto)
            expand += 1

            # Con frecuencia: respeta el PLAZO OPCIONAL (por defecto None = sin
            # límite de reloj) y refresca la UI —limitado por tiempo real— para
            # que la ventana no se congele durante una búsqueda larga.
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
                # solo acepta si la plaza de aparcamiento queda libre para siempre
                if self._aparcamiento_libre((x, y, th), k, reservas):
                    self.motivo = "ok"
                    return self._reconstruir(padre_id, arista_id, cid, (sx, sy, sth))
                # si no, sigue buscando otra pose de llegada (dentro de goal_tol)

            elif d_goal <= self.dist_tiro:
                # remata solo si ya apunta bien a la meta (o está muy cerca);
                # así no genera un arco amplio desde una orientación mala.
                alpha0 = math.atan2(gy - y, gx - x) - th
                alpha0 = math.atan2(math.sin(alpha0), math.cos(alpha0))
                if abs(alpha0) <= self.ang_tiro or d_goal <= self.goal_tol * 1.5:
                    tiro = self._tiro_directo(x, y, th, v, k, gx, gy, L, dmax, reservas)
                    if tiro is not None:
                        kfin = k + len(tiro)
                        if self._aparcamiento_libre(tiro[-1], kfin, reservas):
                            base = self._reconstruir(padre_id, arista_id, cid,
                                                     (sx, sy, sth))
                            self.motivo = "ok"
                            return base + tiro

            if k >= self.k_max:
                continue

            dprev = delta_prev.get(cid, 0.0)
            for accel, delta in acciones:
                subs, nv = self._mover(x, y, th, v, accel, delta, L)
                if subs is None:
                    continue
                if not self._subs_libres_din(subs, k, reservas):
                    continue
                nx, ny, nth = subs[-1]
                nk = k + ns
                # COSTE = TIEMPO. Ese es el objetivo real: una ruta más larga o
                # con más rodeos acumula más pasos (más 'dt') y ya sale peor por sí
                # sola. No se penaliza girar (un arco para bordear un obstáculo es
                # tan válido como ir recto; lo que decide es el tiempo total).
                ng = g + self.dt
                if nv < 0:
                    ng += 0.6 * self.dt               # marcha atrás: maniobra lenta/indeseada
                if abs(nv) < 1e-3 and accel <= 0.0:
                    ng += 0.20 * self.dt              # pararse sin motivo cuesta tiempo
                # ε de CURVATURA (subordinado al tiempo): a igualdad de eficiencia,
                # prefiere ir RECTO. Sin esto, una curva suave y una recta que
                # avanzan lo mismo cuestan igual y la búsqueda trazaba curvas
                # gratuitas donde bastaba la línea recta hasta el siguiente giro.
                ng += 0.10 * self.dt * abs(delta) / dmax
                # Desempate MENOR: evita el temblor de la rejilla discreta
                # prefiriendo no cambiar bruscamente de dirección.
                ng += 0.08 * self.dt * abs(delta - dprev)
                key = self._clave(nx, ny, nth, nv, nk)
                if ng < mejor_g.get(key, float("inf")):
                    mejor_g[key] = ng
                    nid += 1
                    padre_id[nid] = cid
                    arista_id[nid] = subs
                    delta_prev[nid] = delta
                    h = self._h_time(nx, ny)
                    heapq.heappush(abierto,
                                   (ng + self.peso_h * h, nid, nx, ny, nth, nv, nk, ng))
        # Salida del bucle: si la frontera quedó VACÍA, la búsqueda agotó todo el
        # espacio alcanzable sin llegar a la meta → NO EXISTE ruta (respuesta
        # definitiva). Si aún quedaban nodos, paró por el techo de expansiones →
        # resultado INCONCLUSO (la búsqueda era demasiado grande para agotarla).
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
            traj.extend(t)
        return traj


# --------------------------------------------------------------------------- #
# Ejecución: reproduce las trayectorias coordinadas (ya libres de colisión)
# --------------------------------------------------------------------------- #
def construir_frames(vehiculos):
    """Muestrea todas las trayectorias temporales a intervalos DT y produce los
    fotogramas {idx: (x,y,theta)} para la reproducción fluida."""
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
        self.root = root
        root.title("Simulador físico multi-vehículo · espacio continuo + SAT")
        root.resizable(False, False)

        self.env = Entorno()
        self.planificador = Planificador(self.env)
        self.vehiculos = []
        self.inicios = []          # [(x,y,theta), ...]
        self.metas = []            # [(x,y), ...]
        self.frames = []
        self.frame = 0
        self.anim_id = None
        self.reproduciendo = False

        self.modo_manual = False
        self.colocando_inicio = True
        self.pend_inicio = None
        self._ocupado = False      # evita reentrancia durante la planificación
        self._plan_msg = ""        # texto de progreso durante la planificación

        self._construir_ui()
        self.env.generar(0.0)
        self._dibujar_estatico()

    # --------------------------- blindaje --------------------------------- #
    def _seguro(self, fn):
        """Envuelve un callback de la UI para que NINGUNA combinación de
        pulsaciones pueda colgar o romper la aplicación:
          · ignora la acción si hay una planificación en curso (reentrancia),
          · absorbe los errores de Tk cuando la ventana ya se cerró,
          · y muestra cualquier excepción inesperada en un diálogo en vez de
            propagarla (que abortaría el programa)."""
        def envuelto(*args, **kwargs):
            if self._ocupado:
                return None
            try:
                return fn(*args, **kwargs)
            except tk.TclError:
                return None            # la ventana pudo destruirse a mitad
            except Exception as e:     # noqa: BLE001 — red de seguridad global
                try:
                    messagebox.showerror("Error inesperado",
                                         f"{type(e).__name__}: {e}")
                except Exception:
                    pass
                return None
        return envuelto

    # ------------------------------ UI ------------------------------------ #
    def _construir_ui(self):
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
        # width fijo (en caracteres) + anchor w: la etiqueta reserva SIEMPRE el
        # mismo ancho, así el panel no se reajusta al cambiar el nivel.
        ttk.Label(panel, textvariable=self.calidad_txt, width=38,
                  anchor="w").grid(
            row=3, column=0, columnspan=2, sticky="w", pady=(6, 0))
        # tk.Scale (no ttk) para tener 5 PARADAS DISCRETAS y uniformes: resolution
        # =1 hace que salte a enteros y tickinterval=1 marca las 5 posiciones.
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
        # width=1 + sticky="ew": la etiqueta se ESTIRA para llenar el ancho ya
        # fijado por (panel + lienzo) y recorta el texto sobrante, en vez de
        # pedir más ancho y hacer crecer la ventana con los mensajes largos.
        ttk.Label(cont, textvariable=self.estado, relief="sunken",
                  anchor="w", padding=4, width=1).grid(
            row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))

    def _campo(self, panel, fila, etiqueta, valor):
        ttk.Label(panel, text=etiqueta).grid(row=fila, column=0, sticky="w", pady=1)
        e = ttk.Entry(panel, width=7)
        e.insert(0, valor)
        e.grid(row=fila, column=1, sticky="e", pady=1)
        return e

    # --------------------------- parámetros ------------------------------- #
    def _calidad_cambia(self, *_):
        """Actualiza la etiqueta y aplica el nivel de calidad al planificador."""
        nivel = int(round(float(self.calidad.get())))
        # Solo el número (1..5), sin descriptores: así el texto NUNCA cambia de
        # longitud/ancho al variar el nivel y el panel no se reajusta. La escala
        # 1 = rápido … 5 = máxima calidad se explica en la etiqueta fija.
        self.calidad_txt.set(f"Calidad de ruta (1 rápida ⟷ 5 máxima):  {nivel}")
        self.planificador.configurar_calidad(nivel)

    def _params(self):
        # Tamaño y velocidad de los vehículos: valores fijos (ya no editables).
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

    # ----------------------------- mapa ----------------------------------- #
    def nuevo_mapa(self):
        p = self._params()
        dens = p[4] if p else 0.0
        self._detener()
        self.env.generar(dens)
        self.inicios, self.metas, self.frames, self.vehiculos = [], [], [], []
        self.frame = 0
        self.estado.set("Nuevo mapa generado. Genera posiciones.")
        self._dibujar_estatico()

    # ------------------------ posiciones ---------------------------------- #
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
                ith = th_meta            # orientación inicial hacia la meta si cabe
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
        """Reasigna inicio y destino del vehículo (modo aleatorio) evitando a
        los demás, para reintentar la planificación si no halló ruta."""
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

    # -------------------- colocación manual (clics) ----------------------- #
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

    # ----------------------- calcular y simular --------------------------- #
    def calcular_y_simular(self):
        p = self._params()
        if not p:
            return
        n, length, width, vmax, _ = p
        if len(self.inicios) < n or len(self.metas) < n:
            messagebox.showinfo("Faltan posiciones",
                                "Genera o coloca primero todas las posiciones.")
            return

        self._detener()
        # Reconstruye los vehículos desde las posiciones FUENTE en CADA cálculo.
        # Así, al cambiar la calidad (u otro parámetro) y pulsar de nuevo, se
        # REPLANIFICA de verdad todo el conjunto original —incluidos los que
        # fallaron en un intento previo— en vez de arrastrar las trayectorias ya
        # calculadas o quedarse solo con los supervivientes del intento anterior.
        self._crear_vehiculos(length, width, vmax, n)
        self._ocupado = True
        try:
            self._planificar_todo(n)
        finally:
            self._ocupado = False

    def _planificar_todo(self, n):
        self.estado.set("Planificando de forma cooperativa (Hybrid A* espacio-tiempo)…")
        self.root.update()
        if not self.root.winfo_exists():
            return
        aleatorio = self.modo.get() == "aleatorio"

        # Planificación cooperativa priorizada: el vehículo i evita (a) los
        # obstáculos fijos, (b) las metas de los vehículos POSTERIORES (bloqueos
        # estáticos, para no aparcar donde otro debe aparcar) y (c) las
        # TRAYECTORIAS temporales de los ANTERIORES (obstáculos móviles).
        #
        # SIN límite de reloj: cada búsqueda corre hasta CONCLUIR de verdad —
        # encuentra ruta o agota la frontera (⇒ la ruta NO existe, definitivo).
        # El único tope es un TECHO DE EXPANSIONES muy alto (nodos, no segundos)
        # como red anti-cuelgue; si se alcanza, el resultado es INCONCLUSO (la
        # búsqueda era demasiado grande), y así se informa —no se confunde con un
        # "no existe". La UI se refresca periódicamente para no congelarse.
        CAP_COMPLETO = 6_000_000                  # techo de nodos (anti-cuelgue)
        reservas = Reservas()
        planificados = []                        # vehículos con ruta válida
        sin_ruta = 0                             # no existe ruta (definitivo)
        inconcluso = 0                           # búsqueda no agotada (techo)
        self.planificador.tick = self._tick_plan
        self.planificador.deadline = None        # sin plazo de reloj
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
                    # No se pudo con este vehículo: se EXCLUYE y se sigue con el
                    # resto (los mostrados siguen siendo mutuamente sin colisión,
                    # pues cada uno evita a todos los anteriores YA reservados).
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

        # Solo se simulan/dibujan los vehículos con ruta (nada de coches fantasma
        # estáticos que otros pudieran atravesar). NO se tocan self.inicios /
        # self.metas (posiciones FUENTE): así un recálculo posterior reintenta el
        # conjunto ORIGINAL completo. El dibujo colorea por veh.idx (identidad
        # estable = índice en la fuente), de modo que cada coche conserva SU color
        # aunque otros queden excluidos.
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
        """Planifica UN vehículo hasta una CONCLUSIÓN real (sin límite de reloj).
        Devuelve (motivo, traj):
          · ("ok", traj)        → ruta encontrada;
          · ("sin_ruta", None)  → la frontera de búsqueda se agotó: NO existe
                                   ruta para estos extremos (respuesta definitiva);
          · ("limite", None)    → se alcanzó el techo de expansiones: resultado
                                   INCONCLUSO (la búsqueda era demasiado grande).
        En modo aleatorio, ante un "sin_ruta" definitivo prueba a reubicar los
        extremos (quizá otras posiciones sí sean factibles); un "limite" NO se
        reubica (no sabemos si había ruta, reubicar solo escondería el problema).
        La calidad la fija el usuario y NO se altera aquí."""
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
        if traj:                       # traj de 1 sola pose: el coche YA está en
            # su meta (inicio dentro de la tolerancia). Es un éxito trivial; se
            # devuelve una trayectoria válida (queda parado) para no romper el
            # muestreo de fotogramas ni las reservas.
            return "ok", [traj[0], traj[0]]
        # Sin trayectoria utilizable: NUNCA devolver "ok" (evita meter None en
        # las reservas). Si el planificador dijo "ok" pero no hay traj, es un
        # fallo efectivo → se reporta como sin_ruta.
        return (motivo if motivo != "ok" else "sin_ruta"), None

    def _tick_plan(self, expand):
        """Progreso periódico durante la búsqueda: refresca la ventana (para que
        no se congele) y muestra las expansiones. El guard _ocupado impide que
        cualquier pulsación reentrante afecte al cálculo."""
        self.estado.set(f"{self._plan_msg}  ({expand:,} nodos explorados)")
        self.root.update()

    def _bloqueos_metas(self, excepto):
        """OBB cuadrados (independientes de la orientación) en las metas de los
        demás vehículos, para que nadie planifique aparcar donde otro aparcará."""
        bloq = []
        for j, veh in enumerate(self.vehiculos):
            if j == excepto:
                continue
            mx, my = veh.meta
            poly = obb_corners(mx, my, 0.0, veh.diag, veh.diag)
            bloq.append((poly, (mx, my, veh.diag / 2)))
        return bloq

    # --------------------------- reproducción ----------------------------- #
    def reproducir(self):
        if not self.frames:
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
        except tk.TclError:
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

    # ------------------------------ dibujo -------------------------------- #
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
            # dimensiones FIJAS de vehículo (constantes): así el OBB de inicio no
            # depende de self.vehiculos[i] —que puede no existir tras excluir a un
            # fallido— y nunca aparece el rectángulo enorme del antiguo fallback.
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
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
