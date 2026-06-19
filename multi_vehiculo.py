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
import bisect
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
        self.a_max = 3.0
        self.traj = []                # [(x, y, theta), ...]  una pose por paso
        self.dt_plan = 0.4            # duración de cada paso (s)

    @property
    def radio_giro_min(self):
        return self.wheelbase / math.tan(self.delta_max)

    @property
    def diag(self):
        return math.hypot(self.length, self.width)

    def pose_en_tiempo(self, t):
        """Pose interpolada en el instante t (s) de la trayectoria temporal.
        Tras el final, se queda en su meta (vehículo aparcado)."""
        if not self.traj:
            return self.inicio
        f = t / self.dt_plan
        i = int(f)
        if i >= len(self.traj) - 1:
            return self.traj[-1]
        frac = f - i
        x0, y0, th0 = self.traj[i]
        x1, y1, th1 = self.traj[i + 1]
        dth = math.atan2(math.sin(th1 - th0), math.cos(th1 - th0))
        return (x0 + (x1 - x0) * frac, y0 + (y1 - y0) * frac, th0 + dth * frac)

    @property
    def duracion(self):
        return max(0, len(self.traj) - 1) * self.dt_plan


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
        self.res_ang = math.radians(18)
        self.dt = 0.4                 # duración de cada acción macro (s)
        self.subpasos = 4             # → TAU = dt/subpasos = 0.1 s = DT
        self.goal_tol = 1.6
        self.max_exp = 120000
        self.k_max = 900              # horizonte temporal (pasos finos ≈ 90 s)
        self.dist_tiro = 14.0
        self.margen = 0.10            # holgura contra obstáculos fijos
        self.margen_din = 0.30        # holgura contra otros vehículos
        self.peso_h = 1.5             # A* ponderado (acelera la búsqueda)
        self.bloqueos = []            # metas ajenas (estáticas) [(poly, bb)]

    # ---- utilidades de ocupación estática (obstáculos + metas ajenas) ---- #
    def _clave(self, x, y, th, k):
        return (int(x / self.res_pos), int(y / self.res_pos),
                int((th % (2 * math.pi)) / self.res_ang), k)

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

    def _mover(self, x, y, th, speed, delta, L):
        """Integra una acción macro (modelo de bicicleta) a velocidad 'speed'
        (con signo) y dirección 'delta', devolviendo la lista de poses de cada
        SUBPASO (a TAU = dt/subpasos = DT segundos) sobre el arco real, o None
        si algún subpaso choca con un obstáculo fijo."""
        h = self.dt / self.subpasos
        subs = []
        for _ in range(self.subpasos):
            th = th + (1.0 / L) * math.tan(delta) * (speed * h)
            x = x + math.cos(th) * speed * h
            y = y + math.sin(th) * speed * h
            if not self._libre(x, y, th):
                return None
            subs.append((x, y, th))
        return subs

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

    def _tiro_directo(self, x, y, th, k, gx, gy, L, dmax, v_c, reservas):
        """Conexión analítica hacia la meta (pure pursuit) comprobando estática
        y reservas dinámicas en CADA subpaso. Devuelve la lista densa de poses
        (a TAU) o None."""
        poses = []
        kk = k
        for _ in range(120):
            d = math.hypot(gx - x, gy - y)
            if d <= self.goal_tol:
                return poses
            alpha = math.atan2(gy - y, gx - x) - th
            alpha = math.atan2(math.sin(alpha), math.cos(alpha))
            delta = max(-dmax, min(dmax, math.atan2(2.0 * L * math.sin(alpha), max(d, 1e-3))))
            subs = self._mover(x, y, th, v_c, delta, L)
            if subs is None or not self._subs_libres_din(subs, kk, reservas):
                return None
            poses.extend(subs)
            x, y, th = subs[-1]
            kk += self.subpasos
        return None

    def planificar(self, veh, reservas):
        """Devuelve la trayectoria temporal [(x,y,th), ...] (una pose por paso)
        o None si no halla solución coordinada."""
        self._len, self._wid = veh.length, veh.width
        self.diag = veh.diag
        L = veh.wheelbase
        dmax = veh.delta_max
        sx, sy, sth = veh.inicio
        gx, gy = veh.meta
        v_c = veh.v_max

        # repertorio de acciones (velocidad con signo, dirección)
        deltas = [-dmax, -dmax / 2, 0.0, dmax / 2, dmax]
        acciones = [(0.0, 0.0)]                       # ESPERAR
        for s in (v_c, 0.6 * v_c):                    # avanzar
            acciones += [(s, d) for d in deltas]
        for d in (-dmax, 0.0, dmax):                  # marcha atrás
            acciones.append((-0.5 * v_c, d))

        clave0 = self._clave(sx, sy, sth, 0)
        contador = 0
        h0 = math.hypot(sx - gx, sy - gy) / v_c
        # cola: (f, contador, x, y, theta, k_fino, g)
        abierto = [(self.peso_h * h0, contador, sx, sy, sth, 0, 0.0)]
        mejor_g = {clave0: 0.0}
        padre = {}
        arista = {}                       # clave -> poses densas del tramo (subpasos)
        expand = 0
        ns = self.subpasos

        while abierto and expand < self.max_exp:
            _f, _, x, y, th, k, g = heapq.heappop(abierto)
            expand += 1
            ck = self._clave(x, y, th, k)

            d_goal = math.hypot(x - gx, y - gy)
            if d_goal <= self.goal_tol:
                return self._reconstruir(padre, arista, ck, (sx, sy, sth))

            if d_goal <= self.dist_tiro:
                tiro = self._tiro_directo(x, y, th, k, gx, gy, L, dmax, v_c, reservas)
                if tiro is not None:
                    base = self._reconstruir(padre, arista, ck, (sx, sy, sth))
                    return base + tiro

            if k >= self.k_max:
                continue

            for speed, delta in acciones:
                if speed == 0.0:                      # esperar: misma pose, ns subpasos
                    subs = [(x, y, th)] * ns
                else:
                    subs = self._mover(x, y, th, speed, delta, L)
                    if subs is None:
                        continue
                if not self._subs_libres_din(subs, k, reservas):
                    continue
                nx, ny, nth = subs[-1]
                nk = k + ns
                ng = g + self.dt                      # coste = tiempo transcurrido
                if speed < 0:
                    ng += 0.6 * self.dt               # penaliza la marcha atrás
                elif speed == 0.0:
                    ng += 0.15 * self.dt              # leve penalización por esperar
                ng += 0.05 * self.dt * abs(delta)     # suaviza el giro
                key = self._clave(nx, ny, nth, nk)
                if ng < mejor_g.get(key, float("inf")):
                    mejor_g[key] = ng
                    padre[key] = ck
                    arista[key] = subs
                    contador += 1
                    h = math.hypot(nx - gx, ny - gy) / v_c
                    heapq.heappush(abierto,
                                   (ng + self.peso_h * h, contador, nx, ny, nth, nk, ng))
        return None

    def _reconstruir(self, padre, arista, ck, inicio):
        tramos = []
        while ck in padre:
            tramos.append(arista[ck])
            ck = padre[ck]
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

        self._construir_ui()
        self.env.generar(0.0)
        self._dibujar_estatico()

    # ------------------------------ UI ------------------------------------ #
    def _construir_ui(self):
        cont = ttk.Frame(self.root, padding=8)
        cont.grid(row=0, column=0)

        panel = ttk.Frame(cont, padding=(0, 0, 12, 0))
        panel.grid(row=0, column=0, sticky="n")

        ttk.Label(panel, text="Parámetros", font=("", 13, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 6))
        self.e_num = self._campo(panel, 1, "Nº de vehículos:", "4")
        self.e_len = self._campo(panel, 2, "Largo vehículo (m):", "2.6")
        self.e_wid = self._campo(panel, 3, "Ancho vehículo (m):", "1.4")
        self.e_vmax = self._campo(panel, 4, "Velocidad máx (m/s):", "5.0")
        self.e_dens = self._campo(panel, 5, "Densidad obstáculos (0-1):", "0.0")

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
                   command=self.generar_posiciones).grid(
            row=10, column=0, columnspan=2, sticky="ew", pady=(6, 2))
        ttk.Button(panel, text="Nuevo mapa de obstáculos",
                   command=self.nuevo_mapa).grid(
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
                   command=self.calcular_y_simular).grid(
            row=16, column=0, columnspan=2, sticky="ew", pady=2)
        ttk.Button(panel, text="↺  Reproducir de nuevo",
                   command=self.reproducir).grid(
            row=17, column=0, columnspan=2, sticky="ew", pady=2)
        ttk.Button(panel, text="⏸  Pausar / reanudar",
                   command=self.pausar).grid(
            row=18, column=0, columnspan=2, sticky="ew", pady=2)
        ttk.Button(panel, text="⟲  Reiniciar",
                   command=self.reiniciar).grid(
            row=19, column=0, columnspan=2, sticky="ew", pady=2)
        ttk.Button(panel, text="✕  Salir",
                   command=self.root.destroy).grid(
            row=20, column=0, columnspan=2, sticky="ew", pady=(2, 0))

        self.canvas = tk.Canvas(cont, width=int(W * SCALE), height=int(H * SCALE),
                                bg=COL_FONDO, highlightthickness=2,
                                highlightbackground=COL_BORDE)
        self.canvas.grid(row=0, column=1)
        self.canvas.bind("<Button-1>", self.click_mapa)

        self.estado = tk.StringVar(value="Listo. Ajusta parámetros y genera posiciones.")
        ttk.Label(cont, textvariable=self.estado, relief="sunken",
                  anchor="w", padding=4).grid(
            row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))

    def _campo(self, panel, fila, etiqueta, valor):
        ttk.Label(panel, text=etiqueta).grid(row=fila, column=0, sticky="w", pady=1)
        e = ttk.Entry(panel, width=7)
        e.insert(0, valor)
        e.grid(row=fila, column=1, sticky="e", pady=1)
        return e

    # --------------------------- parámetros ------------------------------- #
    def _params(self):
        try:
            n = int(self.e_num.get())
            length = float(self.e_len.get())
            width = float(self.e_wid.get())
            vmax = float(self.e_vmax.get())
            dens = float(self.e_dens.get())
        except ValueError:
            messagebox.showerror("Error", "Los parámetros deben ser numéricos.")
            return None
        if not (1 <= n <= len(PALETA)):
            messagebox.showerror("Error", f"Nº de vehículos entre 1 y {len(PALETA)}.")
            return None
        if not (1.0 <= length <= 8.0 and 0.8 <= width <= 4.0 and width < length):
            messagebox.showerror("Error", "Tamaño irreal: largo 1-8 m, ancho 0.8-4 m, ancho<largo.")
            return None
        if not (0.5 <= vmax <= 20):
            messagebox.showerror("Error", "Velocidad máx entre 0.5 y 20 m/s.")
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

    def _crear_vehiculos(self, length, width, vmax):
        self.vehiculos = [
            Vehiculo(i, self.inicios[i], self.metas[i], length, width, vmax)
            for i in range(len(self.inicios))]

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
        n = p[0]
        if len(self.vehiculos) < n:
            messagebox.showinfo("Faltan posiciones",
                                "Genera o coloca primero todas las posiciones.")
            return

        self._detener()
        self.estado.set("Planificando de forma cooperativa (Hybrid A* espacio-tiempo)…")
        self.root.update()
        aleatorio = self.modo.get() == "aleatorio"

        # Planificación cooperativa priorizada: el vehículo i evita (a) los
        # obstáculos fijos, (b) las metas de los vehículos POSTERIORES (bloqueos
        # estáticos, para no aparcar donde otro debe aparcar) y (c) las
        # TRAYECTORIAS temporales de los ANTERIORES (obstáculos móviles).
        reservas = Reservas()
        for i, veh in enumerate(self.vehiculos):
            self.planificador.bloqueos = self._bloqueos_metas(excepto=i)
            traj = self.planificador.planificar(veh, reservas)
            intentos = 0
            while (traj is None or len(traj) < 2) and aleatorio and intentos < 12:
                if not self._reubicar(veh):
                    break
                self.planificador.bloqueos = self._bloqueos_metas(excepto=i)
                traj = self.planificador.planificar(veh, reservas)
                intentos += 1
            if traj is None or len(traj) < 2:
                messagebox.showwarning("Sin ruta",
                    f"No se encontró ruta coordinada para el vehículo {veh.idx + 1}.\n"
                    "Prueba otras posiciones, menos vehículos/obstáculos o uno más pequeño.")
                return
            veh.traj = traj
            veh.dt_plan = self.planificador.dt / self.planificador.subpasos
            reservas.add(traj, veh.length, veh.width)
            self.estado.set(f"Planificado vehículo {i + 1}/{n}…")
            self.root.update()
        self._dibujar_estatico()

        self.frames = construir_frames(self.vehiculos)
        self.estado.set(f"Rutas coordinadas ({len(self.frames)} fotogramas). "
                        f"Todos llegan sin colisiones. Reproduciendo…")
        self.reproducir()

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
        if not self.reproduciendo:
            return
        self._dibujar_frame()
        if self.frame >= len(self.frames) - 1:
            self.reproduciendo = False
            self.estado.set("Reproducción finalizada.")
            return
        self.frame += 1
        self.anim_id = self.root.after(int(self.vel.get()), self._anim)

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
            length = self.vehiculos[i].length if i < len(self.vehiculos) else 4.0
            width = self.vehiculos[i].width if i < len(self.vehiculos) else 2.0
            poly = obb_corners(ini[0], ini[1], ini[2], length, width)
            c.create_polygon(self._poly_px(poly), outline=col, fill="",
                             width=1, dash=(3, 3))

    def _rutas(self):
        for i, veh in enumerate(self.vehiculos):
            if veh.traj:
                col = PALETA[i % len(PALETA)]
                pts = []
                for x, y, _ in veh.traj:
                    pts.extend((x * SCALE, y * SCALE))
                self.canvas.create_line(pts, fill=col, width=1, smooth=True)

    def _dibujar_estatico(self):
        self._fondo()
        for i, veh in enumerate(self.vehiculos):
            self._dibujar_coche(veh.inicio[0], veh.inicio[1], veh.inicio[2],
                                veh.length, veh.width, PALETA[i % len(PALETA)], i + 1)

    def _dibujar_frame(self):
        self._fondo()
        self._rutas()
        fr = self.frames[self.frame]
        for i, veh in enumerate(self.vehiculos):
            x, y, th = fr[veh.idx]
            self._dibujar_coche(x, y, th, veh.length, veh.width,
                                PALETA[i % len(PALETA)], i + 1)
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
