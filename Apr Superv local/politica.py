#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Política neuronal de control de flota: definición COMPARTIDA de las entradas
de la red, del modelo y del despliegue en bucle cerrado.

La red recibe, por vehículo y por instante, un vector con:
  · estado propio (posición, orientación, velocidad) y parámetros del vehículo,
  · la meta (en el sistema de referencia del propio vehículo) y el ángulo de
    llegada exigido,
  · los últimos H_PASADO controles aplicados (a, giro), normalizados,
  · los vehículos vecinos RELEVANTES (hasta N_VECINOS): un vehículo entra solo
    si, yendo ambos a su velocidad máxima en línea recta el uno hacia el otro,
    podrían colisionar dentro de HORIZONTE_VECINO iteraciones; de cada uno se dan
    las DOS ESQUINAS OPUESTAS de su rectángulo en los ejes del propio vehículo
    (posición + tamaño + orientación a la vez), distancia, ángulo y velocidad
    relativas y su preferencia (grupo/prioridad),
  · el CONTEXTO del escenario (igual para toda la flota): el modo de optimización
    elegido por el usuario (secuencial/global/prioridades, one-hot) y el nº de
    vehículos de la flota, para que la red aprenda a comportarse según ellos.
y predice los próximos N_PRED pares (a, giro) NORMALIZADOS por (a_max, giro_max)
del propio vehículo. El mapa NO se codifica: la red se entrena y se usa siempre
sobre el mismo mapa (entorno controlado), así que lo aprende implícitamente de
las coordenadas absolutas.

Este módulo lo usan `entrenar.py` (construcción de muestras desde el CSV) y
`gui.py` (despliegue). Torch solo se importa al crear/cargar la red, de modo que
la GUI en modo clásico no lo necesita.
"""

import math
import os

import numpy as np

from nucleo import DT, W, H, VEH_LEN, ang_norm

# ----------------------------- hiperparámetros ----------------------------- #
N_PRED = 10        # pasos futuros (a, giro) que predice la red de una vez
H_PASADO = 10      # controles pasados que ve como entrada
# Vecinos: TOPE de vehículos incluidos y HORIZONTE del filtro de relevancia.
# Un vecino entra solo si su "tiempo de cierre" —la distancia entre bordes
# dividida por la suma de sus v_max, es decir el mínimo tiempo físico para que
# ambos, a tope y el uno hacia el otro, lleguen a tocarse— es <= HORIZONTE_VECINO
# iteraciones (× DT segundos). Debe ser >= N_PRED (10, los pasos que se
# predicen); 20 añade margen. Ambos son ajustables como hiperparámetros de red
# (equilibrio calidad/cómputo): más vecinos y más horizonte ven más contexto
# pero engordan la entrada.
N_VECINOS = 10
HORIZONTE_VECINO = 20

DIM_EGO = 10       # x, y, sinθ, cosθ, v, largo, ancho, v_max, a_max, giro_max
DIM_META = 6       # meta en ejes propios (dx, dy), dist, sin/cos Δθ_llegada, libre
# Vecino: 2 esquinas OPUESTAS de su rectángulo en ejes propios (4), distancia,
# sinΔθ, cosΔθ, velocidad, preferencia y bandera de presencia.
DIM_VECINO = 10

# CONTEXTO DE ESCENARIO: preferencias del usuario que influyen en la forma de las
# rutas y que son iguales para todos los vehículos e instantes del mismo run.
# La red las recibe como entrada para aprender a comportarse según ellas.
MODOS_OPT = ("secuencial", "global", "prioridades")   # one-hot del modo de flota
DIM_CTX = len(MODOS_OPT) + 1   # one-hot del modo + nº de vehículos (saturado)

# POSICIÓN EN SENOS Y COSENOS (codificación de Fourier). El mapa no es una
# entrada: la red tiene que deducir dónde están los muros a partir de la x y la
# y crudas. Y eso una red densa lo hace fatal, porque un muro es una función
# ABRUPTA de la coordenada —a un lado se puede pasar y un palmo más allá no— y
# una MLP alimentada con la coordenada cruda tiende a interpolar suave entre
# ejemplos lejanos. El arreglo estándar es dar también la posición en varias
# longitudes de onda: sin/cos de 2π·x/λ y de 2π·y/λ para cada λ. Así dos puntos
# separados por un palmo llegan a la primera capa como vectores DISTINTOS, sin
# que la red tenga que fabricar esa distinción ella sola.
#
# Las λ se dan en LARGOS DE VEHÍCULO y cubren un ABANICO ancho, de 1/32 de
# vehículo a 64, doblando cada vez:
#     1/32 · 1/16 · 1/8 · 1/4 · 1/2 · 1 · 2 · 4 · 8 · 16 · 32 · 64
#
# Ancho a propósito. La escala de las ondas es FIJA: no se ajusta al vehículo
# que mira. Tiene que ser así porque lo que codifican es la posición EN EL MAPA,
# y el mapa es el mismo siempre; si cada vehículo usara su propia regla, el
# mismo punto se codificaría distinto según quién pase por ahí y la red no
# podría compartir entre unos y otros lo que aprende sobre dónde están los
# muros —justo para lo que sirve este bloque—. Con un tamaño nunca visto la
# codificación sería nueva entera y la red no reconocería un muro que conoce.
#
# La forma de aguantar vehículos de cualquier tamaño no es mover la regla, es
# que la regla tenga marcas a todas las escalas. Con este abanico, un vehículo
# 4 veces menor que el típico sigue teniendo ondas a 1/8 de su cuerpo, y uno 8
# veces mayor también: siempre hay unas cuantas a la escala de su holgura, y la
# red usa las que le sirven. Cuál le sirve lo sabe porque su largo y su ancho
# van en el vector de entrada.
#
# Doblando y no sumando de una en una: con doce ondas lineales (1, 2, 3…
# vehículos) no se bajaría nunca del vehículo entero, que es donde está el
# problema —se conduce con holguras de en torno a un 10 % del cuerpo—.
#
# LARGO_REF es solo la marca del "1" de esa regla, no una suposición sobre lo
# que se va a usar: el abanico se extiende cinco dobleces por debajo y seis por
# encima, así que acertar con él no es crítico.
#
# La λ es la MISMA para todos los vehículos (sale del largo POR DEFECTO, no del
# de cada uno). Si cada vehículo usara su propio tamaño, dos vehículos distintos
# codificarían el mismo punto del mapa de formas distintas y la red no podría
# compartir lo que aprende sobre dónde están los muros.
#
# Van ordenadas de MÁS FINA a más gruesa: quedarse con las primeras conserva el
# detalle local, que es lo que este bloque aporta —la posición global ya está en
# el vector, en las x e y crudas—.
#
# Va como EJE DEL BARRIDO (0 lo desactiva) porque es una hipótesis, no un
# hecho: si con 0 se choca igual que con 8, el problema estaba en otro sitio.
# El valor 0 por defecto mantiene DIM_ENTRADA como estaba, así que los modelos
# guardados antes de esto siguen cargando.
#
# LAMBDA_FINA se GUARDA en el .pt de cada modelo y se restaura al cargarlo. Si
# mañana se cambia el tamaño del vehículo, los modelos entrenados antes siguen
# viendo la misma codificación con la que aprendieron —que es lo único que les
# vale— y solo los nuevos se recalibran al tamaño nuevo.
# RAYOS al obstáculo (ver rayos.py): cuántas direcciones mira el vehículo. 0 lo
# desactiva. A diferencia de las ondas, esto no recodifica lo que la red ya
# tiene: le da la distancia libre, que es la magnitud que decide si choca.
N_RAYOS = 0

N_FOURIER = 0        # nº de longitudes de onda (0 = desactivado)
N_FOURIER_MAX = 12   # tope del superset; subirlo obliga a rehacer las muestras
DIM_FOURIER = 4      # sin/cos × (x, y) por longitud de onda

# Largo de vehículo que marca el "1" de la regla de las ondas. Los vehículos que
# se usen pueden ser mayores o menores que esto —el dataset actual los sortea
# entre 0,70 y 1,20, pero eso no ata a nada—, y por eso el abanico se extiende
# muy por debajo y muy por encima: LARGO_REF sitúa la regla, no predice el
# tamaño. `preparar_datos` avisa si la mediana real se aleja mucho, no porque
# rompa nada, sino porque entonces conviene recentrar el abanico.
#
# VEH_LEN no sirve para esto: es el largo por defecto de la clase (1,3), un
# tamaño que en los datos no aparece nunca.
LARGO_REF = 0.95
LAMBDA_FINA = LARGO_REF / 32.0   # onda más corta: 1/32 de vehículo

DIM_ENTRADA = (DIM_EGO + DIM_META + 2 * H_PASADO + DIM_VECINO * N_VECINOS
               + DIM_CTX + DIM_FOURIER * N_FOURIER + N_RAYOS)

# Mapa contra el que se trazan los rayos. Lo registra quien sepa cuál es
# (`vectorizado`, `preparar_datos`); si nadie lo hace, se carga el de
# entrenamiento, que es el único que usa este proyecto.
_MAPA_RAYOS = None


def configurar_mapa_rayos(oc, mundo):
    global _MAPA_RAYOS
    _MAPA_RAYOS = (oc, mundo)


def mapa_rayos():
    global _MAPA_RAYOS
    if _MAPA_RAYOS is None:
        from nucleo import Entorno, cargar_mapa_en
        ruta = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "mapas", "mapa_entrenamiento.json")
        env = Entorno()
        cargar_mapa_en(env, ruta)
        _MAPA_RAYOS = ((env.np_c, env.np_bb), (W, H))
    return _MAPA_RAYOS


def rasgos_rayos(x, y, th, n_rayos=None):
    """(..., n) con la distancia libre normalizada en cada dirección."""
    import rayos
    n = N_RAYOS if n_rayos is None else int(n_rayos)
    (oc, _obb), mundo = mapa_rayos()
    return rayos.distancias(x, y, th, n, oc, mundo)

MODELO_PT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "modelos", "politica.pt")

# Criterio de LLEGADA durante el despliegue (coherente con la prioridad del
# proyecto: llegar siempre y rápido pesa más que clavar el ángulo exacto).
DIST_LLEGADA = 0.40    # m al centro de la meta
V_LLEGADA = 0.25       # |v| máxima para considerarse aparcado


# --------------------------------------------------------------------------- #
# Vector de entrada (idéntico en entrenamiento y despliegue)
# --------------------------------------------------------------------------- #
def rasgos_fourier(x, y, n_frecs=None):
    """(..., 4·n_frecs) con la posición en senos y cosenos.

    Acepta escalares o arrays. Las columnas van AGRUPADAS POR LONGITUD DE ONDA
    —(sin x, cos x, sin y, cos y) de λ = 1/8 de vehículo, luego las de 1/4…— y
    de la más FINA a la más gruesa. Ese orden es el que permite quedarse con las
    4·L primeras columnas para usar solo las L ondas más finas, que es como el
    barrido recorta este bloque del superset sin rehacer los datos.

    La salida está ya en [-1, 1] por construcción: no hacen falta escalas
    aprendidas ni normalización aparte."""
    n = N_FOURIER if n_frecs is None else int(n_frecs)
    u = np.asarray(x, dtype=np.float64)
    w = np.asarray(y, dtype=np.float64)
    if n <= 0:
        return np.zeros(u.shape + (0,), dtype=np.float32)
    # k = 2π/λ con λ = LAMBDA_FINA · 2^j, de la más fina a la más gruesa.
    k = (2.0 * math.pi) / (LAMBDA_FINA * (2.0 ** np.arange(n)))
    ku, kw = u[..., None] * k, w[..., None] * k
    out = np.stack([np.sin(ku), np.cos(ku), np.sin(kw), np.cos(kw)], axis=-1)
    return out.reshape(u.shape + (DIM_FOURIER * n,)).astype(np.float32)


def vector_entrada(ego, otros):
    """Construye el vector de entrada de la red (np.float32, DIM_ENTRADA).

    'ego': dict con x, y, th, v, largo, ancho, v_max, a_max, giro_max (rad),
           grupo, prioridad, meta=(mx, my), meta_th (rad o None) y
           pasado=[(a, giro), ...] en unidades REALES (más antiguo primero,
           hasta H_PASADO elementos).
    'otros': lista de dicts con x, y, th, v, largo, ancho, grupo, prioridad."""
    x, y, th = ego["x"], ego["y"], ego["th"]
    c, s = math.cos(th), math.sin(th)
    out = np.zeros(DIM_ENTRADA, dtype=np.float32)

    out[0:DIM_EGO] = (x, y, s, c, ego["v"], ego["largo"], ego["ancho"],
                      ego["v_max"], ego["a_max"], ego["giro_max"])

    # Meta en el sistema de referencia del vehículo.
    mx, my = ego["meta"]
    dx, dy = mx - x, my - y
    dxe = dx * c + dy * s
    dye = -dx * s + dy * c
    meta_th = ego.get("meta_th")
    if meta_th is None:
        dth_m, libre = 0.0, 1.0
    else:
        dth_m, libre = ang_norm(meta_th - th), 0.0
    out[DIM_EGO:DIM_EGO + DIM_META] = (dxe, dye, math.hypot(dx, dy),
                                       math.sin(dth_m), math.cos(dth_m), libre)

    # Controles pasados, normalizados, alineados a la DERECHA (el más reciente
    # al final; sin historia suficiente se rellena con ceros por la izquierda).
    base = DIM_EGO + DIM_META
    pasado = list(ego.get("pasado", ()))[-H_PASADO:]
    off = base + 2 * (H_PASADO - len(pasado))
    for j, (a, g) in enumerate(pasado):
        out[off + 2 * j] = a / max(ego["a_max"], 1e-6)
        out[off + 2 * j + 1] = g / max(ego["giro_max"], 1e-6)

    # Vecinos RELEVANTES por TIEMPO DE CIERRE: un vecino entra solo si, yendo
    # ambos a su v_max en línea recta el uno hacia el otro, podrían tocarse
    # dentro de HORIZONTE_VECINO iteraciones (t_cierre <= umbral). Se ordenan por
    # urgencia (t_cierre ascendente) y se toman hasta N_VECINOS; los huecos
    # restantes quedan a cero con la bandera de presencia (última) en 0, así que
    # "menos de N_VECINOS cerca" se representa de forma natural.
    base = DIM_EGO + DIM_META + 2 * H_PASADO
    clave_ego = (ego.get("grupo", 1), ego.get("prioridad", 1))
    umbral = HORIZONTE_VECINO * DT
    radio_ego = 0.5 * math.hypot(ego["largo"], ego["ancho"])
    relevantes = []
    for o in otros:
        dx, dy = o["x"] - x, o["y"] - y
        dist = math.hypot(dx, dy)
        # Distancia entre BORDES (aprox. por los radios de cada OBB) y tiempo de
        # cierre en el peor caso: ambos a v_max, directos el uno al otro.
        holgura = max(0.0, dist - radio_ego
                      - 0.5 * math.hypot(o["largo"], o["ancho"]))
        v_cierre = ego["v_max"] + o["v_max"]
        t_cierre = holgura / v_cierre if v_cierre > 1e-6 else 0.0
        if t_cierre <= umbral:
            relevantes.append((t_cierre, dx, dy, dist, o))
    # Orden por tiempo de cierre (más urgente primero): solo fija a qué slot va
    # cada vecino, para que la entrada sea determinista; no es una característica.
    relevantes.sort(key=lambda r: r[0])
    for j, (t_cierre, dx, dy, dist, o) in enumerate(relevantes[:N_VECINOS]):
        dth = ang_norm(o["th"] - th)
        clave_o = (o.get("grupo", 1), o.get("prioridad", 1))
        # Menor (grupo, prioridad) → planifica antes → tiene preferencia.
        pref = 1.0 if clave_o < clave_ego else (-1.0 if clave_o > clave_ego
                                                else 0.0)
        # Dos ESQUINAS OPUESTAS del rectángulo del vecino, en los EJES DEL PROPIO
        # vehículo: entre ambas codifican a la vez su posición, su tamaño y su
        # orientación (el mismo estado que guarda el CSV), sin necesidad de pasar
        # largo y ancho por separado.
        co, so = math.cos(o["th"]), math.sin(o["th"])
        hx = 0.5 * o["largo"] * co - 0.5 * o["ancho"] * so
        hy = 0.5 * o["largo"] * so + 0.5 * o["ancho"] * co
        p1x, p1y = dx + hx, dy + hy
        p2x, p2y = dx - hx, dy - hy
        k = base + DIM_VECINO * j
        out[k:k + DIM_VECINO] = (p1x * c + p1y * s, -p1x * s + p1y * c,
                                 p2x * c + p2y * s, -p2x * s + p2y * c,
                                 dist, math.sin(dth), math.cos(dth), o["v"],
                                 pref, 1.0)

    # Contexto de ESCENARIO (constante en todo el run): modo de optimización de
    # flota elegido por el usuario (one-hot) y nº de vehículos de la flota. El nº
    # se deduce de 'otros' (que aquí es la lista COMPLETA de compañeros, antes del
    # filtro de vecinos) salvo que se pase explícito en el ego.
    base = DIM_ENTRADA - DIM_CTX - DIM_FOURIER * N_FOURIER - N_RAYOS
    modo = ego.get("opt", "secuencial")
    if modo in MODOS_OPT:
        out[base + MODOS_OPT.index(modo)] = 1.0
    n_veh = ego.get("n_veh")
    if n_veh is None:
        n_veh = len(otros) + 1
    # Nº de vehículos por una función SATURANTE y sin parámetros: 1 - 1/n mapea
    # [1, ∞) → [0, 1) de forma monótona, sin imponer un máximo, con la mayor
    # resolución en flotas pequeñas (donde el nº de compañeros aún importa) y
    # saturando cuando ya solo cuenta la densidad local (que ven los vecinos).
    out[base + len(MODOS_OPT)] = 1.0 - 1.0 / max(1, n_veh)

    # Posición en senos y cosenos, AL FINAL del vector: así añadirla no mueve
    # ninguna de las columnas anteriores.
    fin_ondas = base + DIM_CTX + DIM_FOURIER * N_FOURIER
    if N_FOURIER > 0:
        out[base + DIM_CTX:fin_ondas] = rasgos_fourier(x, y)
    if N_RAYOS > 0:
        out[fin_ondas:] = rasgos_rayos(x, y, th)
    return out


def vehiculo_a_ego(veh, x, y, th, v, pasado, opt="secuencial", n_veh=None):
    """dict 'ego' de vector_entrada a partir de un Vehiculo y su estado actual.
    'opt' es el modo de optimización de flota del escenario y 'n_veh' el nº de
    vehículos de la flota (si None, se deduce de los compañeros en el vector)."""
    return {"x": x, "y": y, "th": th, "v": v,
            "largo": veh.length, "ancho": veh.width,
            "v_max": veh.v_max, "a_max": veh.a_max, "giro_max": veh.delta_max,
            "grupo": veh.grupo, "prioridad": veh.prioridad,
            "meta": veh.meta, "meta_th": veh.meta_th, "pasado": pasado,
            "opt": opt, "n_veh": n_veh}


# --------------------------------------------------------------------------- #
# Red (torch solo se importa aquí dentro)
# --------------------------------------------------------------------------- #
def configurar_representacion(n_vecinos=N_VECINOS, horizonte=HORIZONTE_VECINO,
                              h_pasado=H_PASADO, n_fourier=None,
                              lambda_fina=None, n_rayos=None):
    """Fija los hiperparámetros de REPRESENTACIÓN (globales del módulo) y recalcula
    DIM_ENTRADA. Como las features se reconstruyen desde el CSV en cada
    entrenamiento, basta llamar aquí antes de construir las muestras para barrer
    estos valores SIN regenerar datos. Devuelve la nueva DIM_ENTRADA."""
    global N_VECINOS, HORIZONTE_VECINO, H_PASADO, DIM_ENTRADA, N_FOURIER
    global LAMBDA_FINA, N_RAYOS
    N_VECINOS = int(n_vecinos)
    HORIZONTE_VECINO = int(horizonte)
    H_PASADO = int(h_pasado)
    if n_fourier is not None:
        N_FOURIER = int(n_fourier)
    if lambda_fina is not None:
        LAMBDA_FINA = float(lambda_fina)
    if n_rayos is not None:
        N_RAYOS = int(n_rayos)
    DIM_ENTRADA = (DIM_EGO + DIM_META + 2 * H_PASADO
                   + DIM_VECINO * N_VECINOS + DIM_CTX
                   + DIM_FOURIER * N_FOURIER + N_RAYOS)
    return DIM_ENTRADA


def crear_red(dim_entrada=DIM_ENTRADA, oculto=512, n_pred=N_PRED,
              n_capas=3, dropout=0.0, activacion="relu", normalizacion="no"):
    """MLP: dim_entrada → (oculto)×n_capas → 2·n_pred (a y giro normalizados).
    'n_capas' es el nº de capas OCULTAS, 'dropout' la prob. de apagado tras cada
    activación, 'activacion' la no linealidad (relu/gelu/silu) y 'normalizacion'
    añade LayerNorm tras cada capa lineal ("no" o "layernorm")."""
    import torch.nn as nn
    actos = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}
    Act = actos.get(activacion, nn.ReLU)
    capas, prev = [], dim_entrada
    for _ in range(max(1, n_capas)):
        capas.append(nn.Linear(prev, oculto))
        if normalizacion == "layernorm":
            capas.append(nn.LayerNorm(oculto))
        capas.append(Act())
        if dropout > 0:
            capas.append(nn.Dropout(dropout))
        prev = oculto
    capas.append(nn.Linear(prev, 2 * n_pred))
    return nn.Sequential(*capas)


class Politica:
    """Red entrenada + normalización de entradas, lista para inferencia."""

    def __init__(self, red, media, escala, device):
        self.red = red
        self.media = media
        self.escala = escala
        self.device = device

    @classmethod
    def cargar(cls, ruta=MODELO_PT):
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        punto = torch.load(ruta, map_location=device, weights_only=False)
        cfg = punto["config"]
        # Reconfigura la REPRESENTACIÓN a la del modelo guardado (por si se entrenó
        # con otro nº de vecinos/horizonte/historia) ANTES de comprobar la dim.
        configurar_representacion(cfg.get("n_vecinos", N_VECINOS),
                                  cfg.get("horizonte", HORIZONTE_VECINO),
                                  cfg.get("h_pasado", H_PASADO),
                                  cfg.get("n_fourier", 0),
                                  cfg.get("lambda_fina", LARGO_REF / 32.0),
                                  cfg.get("n_rayos", 0))
        if cfg["dim_entrada"] != DIM_ENTRADA or cfg["n_pred"] != N_PRED:
            raise ValueError(
                f"El modelo de {ruta} se entrenó con otra definición de entrada "
                f"({cfg}); reentrena con la versión actual de politica.py.")
        red = crear_red(cfg["dim_entrada"], cfg["oculto"], cfg["n_pred"],
                        cfg.get("n_capas", 3), cfg.get("dropout", 0.0),
                        cfg.get("activacion", "relu"),
                        cfg.get("normalizacion", "no"))
        red.load_state_dict(punto["state_dict"])
        red.to(device).eval()
        media = torch.tensor(punto["media"], dtype=torch.float32, device=device)
        escala = torch.tensor(punto["escala"], dtype=torch.float32, device=device)
        return cls(red, media, escala, device)

    def predecir_lote(self, obs):
        """obs (B, DIM_ENTRADA) → (B, N_PRED, 2) en unidades NORMALIZADAS."""
        import torch
        with torch.no_grad():
            x = torch.as_tensor(np.asarray(obs, dtype=np.float32),
                                device=self.device)
            x = (x - self.media) / self.escala
            y = self.red(x)
        return y.cpu().numpy().reshape(len(obs), N_PRED, 2)


# --------------------------------------------------------------------------- #
# Despliegue en bucle cerrado sobre una flota (mismo modelo de bicicleta)
# --------------------------------------------------------------------------- #
# Segundos SIMULADOS que se le dan a una flota para llegar. La ruta más larga
# del dataset dura 107 s, así que esto deja margen de sobra para una red más
# lenta que el planificador clásico sin dar por fallido a quien sí habría
# llegado. No puede quitarse: una red que conduzca en círculos no terminaría
# nunca. En cuanto llegan todos, la simulación sale sola.
T_MAX = 300.0


def rollout_multiflota(flotas, politica, t_max=T_MAX, opts=None):
    """Simula VARIAS flotas independientes a la vez, agrupando en UNA sola consulta
    a la red todos los vehículos activos de todas ellas. Cada vehículo recibe
    exactamente las mismas entradas que en rollout_flota —los vecinos siguen siendo
    solo los de su propia flota—; lo único que cambia es que las consultas se
    agrupan, lo que evita miles de llamadas diminutas a la GPU. Escribe veh.traj y
    veh.mision_ok; devuelve el nº de llegados por flota."""
    if opts is None:
        opts = ["secuencial"] * len(flotas)
    est, planes = [], []
    for vehiculos in flotas:
        e = []
        for veh in vehiculos:
            x, y, th = veh.inicio
            e.append({"x": x, "y": y, "th": th, "v": veh.v_inicial,
                      "pasado": [], "traj": [(x, y, th)], "llegado": False})
        est.append(e)
        planes.append([None] * len(vehiculos))

    pasos = int(round(t_max / DT))
    for paso in range(pasos):
        activos = [(f, i) for f, vehs in enumerate(flotas)
                   for i in range(len(vehs)) if not est[f][i]["llegado"]]
        if not activos:
            break
        if paso % N_PRED == 0:
            obs = []
            for f, i in activos:
                vehs, ef = flotas[f], est[f]
                otros = [{"x": ef[j]["x"], "y": ef[j]["y"], "th": ef[j]["th"],
                          "v": ef[j]["v"], "largo": vehs[j].length,
                          "ancho": vehs[j].width, "v_max": vehs[j].v_max,
                          "grupo": vehs[j].grupo,
                          "prioridad": vehs[j].prioridad}
                         for j in range(len(vehs)) if j != i]
                obs.append(vector_entrada(
                    vehiculo_a_ego(vehs[i], ef[i]["x"], ef[i]["y"], ef[i]["th"],
                                   ef[i]["v"], ef[i]["pasado"], opt=opts[f],
                                   n_veh=len(vehs)), otros))
            pred = politica.predecir_lote(obs)
            for k, (f, i) in enumerate(activos):
                planes[f][i] = pred[k]

        for f, i in activos:
            veh, e = flotas[f][i], est[f][i]
            a_n, g_n = planes[f][i][paso % N_PRED]
            a = float(np.clip(a_n, -1.0, 1.0)) * veh.a_max
            giro = float(np.clip(g_n, -1.0, 1.0)) * veh.delta_max
            x, y, th, v = e["x"], e["y"], e["th"], e["v"]
            e["x"] = x + v * DT * math.cos(th)
            e["y"] = y + v * DT * math.sin(th)
            e["th"] = ang_norm(th + v * DT * math.tan(giro) / veh.wheelbase)
            e["v"] = max(-veh.v_max, min(veh.v_max, v + a * DT))
            e["pasado"].append((a, giro))
            if len(e["pasado"]) > H_PASADO:
                e["pasado"].pop(0)
            mx, my = veh.meta
            if (math.hypot(e["x"] - mx, e["y"] - my) < DIST_LLEGADA
                    and abs(e["v"]) < V_LLEGADA):
                e["llegado"] = True
                e["v"] = 0.0
            e["traj"].append((e["x"], e["y"], e["th"]))

    llegados = []
    for vehs, ef in zip(flotas, est):
        n_ok = 0
        for veh, e in zip(vehs, ef):
            veh.traj = e["traj"]
            veh.dt_plan = DT
            veh.mision_ok = e["llegado"]
            n_ok += int(e["llegado"])
        llegados.append(n_ok)
    return llegados


def rollout_flota(vehiculos, politica, t_max=T_MAX, opt="secuencial"):
    """Simula la flota entera con la red: cada N_PRED pasos consulta la red (una
    pasada por vehículo activo, en lote) y aplica los controles predichos con el
    modelo de bicicleta. Escribe veh.traj y veh.mision_ok; devuelve el nº de
    vehículos que llegan (distancia y velocidad bajo umbral)."""
    n = len(vehiculos)
    est = []                        # estado vivo por vehículo
    for veh in vehiculos:
        x, y, th = veh.inicio
        est.append({"x": x, "y": y, "th": th, "v": veh.v_inicial,
                    "pasado": [], "traj": [(x, y, th)], "llegado": False})
    planes = [None] * n

    pasos = int(round(t_max / DT))
    for paso in range(pasos):
        activos = [i for i in range(n) if not est[i]["llegado"]]
        if not activos:
            break
        if paso % N_PRED == 0:
            obs = []
            for i in activos:
                e = est[i]
                otros = [{"x": est[j]["x"], "y": est[j]["y"],
                          "th": est[j]["th"], "v": est[j]["v"],
                          "largo": vehiculos[j].length,
                          "ancho": vehiculos[j].width,
                          "v_max": vehiculos[j].v_max,
                          "grupo": vehiculos[j].grupo,
                          "prioridad": vehiculos[j].prioridad}
                         for j in range(n) if j != i]
                obs.append(vector_entrada(
                    vehiculo_a_ego(vehiculos[i], e["x"], e["y"], e["th"],
                                   e["v"], e["pasado"], opt=opt, n_veh=n),
                    otros))
            pred = politica.predecir_lote(obs)
            for k, i in enumerate(activos):
                planes[i] = pred[k]

        for i in activos:
            veh, e = vehiculos[i], est[i]
            a_n, g_n = planes[i][paso % N_PRED]
            a = float(np.clip(a_n, -1.0, 1.0)) * veh.a_max
            giro = float(np.clip(g_n, -1.0, 1.0)) * veh.delta_max
            x, y, th, v = e["x"], e["y"], e["th"], e["v"]
            # Integración idéntica a la reconstrucción del dataset: la posición
            # y el rumbo avanzan con la v y θ actuales; después se actualiza v.
            e["x"] = x + v * DT * math.cos(th)
            e["y"] = y + v * DT * math.sin(th)
            e["th"] = ang_norm(th + v * DT * math.tan(giro) / veh.wheelbase)
            e["v"] = max(-veh.v_max, min(veh.v_max, v + a * DT))
            e["pasado"].append((a, giro))
            if len(e["pasado"]) > H_PASADO:
                e["pasado"].pop(0)
            mx, my = veh.meta
            if (math.hypot(e["x"] - mx, e["y"] - my) < DIST_LLEGADA
                    and abs(e["v"]) < V_LLEGADA):
                e["llegado"] = True
                e["v"] = 0.0
            e["traj"].append((e["x"], e["y"], e["th"]))

    llegados = 0
    for veh, e in zip(vehiculos, est):
        veh.traj = e["traj"]
        veh.dt_plan = DT
        veh.mision_ok = e["llegado"]
        llegados += int(e["llegado"])
    return llegados
