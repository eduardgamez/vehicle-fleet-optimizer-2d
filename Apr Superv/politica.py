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
  · los N_VECINOS vehículos más cercanos: pose y velocidad relativas, tamaño y
    si tienen mayor o menor preferencia (grupo, prioridad).
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

from nucleo import DT, W, H, ang_norm

# ----------------------------- hiperparámetros ----------------------------- #
N_PRED = 10        # pasos futuros (a, giro) que predice la red de una vez
H_PASADO = 10      # controles pasados que ve como entrada
N_VECINOS = 5      # vehículos vecinos más cercanos incluidos en la entrada

DIM_EGO = 10       # x, y, sinθ, cosθ, v, largo, ancho, v_max, a_max, giro_max
DIM_META = 6       # meta en ejes propios (dx, dy), dist, sin/cos Δθ_llegada, libre
DIM_VECINO = 10    # dx, dy, dist, sinΔθ, cosΔθ, v, largo, ancho, pref, presente
DIM_ENTRADA = DIM_EGO + DIM_META + 2 * H_PASADO + DIM_VECINO * N_VECINOS

MODELO_PT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "modelos", "politica.pt")

# Criterio de LLEGADA durante el despliegue (coherente con la prioridad del
# proyecto: llegar siempre y rápido pesa más que clavar el ángulo exacto).
DIST_LLEGADA = 0.40    # m al centro de la meta
V_LLEGADA = 0.25       # |v| máxima para considerarse aparcado


# --------------------------------------------------------------------------- #
# Vector de entrada (idéntico en entrenamiento y despliegue)
# --------------------------------------------------------------------------- #
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

    # Vecinos más cercanos (con padding y bandera de presencia).
    base = DIM_EGO + DIM_META + 2 * H_PASADO
    clave_ego = (ego.get("grupo", 1), ego.get("prioridad", 1))
    vecinos = sorted(otros, key=lambda o: (o["x"] - x) ** 2 + (o["y"] - y) ** 2)
    for j, o in enumerate(vecinos[:N_VECINOS]):
        dx, dy = o["x"] - x, o["y"] - y
        dth = ang_norm(o["th"] - th)
        clave_o = (o.get("grupo", 1), o.get("prioridad", 1))
        # Menor (grupo, prioridad) → planifica antes → tiene preferencia.
        pref = 1.0 if clave_o < clave_ego else (-1.0 if clave_o > clave_ego
                                                else 0.0)
        k = base + DIM_VECINO * j
        out[k:k + DIM_VECINO] = (dx * c + dy * s, -dx * s + dy * c,
                                 math.hypot(dx, dy), math.sin(dth),
                                 math.cos(dth), o["v"], o["largo"], o["ancho"],
                                 pref, 1.0)
    return out


def vehiculo_a_ego(veh, x, y, th, v, pasado):
    """dict 'ego' de vector_entrada a partir de un Vehiculo y su estado actual."""
    return {"x": x, "y": y, "th": th, "v": v,
            "largo": veh.length, "ancho": veh.width,
            "v_max": veh.v_max, "a_max": veh.a_max, "giro_max": veh.delta_max,
            "grupo": veh.grupo, "prioridad": veh.prioridad,
            "meta": veh.meta, "meta_th": veh.meta_th, "pasado": pasado}


# --------------------------------------------------------------------------- #
# Red (torch solo se importa aquí dentro)
# --------------------------------------------------------------------------- #
def crear_red(dim_entrada=DIM_ENTRADA, oculto=512, n_pred=N_PRED):
    """MLP sencillo: DIM_ENTRADA → oculto×3 → 2·N_PRED (a y giro normalizados)."""
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(dim_entrada, oculto), nn.ReLU(),
        nn.Linear(oculto, oculto), nn.ReLU(),
        nn.Linear(oculto, oculto), nn.ReLU(),
        nn.Linear(oculto, 2 * n_pred),
    )


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
        if cfg["dim_entrada"] != DIM_ENTRADA or cfg["n_pred"] != N_PRED:
            raise ValueError(
                f"El modelo de {ruta} se entrenó con otra definición de entrada "
                f"({cfg}); reentrena con la versión actual de politica.py.")
        red = crear_red(cfg["dim_entrada"], cfg["oculto"], cfg["n_pred"])
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
def rollout_flota(vehiculos, politica, t_max=180.0):
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
                          "grupo": vehiculos[j].grupo,
                          "prioridad": vehiculos[j].prioridad}
                         for j in range(n) if j != i]
                obs.append(vector_entrada(
                    vehiculo_a_ego(vehiculos[i], e["x"], e["y"], e["th"],
                                   e["v"], e["pasado"]), otros))
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
