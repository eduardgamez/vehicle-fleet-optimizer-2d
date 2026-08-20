#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FASE 2 (GPU) · Barrido de hiperparámetros a gran escala.

Mismo espacio de búsqueda, misma red, misma pérdida y misma nota de rollout que
el pipeline local. Lo que cambia es cómo se paga cada configuración:

  1. Las muestras se cargan UNA vez desde el superset binario y se quedan en la
     memoria de la GPU. Cada configuración obtiene su representación recortando
     columnas (ver `vectorizado.py`), no releyendo los CSV: en el pipeline local
     cambiar de nº de vecinos o de historia costaba reconstruir el dataset
     entero en Python.
  2. La nota de rollout se calcula con el simulador vectorizado, no con bucles
     de Python paso a paso.
  3. La nota se mide sobre ESCENARIOS NUEVOS (conjunto de selección), no sobre
     los escenarios con los que se ha entrenado.
  4. Criba temprana opcional: una configuración que a mitad de camino va peor
     que la mayoría de las ya vistas se abandona en vez de agotar sus épocas.
     No toca el espacio de búsqueda, solo deja de gastar en lo que ya se sabe
     que no gana.
  5. El trabajo se reparte entre tareas y cada una lleva su registro, con la
     misma reanudación que ya tenía el barrido local.

Uso:
    python entrenar_nube.py --n-configs 400
    python entrenar_nube.py --espacio espacio_fino.json --exhaustivo
"""

import argparse
import csv
import itertools
import json
import math
import os
import random as _rnd
import time

import numpy as np

import comun
from comun import MODELOS_DIR, MUESTRAS_DIR, asegurar
import escenarios as esc
import politica as pol
import preparar_datos as prep
import vectorizado as vec

import entrenar as ent
from entrenar import CAMPOS_EXTRA, _puntuar_vehiculo, crear_red

# Espacio de búsqueda: el del pipeline local con el eje de VECINOS reajustado al
# tamaño de flota que se genera. El tope sale del propio problema, no de la red:
# con flotas de hasta 8 vehículos nadie puede ver más de 7 vecinos, y valores más
# altos serían bloques de entrada siempre a cero. El resto de ejes queda igual.
ESPACIO = dict(ent.ESPACIO)
ESPACIO["n_vecinos"] = [3, 4, 5, 7]

# MEZCLA de tipos de escenario: cuánta presencia tiene cada tipo en el
# entrenamiento. Es un HIPERPARÁMETRO más, no una constante: cuál conviene
# depende de la red y no se sabe de antemano, así que el barrido lo prueba.
#
# Cada valor es un multiplicador sobre el nº NATURAL de muestras de ese tipo:
# 1.0 lo deja como está, 0.4 se queda con el 40 %, 3.0 lo repite tres veces.
# Lo que no se nombra vale 1.0. 'equilibrado' es especial: iguala todos los
# tipos al más numeroso.
MEZCLAS = {
    # Tal cual salió del generador (80/10/10 más las reutilizadas).
    "natural": {},
    # El tope del pipeline local: las reutilizadas, que son gratis pero llegan
    # sesgadas hacia jerarquías planas, se quedan en el 40 %.
    "cap_rec": {"secuencial_rec": 0.4},
    # Los modos raros pesan más, que es lo que hacía `pesos_de_modos` con la
    # pérdida ponderada; aquí se hace con la proporción de datos.
    "raros_x2": {"global": 2.0, "prioridades": 2.0, "secuencial_rec": 0.5},
    "raros_x4": {"global": 4.0, "prioridades": 4.0, "secuencial_rec": 0.5},
    # Todos los tipos con el mismo nº de muestras.
    "equilibrado": "equilibrado",
}
ESPACIO["mezcla"] = ["natural", "cap_rec", "raros_x2", "raros_x4"]

# POSICIÓN EN SENOS Y COSENOS: nº de longitudes de onda (ver politica.N_FOURIER).
# Entra como eje y no activado a secas porque es una HIPÓTESIS sobre por qué se
# choca tanto —que la red no distingue bien un palmo a partir de la coordenada
# cruda, y por eso no aprende dónde está el borde de un muro—. Con 0 en la lista,
# el propio barrido dice si sirve: si las de 0 chocan lo mismo que las de 8, el
# problema estaba en otro sitio. Las ondas van de la más fina hacia arriba, así
# que 6 cubren de 1/32 a 1 vehículo (el detalle local, que es lo que se pone a
# prueba) y 12 llegan hasta 64 vehículos, o sea mucho más que el mapa entero.
ESPACIO["n_fourier"] = [0, 6, 12]

# RAYOS al obstáculo (ver rayos.py): en cuántas direcciones mira el vehículo.
# Es la primera idea que añade INFORMACIÓN en vez de recodificar la que ya hay:
# la distancia libre es lo que decide si chocas, y hasta ahora había que
# deducirla de (x, y) más el mapa aprendido en los pesos.
#
# El 0 sigue en la lista para la fase 2, pero la tanda de fase 1 que mide esto
# se lanza SIN él (--espacio con solo 8, 12 y 18): las 241 configuraciones ya
# evaluadas son el grupo de control con 0 rayos, así que repetirlo sería gastar
# pruebas en algo que ya está medido.
ESPACIO["n_rayos"] = [0, 8, 12, 18]

CAMPOS = ["nota_seleccion" if c == "nota_rollout" else c for c in CAMPOS_EXTRA]
CAMPOS += ["epocas_hechas", "pct_choque", "pct_llegada", "seg_choque"]


# --------------------------------------------------------------------------- #
# Carga del superset a la GPU
# --------------------------------------------------------------------------- #
def _opcional(carpeta, lote, sufijo, n, relleno):
    """Array por fila que puede no existir (datasets preparados antes de que se
    guardaran el tipo de escenario y el tamaño de flota). Si falta, se rellena
    con un valor neutro: el entrenamiento funciona igual, pero sin poder elegir
    la mezcla de tipos."""
    ruta = os.path.join(carpeta, f"{lote}_{sufijo}.npy")
    if os.path.exists(ruta):
        return np.load(ruta)
    return np.full(n, relleno, dtype=np.int8)


def _seleccion(modo, nveh, val, fraccion, frac_modos, etiquetas, semilla):
    """Índices de las filas que se usan, estratificando por (tipo, tamaño de
    flota) y dejando la validación entera. Los subconjuntos ANIDAN: dentro de
    cada estrato se baraja una sola vez (semilla fija) y se toman las primeras,
    así que una fracción mayor solo AÑADE filas."""
    if fraccion >= 1.0 and not frac_modos:
        return np.arange(len(val))
    rng = np.random.default_rng(semilla)
    trozos = [np.flatnonzero(val)]
    # Orden de estratos fijo (por código y tamaño) para que el barajado sea el
    # mismo llamada tras llamada y el anidamiento se cumpla.
    for code in sorted(np.unique(modo).tolist()):
        etiq = etiquetas[code] if 0 <= code < len(etiquetas) else "?"
        f = min(1.0, max(0.0, float(frac_modos.get(etiq, fraccion))))
        for nv in sorted(np.unique(nveh).tolist()):
            idx = np.flatnonzero((~val) & (modo == code) & (nveh == nv))
            if len(idx) == 0:
                continue
            perm = rng.permutation(len(idx))
            n = 0 if f <= 0 else max(1, int(math.ceil(f * len(idx))))
            trozos.append(idx[perm[:n]])
    return np.sort(np.concatenate(trozos))


def resolver_mezcla(nombre, cuentas):
    """Multiplicador por tipo de escenario. 'cuentas' es {etiqueta: nº de filas
    de entrenamiento} y hace falta para 'equilibrado', que iguala todos al más
    numeroso."""
    m = MEZCLAS.get(nombre, {})
    if m == "equilibrado":
        tope = max(cuentas.values()) if cuentas else 1
        return {e: (tope / c if c else 1.0) for e, c in cuentas.items()}
    return dict(m)


class Datos:
    """Superset de muestras residente en la GPU, con las utilidades para sacar
    de él la vista de cualquier configuración.

    'fraccion' es la parte del dataset que se usa. La idea es escalonarla: la
    búsqueda amplia solo tiene que ORDENAR configuraciones entre sí, y para eso
    sobra con una parte; la rejilla fina afina de verdad y merece bastante más;
    y el modelo final se entrena con todo. Se submuestrea por FILAS y no por
    escenarios, para que con poca fracción se sigan viendo todos los escenarios
    (menos instantes de cada uno) en vez de unos pocos escenarios enteros:
    interesa más la variedad de situaciones que el detalle de cada una.

    El recorte es ESTRATIFICADO por (tipo de escenario, tamaño de flota): dentro
    de cada combinación se conserva la misma proporción, así que usar el 35 % no
    cambia la mezcla del dataset, solo su tamaño. Y es ANIDADO: las filas del
    35 % están dentro de las del 70 %, y estas dentro del 100 %, de modo que la
    rejilla fina amplía lo que vio la búsqueda amplia en vez de cambiar de datos.
    Las filas de VALIDACIÓN se conservan enteras, para que el val_mse siga siendo
    comparable entre las tres pasadas."""

    def __init__(self, carpeta, device, fraccion=1.0, max_muestras=None,
                 semilla=0, en_cpu=False, frac_modos=None):
        import torch
        with open(os.path.join(carpeta, "meta.json"), encoding="utf-8") as f:
            self.meta = json.load(f)
        self.etiquetas = self.meta.get("etiquetas", list(prep.ETIQUETAS))
        Xs, TCs, Ys, Vs, Ms, Ns = [], [], [], [], [], []
        for lote in self.meta["lotes"]:
            Xs.append(np.load(os.path.join(carpeta, f"{lote}_X.npy")))
            TCs.append(np.load(os.path.join(carpeta, f"{lote}_TC.npy")))
            Ys.append(np.load(os.path.join(carpeta, f"{lote}_Y.npy")))
            Vs.append(np.load(os.path.join(carpeta, f"{lote}_val.npy")))
            Ms.append(_opcional(carpeta, lote, "modo", len(Vs[-1]), -1))
            Ns.append(_opcional(carpeta, lote, "nveh", len(Vs[-1]), 0))
        X = np.concatenate(Xs)
        TC = np.concatenate(TCs)
        Y = np.concatenate(Ys)
        val = np.concatenate(Vs)
        modo = np.concatenate(Ms)
        nveh = np.concatenate(Ns)

        sel = _seleccion(modo, nveh, val, fraccion, frac_modos or {},
                         self.etiquetas, semilla)
        if max_muestras and len(sel) > max_muestras:
            # Tope duro por memoria de GPU: se recorta manteniendo el reparto
            # (la selección ya viene ordenada por estratos entrelazados).
            sel = np.sort(np.random.default_rng(semilla).choice(
                sel, max_muestras, replace=False))
        if len(sel) < len(X):
            X, TC, Y, val = X[sel], TC[sel], Y[sel], val[sel]
            modo, nveh = modo[sel], nveh[sel]
        self.modo = modo

        self.n_vec_max = self.meta["n_vec_max"]
        self.h_max = self.meta["h_max"]
        # Datasets preparados antes de que existiera el bloque de Fourier no lo
        # traen: con 0, el eje n_fourier se queda sin efecto en vez de reventar.
        self.n_f_max = self.meta.get("n_f_max", 0)
        self.con_rayos = bool(self.meta.get("con_rayos", False))
        self.device = device
        # Con el dataset completo puede no caber en la GPU. En ese caso se queda
        # en la memoria del ordenador y cada lote se sube al vuelo: es más lento
        # por época, pero permite el entrenamiento final sin alquilar una GPU
        # enorme.
        self.dev_datos = torch.device("cpu") if en_cpu else device
        self.en_cpu = en_cpu
        self.X = torch.from_numpy(X).to(self.dev_datos)
        self.TC = torch.from_numpy(TC).to(self.dev_datos)
        self.Y = torch.from_numpy(Y).to(self.dev_datos)
        self.idx_tr_np = np.flatnonzero(~val)
        self.idx_tr = torch.from_numpy(self.idx_tr_np).to(self.dev_datos)
        self.idx_va = torch.from_numpy(np.flatnonzero(val)).to(self.dev_datos)
        if len(self.idx_va) == 0:               # dataset diminuto
            self.idx_va = self.idx_tr[-1:]
        self.cols_cache = {}
        self.mezcla_cache = {}

    def cuentas_modo(self):
        """{etiqueta: nº de filas de entrenamiento} de cada tipo de escenario."""
        out = {}
        m = self.modo[self.idx_tr_np]
        for code, n in zip(*np.unique(m, return_counts=True)):
            if 0 <= code < len(self.etiquetas):
                out[self.etiquetas[code]] = int(n)
        return out

    def indices(self, mezcla="natural"):
        """Índices de entrenamiento con la MEZCLA de tipos pedida. Un tipo con
        multiplicador < 1 se recorta (se queda con esa parte, siempre las mismas
        filas) y con > 1 se repite, así que la red lo ve más veces por época. Si
        el dataset no trae la etiqueta por fila, devuelve el entrenamiento
        entero."""
        import torch
        if mezcla in self.mezcla_cache:
            return self.mezcla_cache[mezcla]
        cuentas = self.cuentas_modo()
        mult = resolver_mezcla(mezcla, cuentas)
        if not mult or not cuentas:
            self.mezcla_cache[mezcla] = self.idx_tr
            return self.idx_tr
        m = self.modo[self.idx_tr_np]
        rng = np.random.default_rng(0)
        trozos = []
        for code in sorted(np.unique(m).tolist()):
            idx = self.idx_tr_np[m == code]
            etiq = self.etiquetas[code] if 0 <= code < len(self.etiquetas) else "?"
            f = float(mult.get(etiq, 1.0))
            n = int(round(f * len(idx)))
            if n <= 0:
                continue
            if n <= len(idx):
                trozos.append(idx[rng.permutation(len(idx))[:n]])
            else:                       # repetir: copias enteras + un resto
                copias = [idx] * (n // len(idx))
                resto = n % len(idx)
                if resto:
                    copias.append(idx[rng.permutation(len(idx))[:resto]])
                trozos.append(np.concatenate(copias))
        sel = np.sort(np.concatenate(trozos)) if trozos else self.idx_tr_np
        t = torch.from_numpy(sel).to(self.dev_datos)
        self.mezcla_cache[mezcla] = t
        return t

    def pesos(self, mezcla="natural", enfasis=1.0):
        """Peso de cada fila de `indices(mezcla)` en la pérdida, SIEMPRE activo:
        los modos raros (global, prioridades) pesan más que el secuencial y las
        reutilizadas pesan la mitad (mismos niveles que el pipeline local, ver
        `entrenar.pesos_de_modos`).

        Se calcula sobre la mezcla YA aplicada, y como el peso es el inverso de
        la frecuencia, las dos palancas no se suman por partida doble: si la
        mezcla ya ha subido la presencia de un tipo, su peso baja solo hasta
        dejar la sensibilidad en el mismo sitio."""
        import torch
        clave = ("w", mezcla, enfasis)
        if clave in self.mezcla_cache:
            return self.mezcla_cache[clave]
        idx = self.indices(mezcla)
        codes = self.modo[idx.cpu().numpy()]
        etiq = np.array([self.etiquetas[c] if 0 <= c < len(self.etiquetas)
                         else "secuencial" for c in codes])
        w = torch.from_numpy(ent.pesos_de_modos(etiq, enfasis)).to(self.dev_datos)
        self.mezcla_cache[clave] = w
        return w

    def lote(self, V, idx):
        """Un lote de (entradas, objetivos) ya en la GPU."""
        x, y = V[idx], self.Y[idx]
        if self.en_cpu:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
        return x, y

    def _media_escala(self, V):
        """Media y desviación de las filas de ENTRENAMIENTO, por trozos: hacer
        V[idx_tr] de golpe duplicaría el dataset en la GPU. Desviación de
        población (mismo criterio que el np.std del pipeline local)."""
        import torch
        suma = torch.zeros(V.shape[1], dtype=torch.float64, device=V.device)
        suma2 = torch.zeros_like(suma)
        for i in range(0, len(self.idx_tr), 1_000_000):
            b = V[self.idx_tr[i:i + 1_000_000]].double()
            suma += b.sum(0)
            suma2 += (b * b).sum(0)
        n = len(self.idx_tr)
        media = suma / n
        escala = (suma2 / n - media * media).clamp_min(0).sqrt().clamp_min(1e-6)
        return media.float(), escala.float()

    def vista(self, n_vec, horizonte, h_pasado, n_fourier=0, n_rayos=0):
        """(tensor (M, dim) normalizado, media, escala) de una configuración."""
        import torch
        n_fourier = min(int(n_fourier), self.n_f_max)
        n_rayos = int(n_rayos) if self.con_rayos else 0
        clave = (n_vec, h_pasado, n_fourier, n_rayos)
        if clave not in self.cols_cache:
            cols = vec.columnas_vista(n_vec, h_pasado, self.n_vec_max,
                                      self.h_max, n_fourier, self.n_f_max,
                                      n_rayos, self.con_rayos)
            self.cols_cache[clave] = torch.from_numpy(cols).to(self.dev_datos)
        V = torch.index_select(self.X, 1, self.cols_cache[clave])
        # Horizonte: los bloques de vecino cuyo tiempo de cierre lo supera se
        # anulan (con el orden por urgencia son siempre los últimos ocupados).
        # Bloque a bloque, para no materializar una máscara del tamaño del
        # dataset.
        ini = vec.DIM_EGO + vec.DIM_META + 2 * h_pasado
        vivos = (self.TC[:, :n_vec] <= horizonte * float(vec.DT)).to(V.dtype)
        for j in range(n_vec):
            a = ini + j * vec.DIM_VECINO
            V[:, a:a + vec.DIM_VECINO] *= vivos[:, j:j + 1]
        media, escala = self._media_escala(V)
        V.sub_(media).div_(escala)
        return V, media.to(self.device), escala.to(self.device)


# --------------------------------------------------------------------------- #
# Entrenamiento de una configuración
# --------------------------------------------------------------------------- #
# A partir de esta anchura compensa calcular en PRECISIÓN CORTA (bf16): las
# unidades de matrices de la tarjeta solo se activan con números de 16 bits y
# entonces van al doble. Por debajo NO compensa: la red es tan pequeña que el
# trámite de convertir los números cuesta más que lo que se ahorra. Medido en la
# 5060 Ti: 1024×4 pasa de 3,20 a 1,58 ms por paso, pero 512×3 empeora de 0,86 a
# 1,30. Las cuentas del error y del ajuste siguen en precisión normal, así que la
# calidad no cambia.
ANCHURA_BF16 = 1024


def entrenar_config(V, datos, c, device, criba=None, enfasis=1.0, aviso=None):
    """Entrena una red y devuelve (mejor_state_dict, mejor_val, épocas hechas).

    'aviso' es una función opcional que se llama al final de cada época con
    (época, mejor error de validación, la red, el mejor estado guardado). Si
    devuelve True, se corta ahí. Es el enganche que usa la búsqueda con Optuna
    para abandonar pronto las configuraciones que van mal (ver
    buscar_optuna.py); la criba por mediana de aquí arriba hace lo mismo pero
    sin coordinarse con nadie."""
    import copy
    import torch

    epocas = int(c["epocas"])
    lote = int(c["lote"])
    red = crear_red(V.shape[1], c["oculto"], pol.N_PRED, c["n_capas"],
                    c["dropout"], c["activacion"],
                    c.get("normalizacion", "no")).to(device)
    if c.get("optimizador", "adamw") == "sgd":
        opt = torch.optim.SGD(red.parameters(),
                              lr=c["lr"] * ent.FACTOR_LR_SGD, momentum=0.9,
                              weight_decay=c.get("weight_decay", 0.01))
    else:
        opt = torch.optim.AdamW(red.parameters(), lr=c["lr"],
                                weight_decay=c.get("weight_decay", 0.01),
                                fused=(device.type == "cuda"))
    plani = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epocas)
    perdida = torch.nn.MSELoss()
    perdida_m = torch.nn.MSELoss(reduction="none")   # para ponderar por muestra

    # La MEZCLA de tipos de escenario es un hiperparámetro: cambia qué filas
    # entran en el entrenamiento (y cuántas veces), no la validación.
    idx_tr = datos.indices(c.get("mezcla", "natural"))
    idx_va = datos.idx_va
    # Peso por tipo de escenario: se aplica SIEMPRE. La validación no se pondera,
    # para que elegir la mejor época siga siendo una cifra honesta.
    w_tr = datos.pesos(c.get("mezcla", "natural"), enfasis)
    # Los hitos de la criba son FRACCIONES del presupuesto de esta configuración,
    # no épocas absolutas: con presupuestos de 40, 80 y 120 épocas y un ritmo de
    # aprendizaje que se estira con el total, la época 20 significa cosas muy
    # distintas en cada una, y comparar ahí penalizaría a las de presupuesto largo.
    hitos = {}
    if criba is not None:
        for f in criba.fracciones:
            hitos[max(1, int(round(f * epocas)))] = f

    # Precisión corta solo donde compensa (ver ANCHURA_BF16). El error y el
    # ajuste de los pesos se siguen calculando en precisión normal.
    corta = (device.type == "cuda" and int(c["oculto"]) >= ANCHURA_BF16)
    ctx = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if corta \
        else torch.enable_grad

    mejor_val, mejor_estado, hechas = float("inf"), None, 0
    for ep in range(1, epocas + 1):
        red.train()
        perm = torch.randperm(len(idx_tr), device=datos.dev_datos)
        for i in range(0, len(perm), lote):
            pos = perm[i:i + lote]
            xb, yb = datos.lote(V, idx_tr[pos])
            wb = w_tr[pos]
            if datos.en_cpu:
                wb = wb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with ctx():
                pred = red(xb)
            err = perdida_m(pred.float(), yb).mean(dim=1)
            ((err * wb).sum() / wb.sum()).backward()
            opt.step()
        plani.step()
        red.eval()
        with torch.no_grad():
            tot = 0.0
            for i in range(0, len(idx_va), 65536):
                xb, yb = datos.lote(V, idx_va[i:i + 65536])
                with ctx():
                    pred = red(xb)
                tot += perdida(pred.float(), yb).item() * len(xb)
            lv = tot / len(idx_va)
        hechas = ep
        # Solo cuenta una época con pérdida FINITA: con tasas altas el
        # entrenamiento puede divergir a NaN, y NaN < x siempre es falso.
        if math.isfinite(lv) and lv < mejor_val:
            mejor_val = lv
            mejor_estado = copy.deepcopy(red.state_dict())
        if ep in hitos and criba.para(hitos[ep], mejor_val):
            break
        # Se le pasan también la red y el mejor estado: quien avisa puede querer
        # PUNTUAR de verdad en ese momento (conducir las flotas), no solo mirar
        # el error de validación. Ver buscar_optuna.py.
        if aviso is not None and aviso(ep, mejor_val, red, mejor_estado):
            break
    if mejor_estado is None:
        mejor_estado = copy.deepcopy(red.state_dict())
    return red, mejor_estado, mejor_val, hechas


_MAPA = None            # obstáculos y tamaño del mundo, se leen una sola vez


def mapa_de_choques():
    global _MAPA
    if _MAPA is None:
        _MAPA = esc.obstaculos()
    return _MAPA


def nota_de(red, estado, media, escala, device, flotas, opts, c):
    """Nota media por vehículo de una red sobre un conjunto de escenarios.

    Se le pasan los obstáculos para que el simulador detecte choques: un
    vehículo que toque a otro o al mapa puntúa cero (ver
    `entrenar._puntuar_vehiculo`)."""
    from politica import Politica
    red.load_state_dict(estado)
    politica = Politica(red.eval(), media, escala, device)
    obst, mundo = mapa_de_choques()
    vec.rollout(flotas, politica, opts=opts, n_vec=c["n_vecinos"],
                horizonte=c["horizonte"], h_pasado=c["h_pasado"],
                obstaculos=obst, mundo=mundo,
                n_fourier=c.get("n_fourier", 0),
                n_rayos=c.get("n_rayos", 0))
    total = sum(_puntuar_vehiculo(v) for f in flotas for v in f)
    n = sum(len(f) for f in flotas)
    # Se guardan aparte cuántos chocan y cuántos llegan. La nota los mezcla en
    # una sola cifra, y para saber si el problema es de capacidad o de otra cosa
    # hace falta verlos por separado.
    nota_de.choques = (sum(1 for f in flotas for v in f
                           if getattr(v, "choque", False)) / n) if n else 0.0
    nota_de.llegadas = (sum(1 for f in flotas for v in f
                            if v.mision_ok) / n) if n else 0.0
    # Segundos MEDIOS tocando algo: con casi todas chocando, esta es la cifra
    # que de verdad separa a unas de otras.
    nota_de.seg_choque = (sum(getattr(v, "seg_choque", 0.0)
                              for f in flotas for v in f) / n) if n else 0.0
    return total / n if n else 0.0


# --------------------------------------------------------------------------- #
# Criba temprana (regla de la mediana)
# --------------------------------------------------------------------------- #
class Criba:
    """Abandona una configuración que, a media carrera, va peor que casi todas
    las anteriores. No cambia el espacio de búsqueda: las mismas configuraciones
    se prueban y se puntúan, solo que a las que ya van mal se les dedican menos
    épocas.

    Los hitos son FRACCIONES del presupuesto de cada configuración (no épocas
    absolutas), y el corte está calibrado del lado SEGURO: con percentil 85 solo
    se abandona el ~15 % peor de lo visto hasta ese momento, así que una red que
    arranque lenta pero vaya a remontar casi seguro sobrevive. Se ahorra menos
    que con un corte agresivo, pero es muy difícil que tire a una buena."""

    def __init__(self, fracciones=(0.5,), percentil=85.0, minimo=20):
        self.fracciones = sorted(fracciones)
        self.percentil = percentil
        self.minimo = minimo
        self.historial = {f: [] for f in self.fracciones}

    def para(self, fraccion, val):
        """True si hay que abandonar. 'fraccion' identifica el hito; 'val' es el
        mejor error de validación que lleva la configuración."""
        previos = self.historial.get(fraccion)
        if previos is None:
            return False
        if not math.isfinite(val):        # ha divergido: no hay nada que salvar
            previos.append(1e9)
            return True
        # Hacen falta bastantes referencias para que el percentil signifique algo.
        corta = (len(previos) >= self.minimo
                 and val > float(np.percentile(previos, self.percentil)))
        previos.append(val)
        return corta


# --------------------------------------------------------------------------- #
# Barrido
# --------------------------------------------------------------------------- #
def _configs(args):
    if args.espacio:
        with open(args.espacio, encoding="utf-8") as f:
            espacio = json.load(f)
    else:
        espacio = ESPACIO
    claves = list(ESPACIO.keys())
    _rnd.seed(args.semilla)

    # Cuántas combinaciones tiene el espacio, SIN construirlas. Hace falta el
    # número para la opción --fraccion, y construir la lista es justo lo que no
    # se puede hacer: con quince ejes el producto cartesiano son 139 millones de
    # tuplas y el proceso se queda sin memoria antes de empezar (pasó: tres
    # trabajadores veinte minutos atascados ahí, con la tarjeta al 1 %).
    total = 1
    for k in claves:
        total *= len(espacio[k])

    if args.exhaustivo:
        # Recorrer el espacio ENTERO solo tiene sentido cuando es pequeño (una
        # rejilla fina alrededor de una ganadora, que es para lo que está).
        if total > 2_000_000:
            raise SystemExit(
                f"--exhaustivo sobre {total:,} combinaciones no cabe en "
                f"memoria. Reduce el espacio con --espacio, o quita "
                f"--exhaustivo para sortear al azar.")
        todas = list(itertools.product(*(espacio[k] for k in claves)))
        _rnd.shuffle(todas)
    else:
        n = (int(round(args.fraccion * total)) if args.fraccion
             else args.n_configs)
        n = max(1, min(n, total))
        # Sorteo SIN construir el producto: se elige un valor de cada eje. Se
        # descartan las repetidas para no gastar dos veces en lo mismo; con un
        # espacio tan grande frente a las pocas que se prueban, las colisiones
        # son rarísimas, pero el tope de intentos evita el bucle infinito si
        # alguien restringe el espacio a menos combinaciones que 'n'.
        vistas, todas, intentos = set(), [], 0
        while len(todas) < n and intentos < 200 * n + 1000:
            intentos += 1
            vals = tuple(_rnd.choice(espacio[k]) for k in claves)
            if vals in vistas:
                continue
            vistas.add(vals)
            todas.append(vals)

    configs = [dict(zip(claves, vals)) for vals in todas]
    return claves, comun.reparto(configs, args.tarea, args.tareas)


def main():
    ap = argparse.ArgumentParser(description="Barrido de hiperparámetros (GPU).")
    ap.add_argument("--muestras", default=MUESTRAS_DIR)
    ap.add_argument("--escenarios", default=esc.CARPETA)
    ap.add_argument("--salida", default=MODELOS_DIR)
    ap.add_argument("--n-configs", dest="n_configs", type=int, default=200)
    ap.add_argument("--fraccion", type=float, default=None)
    ap.add_argument("--exhaustivo", action="store_true")
    ap.add_argument("--espacio", default=None)
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--tarea", type=int, default=None)
    ap.add_argument("--tareas", type=int, default=None)
    ap.add_argument("--fraccion-datos", dest="fraccion_datos", type=float,
                    default=1.0,
                    help="parte del dataset que se usa: 0.35 en la búsqueda "
                         "amplia, 0.65 en la rejilla fina, 1.0 en el modelo "
                         "final (def. 1.0)")
    ap.add_argument("--enfasis", type=float, default=1.0,
                    help="fuerza del peso por tipo de escenario en la pérdida: "
                         "0 = sin ponderar, 1 = influencia equilibrada entre "
                         "tipos (def.), >1 exagera los raros")
    ap.add_argument("--frac-modos", dest="frac_modos", default=None,
                    help="fracción distinta por tipo de escenario al recortar "
                         "el dataset, p. ej. 'global=1,prioridades=1'. Lo que "
                         "no se nombre usa --fraccion-datos. Ojo: esto decide "
                         "qué datos se CARGAN; para probar mezclas dentro del "
                         "barrido está el eje 'mezcla'")
    ap.add_argument("--max-muestras", dest="max_muestras", type=int,
                    default=12_000_000,
                    help="tope duro de muestras, por la memoria de la GPU "
                         "(def. 12 M; con --en-cpu se puede subir)")
    ap.add_argument("--frac-vram", dest="frac_vram", type=float, default=1.0,
                    help="parte de la memoria de la tarjeta que puede usar ESTE "
                         "proceso (0.45 = 45 %%). Con varios barridos a la vez "
                         "es lo que convierte un 'no cabe' en un error limpio "
                         "que se puede saltar, en lugar de dejar sin memoria a "
                         "los demás y tumbar el driver")
    ap.add_argument("--en-cpu", dest="en_cpu", action="store_true",
                    help="deja los datos en la memoria del ordenador y sube "
                         "cada lote al vuelo: más lento, pero permite usar "
                         "dataset completo sin una GPU enorme")
    ap.add_argument("--n-escenarios", dest="n_escenarios", type=int, default=300,
                    help="escenarios de selección por evaluación (def. 300)")
    ap.add_argument("--criba", default="0.5",
                    help="hitos de la criba como FRACCIÓN del presupuesto de "
                         "épocas de cada configuración, separados por comas "
                         "(def. 0.5 = a media carrera); vacío la apaga")
    ap.add_argument("--percentil", type=float, default=85.0,
                    help="se abandona la configuración cuyo error supere este "
                         "percentil de las ya vistas en el mismo hito. Alto = "
                         "seguro (85 descarta solo el ~15 %% peor); bajo = "
                         "ahorra más pero puede tirar a alguna buena (def. 85)")
    ap.add_argument("--minimo", type=int, default=20,
                    help="configuraciones de referencia necesarias antes de "
                         "empezar a cribar (def. 20)")
    ap.add_argument("--tf32", action="store_true", default=True,
                    help="permite TF32 en las multiplicaciones (por defecto sí)")
    ap.add_argument("--sync-min", dest="sync_min", type=int, default=10,
                    help="cada cuántos minutos se sube el registro al bucket")
    args = ap.parse_args()

    import torch
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and 0 < args.frac_vram < 1.0:
        torch.cuda.set_per_process_memory_fraction(args.frac_vram)
        libre = torch.cuda.get_device_properties(0).total_memory * args.frac_vram
        print(f"[barrido] tope de memoria de este proceso: "
              f"{libre / 2**30:.1f} GiB", flush=True)
    idt = args.tarea if args.tarea is not None else comun.indice_tarea()
    asegurar(args.salida)
    remoto = comun.bucket_env("modelos")
    comun.recuperar(args.salida, remoto)
    comun.arrancar_sincronizacion(args.salida, remoto, args.sync_min)

    t0 = time.perf_counter()
    datos = Datos(args.muestras, device, args.fraccion_datos,
                  args.max_muestras, en_cpu=args.en_cpu,
                  frac_modos=ent.parsear_frac_modos(args.frac_modos))
    flotas, opts = esc.flotas(esc.cargar("seleccion", args.escenarios),
                              limite=args.n_escenarios)
    claves, configs = _configs(args)
    log_path = os.path.join(args.salida, f"barrido_t{idt:03d}.csv")
    hechas = ent._ya_evaluadas(log_path, claves)
    if hechas:
        antes = len(configs)
        configs = [c for c in configs
                   if ent._clave_config(c, claves) not in hechas]
        print(f"[barrido] reanudando: {antes - len(configs)} ya evaluadas, "
              f"faltan {len(configs)}", flush=True)
    if not configs:
        raise SystemExit("[barrido] no queda ninguna configuración por evaluar.")

    n_veh = sum(len(f) for f in flotas)
    ct = datos.cuentas_modo()
    print("[barrido] tipos de escenario en train: "
          + (" · ".join(f"{e} {n:,}" for e, n in sorted(ct.items()))
             if ct else "sin etiquetas (dataset antiguo: mezcla desactivada)"),
          flush=True)
    print(f"[barrido] tarea {idt} · {device} · {len(datos.X):,} muestras "
          f"({len(datos.idx_tr):,} train / {len(datos.idx_va):,} val) · "
          f"{len(flotas)} escenarios de selección ({n_veh} vehículos) · "
          f"{len(configs)} configuraciones · carga {time.perf_counter() - t0:.1f} s",
          flush=True)

    criba = None
    if args.criba.strip():
        criba = Criba([float(f) for f in args.criba.split(",")],
                      args.percentil, args.minimo)

    campos = claves + CAMPOS
    mejor = None
    omitidas = []
    t_ini = time.perf_counter()
    with open(log_path, "a" if hechas else "w", newline="",
              encoding="utf-8") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=campos)
        if not hechas:
            w.writeheader()
        for k, c in enumerate(configs, 1):
            t_c = time.perf_counter()
            try:
                V, media, escala = datos.vista(c["n_vecinos"], c["horizonte"],
                                               c["h_pasado"],
                                               c.get("n_fourier", 0),
                                               c.get("n_rayos", 0))
                red, estado, val, eps = entrenar_config(V, datos, c, device,
                                                        criba, args.enfasis)
                nota = nota_de(red, estado, media, escala, device, flotas,
                               opts, c)
            except torch.cuda.OutOfMemoryError:
                # Una red enorme con un lote enorme puede no caber si hay varios
                # barridos a la vez. Se salta para no tumbar una tanda de horas,
                # pero NO se anota en el registro: así queda pendiente y una
                # ejecución posterior con menos procesos la recoge. Perder una
                # configuración del espacio de búsqueda sí sería un problema.
                torch.cuda.empty_cache()
                omitidas.append(c)
                print(f"[barrido] {k}/{len(configs)} · SIN MEMORIA "
                      f"({c['oculto']}x{c['n_capas']}, lote {c['lote']}): "
                      f"pendiente para otra tanda", flush=True)
                continue
            fila = dict(c)
            fila.update({"dim_entrada": V.shape[1], "n_muestras": len(datos.idx_tr),
                         "val_mse": round(val, 5),
                         "nota_seleccion": round(nota, 4),
                         "epocas_hechas": eps,
                         "pct_choque": round(100 * nota_de.choques, 1),
                         "pct_llegada": round(100 * nota_de.llegadas, 1),
                         "seg_choque": round(nota_de.seg_choque, 2),
                         "segundos": round(time.perf_counter() - t_c, 1)})
            w.writerow(fila)
            fcsv.flush()
            if mejor is None or nota > mejor[0]:
                mejor = (nota, val, dict(c),
                         media.cpu().numpy(), escala.cpu().numpy(),
                         {n: t.detach().cpu() for n, t in estado.items()},
                         V.shape[1])
                guardar_mejor(args.salida, idt, mejor)
            transc = time.perf_counter() - t_ini
            print(f"[barrido] {k}/{len(configs)} · nota {nota:.4f} · "
                  f"val {val:.5f} · {eps} épocas · "
                  f"faltan ~{transc / k * (len(configs) - k) / 60:.0f} min · "
                  f"mejor {mejor[0]:.4f}", flush=True)
            # Suelta la red y la vista antes de la siguiente: con varios
            # barridos a la vez, dejar los restos de una red de 80 M en la
            # tarjeta es lo que hace que a la de al lado no le quepa la suya.
            del red, estado, V
            torch.cuda.empty_cache()

    print(f"[barrido] tarea {idt}: {len(configs) - len(omitidas)} "
          f"configuraciones en {(time.perf_counter() - t_ini) / 60:.1f} min · "
          f"mejor {mejor[0]:.4f}" if mejor else
          f"[barrido] tarea {idt}: sin resultados", flush=True)
    if omitidas:
        print(f"[barrido] {len(omitidas)} quedaron sin memoria y siguen "
              f"pendientes; vuelve a lanzar con menos procesos a la vez",
              flush=True)
    if remoto:
        comun.sincronizar(args.salida, remoto)


def guardar_mejor(salida, idt, mejor):
    """Vuelca el mejor de esta tarea en un .pt con el mismo formato que el
    pipeline local (lo carga `politica.Politica.cargar` sin cambios).

    NO pisa un fichero que ya guarde una red MEJOR. El "mejor hasta ahora" se
    lleva en memoria, así que una segunda ejecución del mismo trabajador (para
    recoger las que se quedaron sin memoria, o tras un corte) empieza otra vez
    de cero y, sin esta comprobación, sobrescribe con su ganadora local a la
    campeona de la tanda anterior. Pasó: la red de nota 1,27 de la fase
    aleatoria se perdió así, y hubo que reentrenarla desde el registro."""
    import torch
    nota, val, c, media, escala, estado, dim = mejor
    ruta = os.path.join(salida, f"mejor_t{idt:03d}.pt")
    if os.path.exists(ruta):
        try:
            previa = torch.load(ruta, map_location="cpu",
                                weights_only=False)["config"]["nota_rollout"]
            if previa is not None and previa >= nota:
                return
        except (OSError, KeyError, RuntimeError, EOFError):
            pass                      # fichero ilegible o a medias: se rehace
    pol.configurar_representacion(c["n_vecinos"], c["horizonte"], c["h_pasado"],
                                  c.get("n_fourier", 0), None,
                                  c.get("n_rayos", 0))
    arq = {"oculto": c["oculto"], "n_capas": c["n_capas"],
           "dropout": c["dropout"], "activacion": c["activacion"],
           "normalizacion": c.get("normalizacion", "no")}
    cfg = ent._cfg(arq, nota, val)
    cfg["hiperparametros"] = {k: c[k] for k in c}
    ent._guardar(ruta, cfg, media, escala, estado)


if __name__ == "__main__":
    main()

