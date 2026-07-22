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
import vectorizado as vec

import entrenar as ent
from entrenar import CAMPOS_EXTRA, _puntuar_vehiculo, crear_red

# Espacio de búsqueda: el del pipeline local con el eje de VECINOS reajustado al
# tamaño de flota que se genera. El tope sale del propio problema, no de la red:
# con flotas de hasta 8 vehículos nadie puede ver más de 7 vecinos, y valores más
# altos serían bloques de entrada siempre a cero. El resto de ejes queda igual.
ESPACIO = dict(ent.ESPACIO)
ESPACIO["n_vecinos"] = [3, 4, 5, 7]

CAMPOS = ["nota_seleccion" if c == "nota_rollout" else c for c in CAMPOS_EXTRA]
CAMPOS += ["epocas_hechas"]


# --------------------------------------------------------------------------- #
# Carga del superset a la GPU
# --------------------------------------------------------------------------- #
class Datos:
    """Superset de muestras residente en la GPU, con las utilidades para sacar
    de él la vista de cualquier configuración.

    'fraccion' es la parte del dataset que se usa. La idea es escalonarla: la
    búsqueda amplia solo tiene que ORDENAR configuraciones entre sí, y para eso
    sobra con una parte; la rejilla fina afina de verdad y merece bastante más;
    y el modelo final se entrena con todo. Se submuestrea por FILAS y no por
    escenarios, para que con poca fracción se sigan viendo todos los escenarios
    (menos instantes de cada uno) en vez de unos pocos escenarios enteros:
    interesa más la variedad de situaciones que el detalle de cada una."""

    def __init__(self, carpeta, device, fraccion=1.0, max_muestras=None,
                 semilla=0, en_cpu=False):
        import torch
        with open(os.path.join(carpeta, "meta.json"), encoding="utf-8") as f:
            self.meta = json.load(f)
        Xs, TCs, Ys, Vs = [], [], [], []
        for lote in self.meta["lotes"]:
            Xs.append(np.load(os.path.join(carpeta, f"{lote}_X.npy")))
            TCs.append(np.load(os.path.join(carpeta, f"{lote}_TC.npy")))
            Ys.append(np.load(os.path.join(carpeta, f"{lote}_Y.npy")))
            Vs.append(np.load(os.path.join(carpeta, f"{lote}_val.npy")))
        X = np.concatenate(Xs)
        TC = np.concatenate(TCs)
        Y = np.concatenate(Ys)
        val = np.concatenate(Vs)

        objetivo = len(X)
        if fraccion < 1.0:
            objetivo = max(1, int(round(fraccion * objetivo)))
        if max_muestras:
            objetivo = min(objetivo, max_muestras)
        if objetivo < len(X):
            # Submuestreo aleatorio pero REPRODUCIBLE.
            rng = np.random.default_rng(semilla)
            sel = np.sort(rng.choice(len(X), objetivo, replace=False))
            X, TC, Y, val = X[sel], TC[sel], Y[sel], val[sel]

        self.n_vec_max = self.meta["n_vec_max"]
        self.h_max = self.meta["h_max"]
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
        self.idx_tr = torch.from_numpy(np.flatnonzero(~val)).to(self.dev_datos)
        self.idx_va = torch.from_numpy(np.flatnonzero(val)).to(self.dev_datos)
        if len(self.idx_va) == 0:               # dataset diminuto
            self.idx_va = self.idx_tr[-1:]
        self.cols_cache = {}

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

    def vista(self, n_vec, horizonte, h_pasado):
        """(tensor (M, dim) normalizado, media, escala) de una configuración."""
        import torch
        clave = (n_vec, h_pasado)
        if clave not in self.cols_cache:
            cols = vec.columnas_vista(n_vec, h_pasado, self.n_vec_max, self.h_max)
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
def entrenar_config(V, datos, c, device, criba=None):
    """Entrena una red y devuelve (mejor_state_dict, mejor_val, épocas hechas)."""
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

    idx_tr, idx_va = datos.idx_tr, datos.idx_va
    # Los hitos de la criba son FRACCIONES del presupuesto de esta configuración,
    # no épocas absolutas: con presupuestos de 40, 80 y 120 épocas y un ritmo de
    # aprendizaje que se estira con el total, la época 20 significa cosas muy
    # distintas en cada una, y comparar ahí penalizaría a las de presupuesto largo.
    hitos = {}
    if criba is not None:
        for f in criba.fracciones:
            hitos[max(1, int(round(f * epocas)))] = f

    mejor_val, mejor_estado, hechas = float("inf"), None, 0
    for ep in range(1, epocas + 1):
        red.train()
        perm = torch.randperm(len(idx_tr), device=datos.dev_datos)
        for i in range(0, len(perm), lote):
            idx = idx_tr[perm[i:i + lote]]
            xb, yb = datos.lote(V, idx)
            opt.zero_grad(set_to_none=True)
            perdida(red(xb), yb).backward()
            opt.step()
        plani.step()
        red.eval()
        with torch.no_grad():
            tot = 0.0
            for i in range(0, len(idx_va), 65536):
                xb, yb = datos.lote(V, idx_va[i:i + 65536])
                tot += perdida(red(xb), yb).item() * len(xb)
            lv = tot / len(idx_va)
        hechas = ep
        # Solo cuenta una época con pérdida FINITA: con tasas altas el
        # entrenamiento puede divergir a NaN, y NaN < x siempre es falso.
        if math.isfinite(lv) and lv < mejor_val:
            mejor_val = lv
            mejor_estado = copy.deepcopy(red.state_dict())
        if ep in hitos and criba.para(hitos[ep], mejor_val):
            break
    if mejor_estado is None:
        mejor_estado = copy.deepcopy(red.state_dict())
    return red, mejor_estado, mejor_val, hechas


def nota_de(red, estado, media, escala, device, flotas, opts, c):
    """Nota media por vehículo de una red sobre un conjunto de escenarios."""
    from politica import Politica
    red.load_state_dict(estado)
    politica = Politica(red.eval(), media, escala, device)
    vec.rollout(flotas, politica, opts=opts, n_vec=c["n_vecinos"],
                        horizonte=c["horizonte"], h_pasado=c["h_pasado"])
    total = sum(_puntuar_vehiculo(v) for f in flotas for v in f)
    n = sum(len(f) for f in flotas)
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
    todas = list(itertools.product(*(espacio[k] for k in claves)))
    _rnd.shuffle(todas)
    if not args.exhaustivo:
        n = (int(round(args.fraccion * len(todas))) if args.fraccion
             else args.n_configs)
        todas = todas[:max(1, n)]
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
    ap.add_argument("--max-muestras", dest="max_muestras", type=int,
                    default=12_000_000,
                    help="tope duro de muestras, por la memoria de la GPU "
                         "(def. 12 M; con --en-cpu se puede subir)")
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
    idt = args.tarea if args.tarea is not None else comun.indice_tarea()
    asegurar(args.salida)
    remoto = comun.bucket_env("modelos")
    comun.recuperar(args.salida, remoto)
    comun.arrancar_sincronizacion(args.salida, remoto, args.sync_min)

    t0 = time.perf_counter()
    datos = Datos(args.muestras, device, args.fraccion_datos,
                  args.max_muestras, en_cpu=args.en_cpu)
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
    t_ini = time.perf_counter()
    with open(log_path, "a" if hechas else "w", newline="",
              encoding="utf-8") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=campos)
        if not hechas:
            w.writeheader()
        for k, c in enumerate(configs, 1):
            t_c = time.perf_counter()
            V, media, escala = datos.vista(c["n_vecinos"], c["horizonte"],
                                           c["h_pasado"])
            red, estado, val, eps = entrenar_config(V, datos, c, device, criba)
            nota = nota_de(red, estado, media, escala, device, flotas, opts, c)
            fila = dict(c)
            fila.update({"dim_entrada": V.shape[1], "n_muestras": len(datos.idx_tr),
                         "val_mse": round(val, 5),
                         "nota_seleccion": round(nota, 4),
                         "epocas_hechas": eps,
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

    print(f"[barrido] tarea {idt}: {len(configs)} configuraciones en "
          f"{(time.perf_counter() - t_ini) / 60:.1f} min · mejor "
          f"{mejor[0]:.4f}", flush=True)
    if remoto:
        comun.sincronizar(args.salida, remoto)


def guardar_mejor(salida, idt, mejor):
    """Vuelca el mejor de esta tarea en un .pt con el mismo formato que el
    pipeline local (lo carga `politica.Politica.cargar` sin cambios)."""
    nota, val, c, media, escala, estado, dim = mejor
    pol.configurar_representacion(c["n_vecinos"], c["horizonte"], c["h_pasado"])
    arq = {"oculto": c["oculto"], "n_capas": c["n_capas"],
           "dropout": c["dropout"], "activacion": c["activacion"],
           "normalizacion": c.get("normalizacion", "no")}
    cfg = ent._cfg(arq, nota, val)
    cfg["hiperparametros"] = {k: c[k] for k in c}
    ent._guardar(os.path.join(salida, f"mejor_t{idt:03d}.pt"), cfg,
                 media, escala, estado)


if __name__ == "__main__":
    main()
