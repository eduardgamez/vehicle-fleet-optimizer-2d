#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entrenamiento de la política neuronal a partir de los CSV de rutas.

Lee TODOS los .csv bajo la carpeta rutas/ (recursivo: incluye los clasificados
por nº de vehículos que escribe generador.py y el dataset acumulado de la GUI),
reconstruye por cada instante el vector de entrada definido en politica.py
(estado propio + meta + controles pasados + vecinos) y el objetivo (los
próximos N_PRED pares (a, giro) normalizados), y entrena el MLP con MSE.

La validación se separa POR RUN (escenario completo), no por fila, para no
contaminar la validación con instantes vecinos de las mismas rutas.

Requiere:   pip install torch numpy
Ejecutar:   python entrenar.py                  (usa rutas/ y 30 épocas)
            python entrenar.py --epocas 60 --oculto 1024
"""

import argparse
import ast
import glob
import math
import os
import time
import zlib

import numpy as np

from nucleo import DT, RUTAS_DIR
from politica import (DIM_ENTRADA, H_PASADO, N_PRED, MODELO_PT,
                      vector_entrada, crear_red)


# --------------------------------------------------------------------------- #
# Lectura de los CSV
# --------------------------------------------------------------------------- #
def _parsear_condiciones(linea):
    """De la línea '# run=... condiciones_iniciales=[...]' devuelve el dict
    id → parámetros del vehículo (ángulos ya en RADIANES, meta como tupla)."""
    lista = ast.literal_eval(linea.split("condiciones_iniciales=", 1)[1])
    conds = {}
    for d in lista:
        ang = d.get("angulo_llegada")
        conds[d["id"]] = {
            "meta": (float(d["meta"][0]), float(d["meta"][1])),
            "meta_th": (None if ang in (None, "libre")
                        else math.radians(float(ang))),
            "largo": float(d["largo"]), "ancho": float(d["ancho"]),
            "v_max": float(d["v_max"]), "a_max": float(d["a_max"]),
            "giro_max": math.radians(float(d["giro_max"])),
            "grupo": int(d.get("grupo", 1)),
            "prioridad": int(d.get("prioridad", 1)),
        }
    return conds


def leer_runs(carpeta):
    """Recorre los CSV y devuelve una lista de runs:
    (clave_run, condiciones, {vid: array (n, 6) de [x, y, th, v, a, giro]})."""
    archivos = sorted(glob.glob(os.path.join(carpeta, "**", "*.csv"),
                                recursive=True))
    runs = []
    for arch in archivos:
        conds, filas = None, {}
        run_actual = None

        def cerrar():
            if conds and filas:
                datos = {vid: np.array(v, dtype=np.float64)
                         for vid, v in filas.items()}
                runs.append((f"{os.path.basename(arch)}::{run_actual}",
                             conds, datos))

        with open(arch, encoding="utf-8") as f:
            for linea in f:
                linea = linea.strip()
                if not linea or linea.startswith("run,"):
                    continue
                if linea.startswith("#"):
                    cerrar()                      # fin del run anterior
                    conds, filas = None, {}
                    if "condiciones_iniciales=" in linea:
                        try:
                            conds = _parsear_condiciones(linea)
                            run_actual = linea.split("run=", 1)[1].split()[0]
                        except Exception:
                            conds = None          # comentario corrupto: se salta
                    continue
                if conds is None:
                    continue
                c = linea.split(",")
                vid = int(c[1])
                x1, y1, x2, y2 = map(float, c[3:7])
                v, th, a, giro = map(float, c[7:11])
                # Estado por instante: centro del OBB (mitad de las esquinas
                # opuestas), rumbo, velocidad y el control aplicado.
                filas.setdefault(vid, []).append(
                    ((x1 + x2) / 2.0, (y1 + y2) / 2.0, th, v, a, giro))
        cerrar()
    return runs


# --------------------------------------------------------------------------- #
# Construcción de muestras (una por vehículo e instante)
# --------------------------------------------------------------------------- #
def construir_muestras(runs):
    X, Y = [], []
    for _, conds, datos in runs:
        vids = [vid for vid in datos if vid in conds]
        for vid in vids:
            arr = datos[vid]
            p = conds[vid]
            n = len(arr)
            for t in range(n):
                pasado = [(arr[k, 4], arr[k, 5])
                          for k in range(max(0, t - H_PASADO), t)]
                ego = {"x": arr[t, 0], "y": arr[t, 1], "th": arr[t, 2],
                       "v": arr[t, 3], "largo": p["largo"], "ancho": p["ancho"],
                       "v_max": p["v_max"], "a_max": p["a_max"],
                       "giro_max": p["giro_max"], "grupo": p["grupo"],
                       "prioridad": p["prioridad"], "meta": p["meta"],
                       "meta_th": p["meta_th"], "pasado": pasado}
                otros = []
                for ov in vids:
                    if ov == vid:
                        continue
                    oa, op = datos[ov], conds[ov]
                    if t < len(oa):
                        ox, oy, oth, ovel = oa[t, 0], oa[t, 1], oa[t, 2], oa[t, 3]
                    else:               # ya aparcado: última pose, v = 0
                        ox, oy, oth, ovel = oa[-1, 0], oa[-1, 1], oa[-1, 2], 0.0
                    otros.append({"x": ox, "y": oy, "th": oth, "v": ovel,
                                  "largo": op["largo"], "ancho": op["ancho"],
                                  "grupo": op["grupo"],
                                  "prioridad": op["prioridad"]})
                X.append(vector_entrada(ego, otros))
                obj = np.zeros(2 * N_PRED, dtype=np.float32)
                for k in range(N_PRED):
                    if t + k < n:       # más allá del final: aparcado (0, 0)
                        obj[2 * k] = arr[t + k, 4] / max(p["a_max"], 1e-6)
                        obj[2 * k + 1] = (arr[t + k, 5]
                                          / max(p["giro_max"], 1e-6))
                Y.append(obj)
    return (np.asarray(X, dtype=np.float32),
            np.asarray(Y, dtype=np.float32))


def es_validacion(clave_run):
    """~10 % de los runs a validación, de forma estable entre ejecuciones."""
    return zlib.crc32(clave_run.encode()) % 10 == 0


# --------------------------------------------------------------------------- #
# Entrenamiento
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Entrena la política neuronal.")
    ap.add_argument("--rutas", default=RUTAS_DIR,
                    help="carpeta raíz con los CSV (def. rutas/)")
    ap.add_argument("--epocas", type=int, default=30)
    ap.add_argument("--lote", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--oculto", type=int, default=512,
                    help="anchura de las capas ocultas (def. 512)")
    ap.add_argument("--salida", default=MODELO_PT)
    args = ap.parse_args()

    import torch

    print(f"[datos] leyendo CSVs de {args.rutas} …")
    runs = leer_runs(args.rutas)
    if not runs:
        raise SystemExit("No hay runs en la carpeta de rutas: genera datos "
                         "primero con generador.py (o con la GUI).")
    r_tr = [r for r in runs if not es_validacion(r[0])]
    r_va = [r for r in runs if es_validacion(r[0])]
    if not r_va:                        # pocos runs: aparta el último
        r_va = [r_tr.pop()]
    print(f"[datos] {len(runs)} runs ({len(r_tr)} train / {len(r_va)} val); "
          "construyendo muestras…")
    Xtr, Ytr = construir_muestras(r_tr)
    Xva, Yva = construir_muestras(r_va)
    print(f"[datos] {len(Xtr):,} muestras de train, {len(Xva):,} de val "
          f"(dim entrada {DIM_ENTRADA})")

    media = Xtr.mean(axis=0)
    escala = np.maximum(Xtr.std(axis=0), 1e-6)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] dispositivo: {device}")
    Xtr_t = torch.from_numpy((Xtr - media) / escala).to(device)
    Ytr_t = torch.from_numpy(Ytr).to(device)
    Xva_t = torch.from_numpy((Xva - media) / escala).to(device)
    Yva_t = torch.from_numpy(Yva).to(device)

    red = crear_red(DIM_ENTRADA, args.oculto, N_PRED).to(device)
    opt = torch.optim.AdamW(red.parameters(), lr=args.lr)
    plani = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epocas)
    perdida = torch.nn.MSELoss()

    mejor_val = float("inf")
    os.makedirs(os.path.dirname(args.salida), exist_ok=True)
    for ep in range(1, args.epocas + 1):
        t0 = time.perf_counter()
        red.train()
        orden = torch.randperm(len(Xtr_t), device=device)
        tot = 0.0
        for i in range(0, len(orden), args.lote):
            idx = orden[i:i + args.lote]
            opt.zero_grad()
            l = perdida(red(Xtr_t[idx]), Ytr_t[idx])
            l.backward()
            opt.step()
            tot += l.item() * len(idx)
        plani.step()
        red.eval()
        with torch.no_grad():
            lv = sum(perdida(red(Xva_t[i:i + args.lote]),
                             Yva_t[i:i + args.lote]).item()
                     * len(Xva_t[i:i + args.lote])
                     for i in range(0, len(Xva_t), args.lote)) / len(Xva_t)
        marca = ""
        if lv < mejor_val:
            mejor_val = lv
            torch.save({
                "config": {"dim_entrada": DIM_ENTRADA, "oculto": args.oculto,
                           "n_pred": N_PRED, "h_pasado": H_PASADO},
                "media": media.tolist(), "escala": escala.tolist(),
                "state_dict": red.state_dict(),
            }, args.salida)
            marca = "  ← guardado"
        print(f"[train] época {ep:3d}/{args.epocas} · "
              f"train {tot / len(Xtr_t):.5f} · val {lv:.5f} · "
              f"{time.perf_counter() - t0:.1f} s{marca}")

    print(f"[train] mejor val {mejor_val:.5f} · modelo en {args.salida}")


if __name__ == "__main__":
    main()
