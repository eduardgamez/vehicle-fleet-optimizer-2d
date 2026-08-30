#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FASE 4 · ¿Ayuda una red PROFUNDA?

La fase 1 dejó una conclusión rara: el tamaño de la red no movía el choque
(98,5 % con menos de 0,3M de parámetros frente a 98,9 % con más de 10M). Pero
todas aquellas redes eran ANCHAS y cortas —el espacio de búsqueda llega a seis
capas— y una pila de muchas capas sin más no es comparable: a partir de siete u
ocho, la corrección que viene de la salida se diluye al viajar hacia atrás y las
primeras capas casi no aprenden. Una red honda que sale peor no demuestra que la
profundidad no sirva; demuestra que esa red no se ha entrenado.

Por eso aquí las capas van con ATAJO (ver politica.clase_residual): cada una
suma su resultado a lo que recibió, así que solo tiene que aportar una
corrección y el gradiente baja por un camino directo. Con eso, si las profundas
no ganan, la respuesta ya es sobre la profundidad y no sobre el entrenamiento.

Siete modelos: seis profundos (7, 9 y 11 capas, en dos anchuras) y un CONTROL de
cuatro capas con la misma receta y las mismas épocas. El control es lo que hace
interpretable el resultado: sin él, si las profundas puntúan 0,11 no se sabría
si es por la profundidad o porque 480 épocas suben a cualquiera.

Todo lo demás es la configuración que mejor media saca en la fase 3, tocando
solo lo que el experimento pide: más épocas (480) y algo más de dropout (0,3),
que con tantas capas hace falta.

Uso:
    python fase4.py --idt 0 --frac-vram 0.27      (uno de tres)
    python fase4.py --resumen
"""

import argparse
import os
import time

import numpy as np

from comun import MODELOS_DIR, MUESTRAS_DIR, asegurar
import entrenar_nube as EN
import escenarios as esc
import fase3

# La receta de fondo: la mejor configuración revalidada en la fase 3. No se
# elige nada aquí, se hereda, para que la única diferencia entre los siete
# modelos sea su forma.
BASE = {
    "lote": 2048, "lr": 0.003, "activacion": "gelu", "normalizacion": "layernorm",
    "optimizador": "adamw", "weight_decay": 0.01, "mezcla": "raros_x4",
    "n_vecinos": 3, "h_pasado": 1, "horizonte": 20, "n_fourier": 0,
    "n_rayos": 18,
    # Más apagado de neuronas que en la fase 3 (0,2): cuantas más capas, más
    # fácil es que la red se aprenda los datos de memoria.
    "dropout": 0.3,
    # 480 y no 320: en la fase 3 las mejores seguían mejorando al final, y estas
    # redes son diez veces más pequeñas, así que las épocas de más salen baratas.
    "epocas": 480,
}

# (nombre, capas, ancho, con atajo). El control va SIN atajo y con cuatro capas:
# es la red de siempre, entrenada exactamente igual que las demás.
MODELOS = [
    ("d07x1024", 7, 1024, True),
    ("d07x2048", 7, 2048, True),
    ("d09x1024", 9, 1024, True),
    ("d09x2048", 9, 2048, True),
    ("d11x1024", 11, 1024, True),
    ("d11x2048", 11, 2048, True),
    ("control04x2048", 4, 2048, False),
]

# Una sola semilla por modelo: esto no elige un campeón (para eso está la fase
# 3), solo mira si la profundidad mueve algo. Si dos formas quedan pegadas,
# se repiten con más semillas y ya.
SEMILLAS = (0,)

CARPETA = os.path.join(MODELOS_DIR, "fase4")
PATRON = "fase4_t*.csv"


def configs():
    """(nombre, config) de los siete modelos."""
    salida = []
    for nombre, capas, ancho, atajo in MODELOS:
        c = dict(BASE)
        c.update(n_capas=capas, oculto=ancho, residual=atajo)
        salida.append((nombre, c))
    return salida


CAMPOS = ["id_config", "semilla", "nota", "val_mse", "epocas", "segundos",
          "dim_entrada", "params", "n_capas", "oculto", "residual", "dropout",
          "lr", "lote", "activacion", "normalizacion", "optimizador",
          "weight_decay", "mezcla", "n_vecinos", "h_pasado", "horizonte",
          "n_fourier", "n_rayos"]


def resumen(carpeta=CARPETA):
    """Los siete, del mejor al peor. Lo que se mira no es el número de arriba
    sino la comparación con el control: si las profundas no le sacan nada, el
    tamaño no era el problema."""
    filas = []
    for r in fase3._leer_csv(carpeta, PATRON):
        try:
            filas.append((float(r["nota"]), r))
        except (KeyError, TypeError, ValueError):
            continue
    if not filas:
        print("[fase4] todavía no hay ninguna medida")
        return
    filas.sort(reverse=True)
    print("%-16s %7s %9s %8s %7s" % ("modelo", "nota", "params", "min", "atajo"))
    for nota, r in filas:
        print("%-16s %7.4f %8.1fM %8.1f %7s"
              % (r["id_config"], nota, float(r["params"]) / 1e6,
                 float(r["segundos"]) / 60.0, r["residual"]))


def construir_parser():
    ap = argparse.ArgumentParser(
        description="Experimento de profundidad con capas residuales.")
    ap.add_argument("--muestras", default=MUESTRAS_DIR)
    ap.add_argument("--escenarios", default=esc.CARPETA)
    ap.add_argument("--salida", default=CARPETA)
    ap.add_argument("--idt", type=int, default=0)
    ap.add_argument("--epocas", type=int, default=BASE["epocas"])
    ap.add_argument("--n-escenarios", dest="n_escenarios", type=int, default=400)
    ap.add_argument("--frac-vram", dest="frac_vram", type=float, default=1.0)
    ap.add_argument("--en-cpu", dest="en_cpu", action="store_true")
    ap.add_argument("--enfasis", type=float, default=1.0)
    ap.add_argument("--reanudar", action="store_true",
                    help="liberar las reservas que quedaron a medias tras un "
                         "corte. Solo el PRIMER proceso de la tanda")
    ap.add_argument("--resumen", action="store_true")
    ap.add_argument("--tf32", action="store_true", default=True)
    return ap


def main():
    args = construir_parser().parse_args()
    asegurar(args.salida)
    if args.resumen:
        resumen(args.salida)
        return

    import torch
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and 0 < args.frac_vram < 1.0:
        torch.cuda.set_per_process_memory_fraction(args.frac_vram)

    t0 = time.perf_counter()
    datos = EN.Datos(args.muestras, device, 1.0, None, en_cpu=args.en_cpu)
    flotas, opts = esc.flotas(esc.cargar("seleccion", args.escenarios),
                              limite=args.n_escenarios)
    print("[fase4] proceso %d · %s · %s muestras · %d escenarios · carga %.1f s"
          % (args.idt, device, format(len(datos.X), ","), len(flotas),
             time.perf_counter() - t0), flush=True)

    ruta_csv = os.path.join(args.salida, PATRON.replace("*", str(args.idt)))
    hechas = fase3.filas_hechas(args.salida, PATRON)
    if args.reanudar:
        fase3.limpiar_reservas(args.salida, hechas)
    for semilla in SEMILLAS:
        for nombre, base in configs():
            if ((nombre, semilla) in hechas
                    or not fase3._reservar(args.salida, nombre, semilla)):
                continue
            c = dict(base)
            c["epocas"] = args.epocas
            torch.manual_seed(semilla)
            torch.cuda.manual_seed_all(semilla)
            np.random.seed(semilla)
            ini = time.perf_counter()
            try:
                V, media, escala = datos.vista(c["n_vecinos"], c["horizonte"],
                                               c["h_pasado"], c["n_fourier"],
                                               c["n_rayos"])
                red, estado, val, eps = EN.entrenar_config(
                    V, datos, c, device, criba=None, enfasis=args.enfasis)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                fase3._soltar(args.salida, nombre, semilla)
                print("[fase4] %s: no cabe en memoria" % nombre, flush=True)
                continue
            params = sum(p.numel() for p in red.parameters())
            nota = EN.nota_de(red, estado, media, escala, device, flotas, opts, c)
            segs = round(time.perf_counter() - ini, 1)
            fila = {k: c.get(k) for k in CAMPOS if k in c}
            fila.update(id_config=nombre, semilla=semilla,
                        nota=round(float(nota), 5), val_mse=round(float(val), 5),
                        epocas=int(eps), segundos=segs,
                        dim_entrada=int(V.shape[1]), params=params)
            _anotar(ruta_csv, fila)
            print("[fase4] %s: nota %.4f · %.1fM params · %.1f min"
                  % (nombre, nota, params / 1e6, segs / 60.0), flush=True)
            EN.guardar_mejor(args.salida, nombre,
                             (nota, val, dict(c), media.cpu().numpy(),
                              escala.cpu().numpy(),
                              {n: t.detach().cpu() for n, t in estado.items()},
                              V.shape[1]))
            del red, estado, V
            torch.cuda.empty_cache()
    print("[fase4] no queda nada por hacer en este proceso", flush=True)


def _anotar(ruta, fila):
    """Como el de la fase 3, pero con sus columnas (aquí interesan los
    parámetros y si la red llevaba atajo)."""
    import csv
    nuevo = not os.path.exists(ruta)
    with open(ruta, "a", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CAMPOS)
        if nuevo:
            w.writeheader()
        w.writerow({k: fila.get(k) for k in CAMPOS})


if __name__ == "__main__":
    main()
