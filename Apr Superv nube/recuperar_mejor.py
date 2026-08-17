#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reentrena la MEJOR configuración de un barrido y la guarda.

Los registros del barrido (barrido_t*.csv) guardan la configuración y la nota
de todo lo probado, pero no los pesos: solo se conserva el .pt de la mejor de
cada trabajador. Si esa campeona se pierde —por un corte, o porque una segunda
ejecución del mismo trabajador sobrescribió su fichero— se puede rehacer desde
el registro, que es lo que hace este script.

Entrenar una red concreta cuesta minutos, así que sale mucho más barato que
guardar los pesos de las miles que se prueban.

Ojo: la nota no saldrá idéntica. El arranque de la red es aleatorio, así que
dos entrenamientos de la MISMA configuración se separan un poco. Sirve para
saber, además, cuánta de la nota original era la configuración y cuánta la
suerte del sorteo inicial.

Uso:
    python recuperar_mejor.py
    python recuperar_mejor.py --salida-pt datos/modelos/campeona.pt
"""

import argparse
import csv
import glob
import os
import time

import comun
from comun import MODELOS_DIR, MUESTRAS_DIR, asegurar
import entrenar_nube as EN
import escenarios as esc

import entrenar as ent


def mejor_del_registro(carpeta):
    """(configuración, nota) de la mejor fila de todos los barrido_t*.csv."""
    mejor, nota_mejor = None, -1.0
    for f in sorted(glob.glob(os.path.join(carpeta, "barrido_t*.csv"))):
        with open(f, encoding="utf-8", newline="") as fh:
            for fila in csv.DictReader(fh):
                try:
                    n = float(fila["nota_seleccion"])
                except (KeyError, TypeError, ValueError):
                    continue
                if n > nota_mejor:
                    nota_mejor, mejor = n, fila
    if mejor is None:
        raise SystemExit(f"No hay ninguna fila con nota en {carpeta}.")
    c = {}
    for k, valores in EN.ESPACIO.items():
        if k not in mejor:
            raise SystemExit(f"El registro no trae la columna '{k}'.")
        muestra = valores[0]
        if isinstance(muestra, str):
            c[k] = mejor[k]
        elif isinstance(muestra, int):
            c[k] = int(float(mejor[k]))
        else:
            c[k] = float(mejor[k])
    return c, nota_mejor


def main():
    ap = argparse.ArgumentParser(
        description="Reentrena y guarda la mejor configuración del barrido.")
    ap.add_argument("--muestras", default=MUESTRAS_DIR)
    ap.add_argument("--escenarios", default=esc.CARPETA)
    ap.add_argument("--registros", default=MODELOS_DIR)
    ap.add_argument("--salida-pt", dest="salida_pt", default=None,
                    help="def. modelos/mejor_global.pt")
    ap.add_argument("--n-escenarios", dest="n_escenarios", type=int, default=400)
    ap.add_argument("--enfasis", type=float, default=1.0)
    ap.add_argument("--fraccion-datos", dest="fraccion_datos", type=float,
                    default=1.0)
    ap.add_argument("--max-muestras", dest="max_muestras", type=int,
                    default=12_000_000)
    ap.add_argument("--en-cpu", dest="en_cpu", action="store_true")
    args = ap.parse_args()

    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    c, nota_original = mejor_del_registro(args.registros)
    print(f"[recuperar] mejor del registro: nota {nota_original:.4f}")
    print(f"[recuperar] {c}", flush=True)

    t0 = time.perf_counter()
    datos = EN.Datos(args.muestras, device, args.fraccion_datos,
                     args.max_muestras, en_cpu=args.en_cpu)
    flotas, opts = esc.flotas(esc.cargar("seleccion", args.escenarios),
                              limite=args.n_escenarios)
    V, media, escala = datos.vista(c["n_vecinos"], c["horizonte"], c["h_pasado"])
    print(f"[recuperar] datos listos en {time.perf_counter() - t0:.1f} s · "
          f"entrenando…", flush=True)

    t1 = time.perf_counter()
    red, estado, val, eps = EN.entrenar_config(V, datos, c, device, criba=None,
                                               enfasis=args.enfasis)
    nota = EN.nota_de(red, estado, media, escala, device, flotas, opts, c)
    print(f"[recuperar] entrenada en {(time.perf_counter() - t1) / 60:.1f} min "
          f"· {eps} épocas · val {val:.5f} · nota {nota:.4f} "
          f"(original {nota_original:.4f})", flush=True)

    ruta = args.salida_pt or os.path.join(args.registros, "mejor_global.pt")
    asegurar(os.path.dirname(ruta))
    import politica as pol
    pol.configurar_representacion(c["n_vecinos"], c["horizonte"], c["h_pasado"])
    arq = {"oculto": c["oculto"], "n_capas": c["n_capas"],
           "dropout": c["dropout"], "activacion": c["activacion"],
           "normalizacion": c.get("normalizacion", "no")}
    cfg = ent._cfg(arq, nota, val)
    cfg["hiperparametros"] = dict(c)
    ent._guardar(ruta, cfg, media.cpu().numpy(), escala.cpu().numpy(),
                 {n: t.detach().cpu() for n, t in estado.items()})
    print(f"[recuperar] guardada en {ruta}")


if __name__ == "__main__":
    main()
