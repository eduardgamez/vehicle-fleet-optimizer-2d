#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FASE 5 · Rejilla fina alrededor de la red que gana.

De dónde sale. La fase 4 preguntaba si convenía una red PROFUNDA y contestó que
no: con el atajo residual puesto —o sea, sin la excusa de que no se entrenaban—
la nota bajaba de forma ordenada al añadir capas (7 capas 0,1031 · 11 capas
0,0862), y los dos controles de cuatro capas se quedaron arriba (0,1250 con
2048 de ancho y 0,1229 con 3072). Pero de paso apareció otra cosa: ese control
de 0,1250 es la MISMA configuración que ganó la fase 3 con 0,1130, y la única
diferencia eran las épocas (480 en vez de 320) y el dropout (0,3 en vez de 0,2).

Así que aquí se deja de mover la forma de la red y se barren en rejilla los tres
ejes que sí se están moviendo:

    capas    3 · 4 · 5          (alrededor de las 4 que ganan)
    dropout  0,2 · 0,3 · 0,4    (0,4 no se había probado nunca)
    épocas   480 · 640 · 800    (para ver dónde deja de subir)

27 combinaciones. Las épocas van bien separadas a propósito: entre 480 y 520 hay
un 8 % de diferencia y el ruido entre semillas es de 0,008-0,030, así que ese
eje habría medido ruido en vez de presupuesto.

La anchura se queda fija en 2048: es la que gana (0,1250 frente a 0,1229 del
3072), la que la fase 3 dejó arriba, y encima la más rápida de las dos —26 min
frente a 116—, que es lo que hace viable barrer 27 puntos. Todo lo demás es la
receta heredada, sin atajo residual (con 3-5 capas no hace falta).

Uso:
    python fase5.py --idt 0 --frac-vram 0.27      (uno de tres)
    python fase5.py --resumen
"""

import argparse
import os
import time

import numpy as np

from comun import MODELOS_DIR, MUESTRAS_DIR, asegurar
import entrenar_nube as EN
import escenarios as esc
import fase3
import fase4

# La receta que no se toca: la de la fase 4, que a su vez es la mejor de la
# fase 3. Aquí solo se mueven capas, dropout y épocas.
BASE = dict(fase4.BASE)
BASE.update(oculto=2048, residual=False)

CAPAS = (3, 4, 5)
DROPOUTS = (0.2, 0.3, 0.4)
EPOCAS = (480, 640, 800)

# Una semilla por punto: son 27 puntos y lo que se busca es la FORMA de la
# superficie (dónde sube y dónde baja), no el campeón. El ganador se revalida
# después con varias semillas, como se hizo en la fase 3.
SEMILLAS = (0,)

CARPETA = os.path.join(MODELOS_DIR, "fase5")
PATRON = "fase5_t*.csv"


def configs():
    """(nombre, config) de los 27 puntos de la rejilla.

    El orden importa: se recorre por ÉPOCAS de menor a mayor, así que si hay que
    parar a media tanda queda medida la rejilla entera con el presupuesto corto
    —que ya dice algo— en vez de un tercio de la rejilla con los tres."""
    salida = []
    for epocas in EPOCAS:
        for capas in CAPAS:
            for drop in DROPOUTS:
                c = dict(BASE)
                c.update(n_capas=capas, dropout=drop, epocas=epocas)
                nombre = "c%d-d%02d-e%d" % (capas, round(drop * 100), epocas)
                salida.append((nombre, c))
    return salida


def resumen(carpeta=CARPETA):
    """Dos vistas: la lista de mejores y la rejilla capas x dropout para cada
    presupuesto, que es donde se ve si hay una zona buena o son picos sueltos."""
    filas = []
    for r in fase3._leer_csv(carpeta, PATRON):
        try:
            filas.append((float(r["nota"]), int(r["n_capas"]),
                          float(r["dropout"]), int(r["epocas"]), r))
        except (KeyError, TypeError, ValueError):
            continue
    if not filas:
        print("[fase5] todavía no hay ninguna medida")
        return
    print("Mejores:")
    for nota, capas, drop, epocas, r in sorted(filas, reverse=True)[:8]:
        print("  %7.4f  %d capas · dropout %.1f · %d épocas · %.0f min"
              % (nota, capas, drop, epocas, float(r["segundos"]) / 60.0))
    hecho = {(c, d, e): n for n, c, d, e, _ in filas}
    for epocas in EPOCAS:
        if not any(k[2] == epocas for k in hecho):
            continue
        print("\n%d épocas      %s" % (epocas,
                                       "  ".join("drop %.1f" % d
                                                 for d in DROPOUTS)))
        for capas in CAPAS:
            celdas = []
            for drop in DROPOUTS:
                n = hecho.get((capas, drop, epocas))
                celdas.append("   %6.4f" % n if n is not None else "        ·")
            print("  %d capas   %s" % (capas, " ".join(celdas)))


def construir_parser():
    ap = argparse.ArgumentParser(
        description="Rejilla capas x dropout x épocas sobre la red que gana.")
    ap.add_argument("--muestras", default=MUESTRAS_DIR)
    ap.add_argument("--escenarios", default=esc.CARPETA)
    ap.add_argument("--salida", default=CARPETA)
    ap.add_argument("--idt", type=int, default=0)
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
    print("[fase5] proceso %d · %s · %s muestras · %d escenarios · carga %.1f s"
          % (args.idt, device, format(len(datos.X), ","), len(flotas),
             time.perf_counter() - t0), flush=True)

    ruta_csv = os.path.join(args.salida, PATRON.replace("*", str(args.idt)))
    hechas = fase3.filas_hechas(args.salida, PATRON)
    if args.reanudar:
        fase3.limpiar_reservas(args.salida, hechas)
    for semilla in SEMILLAS:
        for nombre, c in configs():
            if ((nombre, semilla) in hechas
                    or not fase3._reservar(args.salida, nombre, semilla)):
                continue
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
                print("[fase5] %s: no cabe en memoria" % nombre, flush=True)
                continue
            params = sum(p.numel() for p in red.parameters())
            nota = EN.nota_de(red, estado, media, escala, device, flotas, opts, c)
            segs = round(time.perf_counter() - ini, 1)
            fila = {k: c.get(k) for k in fase4.CAMPOS if k in c}
            fila.update(id_config=nombre, semilla=semilla,
                        nota=round(float(nota), 5), val_mse=round(float(val), 5),
                        epocas=int(eps), segundos=segs,
                        dim_entrada=int(V.shape[1]), params=params)
            fase4._anotar(ruta_csv, fila)
            print("[fase5] %s: nota %.4f · %.1f min"
                  % (nombre, nota, segs / 60.0), flush=True)
            EN.guardar_mejor(args.salida, nombre,
                             (nota, val, dict(c), media.cpu().numpy(),
                              escala.cpu().numpy(),
                              {n: t.detach().cpu() for n, t in estado.items()},
                              V.shape[1]))
            del red, estado, V
            torch.cuda.empty_cache()
    print("[fase5] no queda nada por hacer en este proceso", flush=True)


if __name__ == "__main__":
    main()
