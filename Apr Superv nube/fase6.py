#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FASE 6 · Reválida final de las candidatas de la fase 5.

La fase 5 midió cada punto una sola vez. Sus dos mejores resultados están
separados por 0,00038, muy por debajo del ruido entre semillas observado en la
fase 3. Esta fase fija las candidatas antes de mirar más resultados y las mide
con tres semillas nuevas e independientes de la semilla 0 que las seleccionó.

Se elige por la media de las tres notas. Después no se ajustan más
hiperparámetros: solo queda entrenar una vez el ganador con una semilla fijada y
medirlo una única vez sobre test.

Uso:
    python fase6.py --idt 0 --frac-vram 0.27      (uno de tres procesos)
    python fase6.py --resumen
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np

from comun import MODELOS_DIR, MUESTRAS_DIR, asegurar
import entrenar_nube as EN
import escenarios as esc
import fase3
import fase4
import fase5


# Candidatas fijadas con los resultados completos de fase 5. Se conservan los
# tres dropout de la zona ganadora y el mejor control anterior.
CANDIDATAS = (
    ("c5-d20-e640", 5, 0.2, 640),
    ("c5-d30-e640", 5, 0.3, 640),
    ("c5-d40-e640", 5, 0.4, 640),
    ("c4-d30-e480", 4, 0.3, 480),
)

# La semilla 0 sirvió para escoger las candidatas y no entra en la reválida.
SEMILLAS = (1, 2, 3)

CARPETA = os.path.join(MODELOS_DIR, "fase6")
PATRON = "fase6_t*.csv"
PAUSA = "PAUSAR"
CHECKPOINTS = "checkpoints"


def ruta_pausa(carpeta=CARPETA):
    return os.path.join(carpeta, PAUSA)


def ruta_checkpoint(carpeta, nombre, semilla):
    return os.path.join(carpeta, CHECKPOINTS,
                        "%s_s%d.checkpoint.pt" % (nombre, semilla))


def solicitar_pausa(carpeta=CARPETA):
    asegurar(carpeta)
    with open(ruta_pausa(carpeta), "w", encoding="utf-8") as f:
        f.write("solicitada %s\n" % time.strftime("%Y-%m-%d %H:%M:%S"))
    print("[fase6] pausa solicitada; cada proceso guardará al acabar su época",
          flush=True)


def quitar_pausa(carpeta=CARPETA):
    try:
        os.remove(ruta_pausa(carpeta))
    except FileNotFoundError:
        pass


def estado_checkpoints(carpeta=CARPETA):
    metas = sorted(glob.glob(os.path.join(carpeta, CHECKPOINTS,
                                          "*.checkpoint.pt.json")))
    if not metas:
        print("[fase6] no hay entrenamientos parciales guardados")
        return
    print("Entrenamientos parciales:")
    for ruta in metas:
        try:
            with open(ruta, encoding="utf-8") as f:
                meta = json.load(f)
            nombre = os.path.basename(ruta).replace(".checkpoint.pt.json", "")
            print("  %s · época %d/%d · %.1f min guardados"
                  % (nombre, meta["epoca"], meta["epocas"],
                     meta.get("segundos", 0.0) / 60.0))
        except (OSError, ValueError, KeyError):
            print("  %s · metadatos no legibles" % os.path.basename(ruta))


def lanzar_ocultos(args):
    """Reanuda tres trabajadores en segundo plano sin abrir consolas."""
    hechas = fase3.filas_hechas(args.salida, PATRON)
    quitar_pausa(args.salida)
    fase3.limpiar_reservas(args.salida, hechas)
    for nombre, semilla in hechas:
        checkpoint = ruta_checkpoint(args.salida, nombre, semilla)
        for parcial in (checkpoint, checkpoint + ".json"):
            try:
                os.remove(parcial)
            except FileNotFoundError:
                pass
    base = [sys.executable, os.path.abspath(__file__),
            "--muestras", args.muestras,
            "--escenarios", args.escenarios,
            "--salida", args.salida,
            "--n-escenarios", str(args.n_escenarios),
            "--frac-vram", str(args.frac_vram),
            "--enfasis", str(args.enfasis),
            "--cada-checkpoint", str(args.cada_checkpoint)]
    if args.en_cpu:
        base.append("--en-cpu")
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    procesos = []
    for i in range(3):
        salida = open(os.path.join(args.salida, "log_t%d.txt" % i), "ab")
        error = open(os.path.join(args.salida, "err_t%d.txt" % i), "ab")
        try:
            proceso = subprocess.Popen(
                base + ["--idt", str(i)], cwd=os.path.dirname(__file__),
                stdout=salida, stderr=error, creationflags=flags)
        finally:
            salida.close()
            error.close()
        procesos.append(proceso.pid)
    print("[fase6] procesos ocultos: %s" % ", ".join(map(str, procesos)))


def configs():
    salida = []
    for nombre, capas, drop, epocas in CANDIDATAS:
        c = dict(fase5.BASE)
        c.update(n_capas=capas, dropout=drop, epocas=epocas)
        salida.append((nombre, c))
    return salida


def resumen(carpeta=CARPETA):
    grupos = {}
    for r in fase3._leer_csv(carpeta, PATRON):
        try:
            grupos.setdefault(r["id_config"], []).append(
                (int(r["semilla"]), float(r["nota"]), float(r["segundos"])))
        except (KeyError, TypeError, ValueError):
            continue
    if not grupos:
        print("[fase6] todavía no hay ninguna medida")
        return

    filas = []
    for nombre, medidas in grupos.items():
        notas = np.asarray([x[1] for x in medidas], dtype=np.float64)
        filas.append((float(notas.mean()), float(notas.std()), nombre, medidas))
    print("Reválida final (ordenada por media):")
    for media, desv, nombre, medidas in sorted(filas, reverse=True):
        detalle = " · ".join("s%d %.5f" % (s, n) for s, n, _ in
                             sorted(medidas))
        minutos = sum(x[2] for x in medidas) / 60.0
        print("  %s  media %.5f · desv %.5f · %d/3 · %.0f min-proceso · %s"
              % (nombre, media, desv, len(medidas), minutos, detalle))


def construir_parser():
    ap = argparse.ArgumentParser(
        description="Reválida final con tres semillas nuevas.")
    ap.add_argument("--muestras", default=MUESTRAS_DIR)
    ap.add_argument("--escenarios", default=esc.CARPETA)
    ap.add_argument("--salida", default=CARPETA)
    ap.add_argument("--idt", type=int, default=0)
    ap.add_argument("--n-escenarios", dest="n_escenarios", type=int,
                    default=400)
    ap.add_argument("--frac-vram", dest="frac_vram", type=float, default=1.0)
    ap.add_argument("--en-cpu", dest="en_cpu", action="store_true")
    ap.add_argument("--enfasis", type=float, default=1.0)
    ap.add_argument("--reanudar", action="store_true",
                    help="liberar reservas incompletas; solo en el primer proceso")
    ap.add_argument("--pausar", action="store_true",
                    help="pedir checkpoint y salida limpia a los procesos")
    ap.add_argument("--lanzar", action="store_true",
                    help="reanudar tres procesos ocultos")
    ap.add_argument("--estado", action="store_true",
                    help="mostrar los checkpoints parciales")
    ap.add_argument("--cada-checkpoint", type=int, default=10,
                    help="guardar cada N épocas además de al pausar (def. 10)")
    ap.add_argument("--resumen", action="store_true")
    ap.add_argument("--tf32", action="store_true", default=True)
    return ap


def main():
    args = construir_parser().parse_args()
    asegurar(args.salida)
    if args.pausar:
        solicitar_pausa(args.salida)
        return
    if args.lanzar:
        lanzar_ocultos(args)
        return
    if args.estado:
        resumen(args.salida)
        estado_checkpoints(args.salida)
        return
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
    print("[fase6] proceso %d · %s · %s muestras · %d escenarios · carga %.1f s"
          % (args.idt, device, format(len(datos.X), ","), len(flotas),
             time.perf_counter() - t0), flush=True)

    ruta_csv = os.path.join(args.salida, PATRON.replace("*", str(args.idt)))
    hechas = fase3.filas_hechas(args.salida, PATRON)
    if args.reanudar:
        quitar_pausa(args.salida)
        fase3.limpiar_reservas(args.salida, hechas)
    elif os.path.exists(ruta_pausa(args.salida)):
        print("[fase6] está pausada; reanuda el primer proceso con --reanudar",
              flush=True)
        return
    for semilla in SEMILLAS:
        for nombre, c in configs():
            if os.path.exists(ruta_pausa(args.salida)):
                print("[fase6] pausa completada", flush=True)
                return
            if ((nombre, semilla) in hechas
                    or not fase3._reservar(args.salida, nombre, semilla)):
                continue
            torch.manual_seed(semilla)
            torch.cuda.manual_seed_all(semilla)
            np.random.seed(semilla)
            ini = time.perf_counter()
            checkpoint = ruta_checkpoint(args.salida, nombre, semilla)
            try:
                V, media, escala = datos.vista(c["n_vecinos"], c["horizonte"],
                                               c["h_pasado"], c["n_fourier"],
                                               c["n_rayos"])
                red, estado, val, eps = EN.entrenar_config(
                    V, datos, c, device, criba=None, enfasis=args.enfasis,
                    checkpoint=checkpoint, pausa=ruta_pausa(args.salida),
                    cada_checkpoint=args.cada_checkpoint)
            except EN.EntrenamientoPausado as e:
                print("[fase6] %s s%d: pausado en época %d/%d"
                      % (nombre, semilla, e.epoca, c["epocas"]), flush=True)
                return
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                fase3._soltar(args.salida, nombre, semilla)
                print("[fase6] %s s%d: no cabe en memoria"
                      % (nombre, semilla), flush=True)
                continue

            fin_entreno = time.perf_counter()
            params = sum(p.numel() for p in red.parameters())
            nota = EN.nota_de(red, estado, media, escala, device, flotas, opts, c)
            segs = round(getattr(red, "_segundos_entrenamiento",
                                 fin_entreno - ini)
                         + time.perf_counter() - fin_entreno, 1)
            fila = {k: c.get(k) for k in fase4.CAMPOS if k in c}
            fila.update(id_config=nombre, semilla=semilla,
                        nota=round(float(nota), 5), val_mse=round(float(val), 5),
                        epocas=int(eps), segundos=segs,
                        dim_entrada=int(V.shape[1]), params=params)
            EN.guardar_mejor(
                args.salida, "%s_s%d" % (nombre, semilla),
                (nota, val, dict(c), media.cpu().numpy(), escala.cpu().numpy(),
                 {n: t.detach().cpu() for n, t in estado.items()}, V.shape[1]))
            fase4._anotar(ruta_csv, fila)
            for parcial in (checkpoint, checkpoint + ".json"):
                try:
                    os.remove(parcial)
                except FileNotFoundError:
                    pass
            print("[fase6] %s s%d: nota %.4f · %.1f min"
                  % (nombre, semilla, nota, segs / 60.0), flush=True)
            del red, estado, V
            torch.cuda.empty_cache()
    print("[fase6] no queda nada por hacer en este proceso", flush=True)


if __name__ == "__main__":
    main()
