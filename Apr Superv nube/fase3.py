#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FASE 3 · Reválida de las mejores configuraciones.

La fase 2 ordenó 15 ejes de hiperparámetros por una sola nota cada uno. Eso
sirve para explorar, pero no para elegir, y por dos motivos que se pueden medir
en el propio estudio:

  1. La nota es RUIDOSA. Veintiuna configuraciones se repitieron por casualidad
     durante la búsqueda, y entre repeticiones de la MISMA configuración la nota
     baila 0,008 de mediana y hasta 0,030. El récord (0,1118) le saca 0,0035 al
     segundo: menos que el ruido. Con una sola medición no se sabe si el campeón
     es mejor o simplemente tuvo mejor arranque.

  2. El presupuesto se quedaba corto. De la época 96 al final (160), 27 de las
     34 mejores SEGUÍAN mejorando, +0,0054 de mediana. Ninguna había llegado a
     su techo.

Así que aquí no se busca nada nuevo: se cogen las 12 mejores combinaciones tal
cual, se entrenan con TRES semillas distintas y el DOBLE de épocas, y se ordenan
por la MEDIA de sus tres notas. Lo que salga arriba lo estará por la
configuración, no por la suerte del arranque.

Sin podador: aquí interesa la nota final de todas, incluidas las que empiezan
despacio. Cada tarea (configuración, semilla) se reserva con un cerrojo de
fichero, así que se pueden lanzar varios procesos a la vez y se reparten solos.

Uso:
    python fase3.py --idt 0 --frac-vram 0.27      (uno de tres)
    python fase3.py --resumen
"""

import argparse
import csv
import glob
import hashlib
import json
import os
import time

import numpy as np

import comun
from comun import MODELOS_DIR, MUESTRAS_DIR, asegurar
import buscar_optuna as bo
import entrenar_nube as EN
import escenarios as esc

# Las 12 mejores COMBINACIONES distintas de la fase 2. Doce y no veinte porque
# cada una cuesta tres entrenamientos largos: con 3 procesos, 12x3 a 320 épocas
# son unas diez horas, que es lo que cabe en un día de trabajo.
N_CONFIGS = 12

# Tres semillas: la primera dice cuánto vale la configuración y las otras dos
# dicen cuánto de eso era suerte. Con dos no se distingue una mala racha de una
# mala configuración; con cinco se tardaría casi el doble para afinar un ruido
# que ya se ve con tres.
SEMILLAS = (0, 1, 2)

# El doble que en la fase 2 (160). No es un número redondo por gusto: las
# mejores seguían subiendo en el último tramo, así que el presupuesto era su
# techo. Si con 320 la mayoría vuelve a acabar subiendo, es que sigue siéndolo.
EPOCAS = 320

CARPETA = os.path.join(MODELOS_DIR, "fase3")

# Las reservas y los CSV viven DENTRO de la carpeta de salida, no en una ruta
# fija: así una prueba con --salida aparte no deja cerrojos puestos que luego
# hagan saltarse tareas de la tanda de verdad.
def _cerrojos(carpeta):
    return os.path.join(carpeta, "reservas")

# Los ejes que definen una configuración (los mismos que buscó la fase 2). Las
# épocas NO están: aquí las fija esta fase para todas por igual.
EJES = list(bo.ESPACIO.keys())


def id_config(params):
    """Nombre corto y estable de una combinación. Se usa para reservarla, para
    agrupar sus tres semillas y para nombrar su modelo."""
    texto = json.dumps({k: params[k] for k in EJES}, sort_keys=True)
    return hashlib.sha1(texto.encode()).hexdigest()[:8]


def mejores_configs(estudio, n=N_CONFIGS):
    """Las n combinaciones DISTINTAS con mejor nota de la fase 2.

    Si una se probó varias veces se la juzga por su MEJOR nota, que es como se
    la seleccionó en su momento; justamente lo que esta fase viene a revisar."""
    from optuna.trial import TrialState
    mejor = {}
    for t in estudio.get_trials(deepcopy=False):
        if t.state != TrialState.COMPLETE or t.value is None:
            continue
        if any(t.params.get(k) not in bo.ESPACIO[k] for k in EJES):
            continue
        cid = id_config(t.params)
        if cid not in mejor or t.value > mejor[cid][0]:
            mejor[cid] = (t.value, {k: t.params[k] for k in EJES})
    orden = sorted(mejor.items(), key=lambda kv: -kv[1][0])[:n]
    return [(cid, nota, params) for cid, (nota, params) in orden]


def _soltar(carpeta, cid, semilla):
    """Quita la reserva de una tarea que no se pudo hacer. Se usa solo con la
    falta de memoria: la configuración no es mala, es que no cabía con esta
    parte de la tarjeta, y una pasada posterior con un solo proceso (--frac-vram
    0.90) tiene que poder cogerla."""
    try:
        os.remove(os.path.join(_cerrojos(carpeta), "%s_s%d.lock" % (cid, semilla)))
    except OSError:
        pass


def limpiar_reservas(carpeta, hechas, verbose=True):
    """Borra las reservas que no dejaron fila en el CSV.

    Una tarea interrumpida a media (apagar el ordenador, el driver que se cae)
    deja su cerrojo puesto y su medida sin hacer: sin esto, al relanzar se
    saltaría para siempre. Solo se llama al arrancar una tanda, cuando por
    definición no hay nadie trabajando todavía."""
    n = 0
    for f in glob.glob(os.path.join(_cerrojos(carpeta), "*.lock")):
        base = os.path.basename(f)[:-len(".lock")]
        cid, _, semilla = base.rpartition("_s")
        try:
            if (cid, int(semilla)) in hechas:
                continue
        except ValueError:
            continue
        try:
            os.remove(f)
            n += 1
        except OSError:
            pass
    if verbose and n:
        print("[fase3] %d reservas a medias liberadas" % n, flush=True)
    return n


def _reservar(carpeta, cid, semilla):
    """True solo para el primer proceso que pide esta tarea. Mismo cerrojo de
    fichero (O_EXCL, atómico) que usa la siembra de la fase 2."""
    ruta_cerrojos = _cerrojos(carpeta)
    asegurar(ruta_cerrojos)
    ruta = os.path.join(ruta_cerrojos, "%s_s%d.lock" % (cid, semilla))
    try:
        fd = os.open(ruta, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except OSError:
        return False
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write("pid %d" % os.getpid())
    return True


CAMPOS = ["id_config", "semilla", "nota", "nota_fase2", "val_mse", "epocas",
          "segundos", "dim_entrada"] + EJES


def _leer_csv(carpeta=CARPETA, patron="fase3_t*.csv"):
    """Filas de los CSV de todos los procesos. El patrón es un argumento porque
    la fase 4 reutiliza este reparto de trabajo con sus propios ficheros."""
    filas = []
    for f in sorted(glob.glob(os.path.join(carpeta, patron))):
        with open(f, encoding="utf-8", newline="") as fh:
            filas.extend(list(csv.DictReader(fh)))
    return filas


def filas_hechas(carpeta=CARPETA, patron="fase3_t*.csv"):
    """{(id_config, semilla): nota} de lo ya medido. Permite reanudar tras un
    corte sin repetir trabajo."""
    hechas = {}
    for r in _leer_csv(carpeta, patron):
        try:
            hechas[(r["id_config"], int(r["semilla"]))] = float(r["nota"])
        except (KeyError, TypeError, ValueError):
            continue
    return hechas


def _anotar(ruta, fila):
    nuevo = not os.path.exists(ruta)
    with open(ruta, "a", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CAMPOS)
        if nuevo:
            w.writeheader()
        w.writerow(fila)


def resumen(carpeta=CARPETA):
    """Tabla por configuración, ordenada por la MEDIA de sus semillas, que es la
    cifra con la que se elige. Al lado, su nota de la fase 2: la diferencia
    entre las dos es exactamente lo que esta fase viene a medir."""
    import statistics as st
    notas, extra = {}, {}
    for r in _leer_csv(carpeta):
        try:
            notas.setdefault(r["id_config"], []).append(float(r["nota"]))
        except (KeyError, TypeError, ValueError):
            continue
        extra[r["id_config"]] = r
    if not notas:
        print("[fase3] todavía no hay ninguna medida")
        return
    filas = sorted(((st.mean(v), min(v), max(v), len(v), k)
                    for k, v in notas.items()), reverse=True)
    print("%-10s %7s %7s %7s %2s %7s  hiperparámetros"
          % ("config", "media", "peor", "mejor", "n", "fase2"))
    for media, mn, mx, n, cid in filas:
        r = extra[cid]
        print("%-10s %7.4f %7.4f %7.4f %2d %7.4f  %sx%s lote %s lr %s vec %s "
              "h %s hor %s %s drop %s wd %s %s %s f%s r%s"
              % (cid, media, mn, mx, n, float(r["nota_fase2"]),
                 r["n_capas"], r["oculto"], r["lote"], r["lr"], r["n_vecinos"],
                 r["h_pasado"], r["horizonte"], r["activacion"], r["dropout"],
                 r["weight_decay"], r["normalizacion"], r["mezcla"],
                 r["n_fourier"], r["n_rayos"]))


def construir_parser():
    ap = argparse.ArgumentParser(
        description="Reválida de las mejores configuraciones de la fase 2.")
    ap.add_argument("--muestras", default=MUESTRAS_DIR)
    ap.add_argument("--escenarios", default=esc.CARPETA)
    ap.add_argument("--salida", default=CARPETA)
    ap.add_argument("--estudio", default=bo.NOMBRE_ESTUDIO)
    ap.add_argument("--bd", default=None)
    ap.add_argument("--idt", type=int, default=0)
    ap.add_argument("--n-configs", dest="n_configs", type=int, default=N_CONFIGS)
    ap.add_argument("--epocas", type=int, default=EPOCAS)
    ap.add_argument("--n-escenarios", dest="n_escenarios", type=int, default=400)
    ap.add_argument("--frac-vram", dest="frac_vram", type=float, default=1.0)
    ap.add_argument("--en-cpu", dest="en_cpu", action="store_true")
    ap.add_argument("--enfasis", type=float, default=1.0)
    ap.add_argument("--resumen", action="store_true")
    ap.add_argument("--reanudar", action="store_true",
                    help="liberar las reservas que quedaron a medias (tras un "
                         "corte) antes de repartir el trabajo. Solo el PRIMER "
                         "proceso de la tanda debe llevarlo")
    ap.add_argument("--tf32", action="store_true", default=True)
    return ap


def main():
    args = construir_parser().parse_args()
    asegurar(args.salida)
    if args.resumen:
        resumen(args.salida)
        return

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    bd = args.bd or os.path.join(MODELOS_DIR, "optuna.db")
    estudio = optuna.load_study(study_name=args.estudio,
                                storage="sqlite:///%s" % bd)
    tareas = mejores_configs(estudio, args.n_configs)
    print("[fase3] %d configuraciones x %d semillas x %d épocas"
          % (len(tareas), len(SEMILLAS), args.epocas), flush=True)

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
    print("[fase3] proceso %d · %s · %s muestras · %d escenarios · carga %.1f s"
          % (args.idt, device, format(len(datos.X), ","), len(flotas),
             time.perf_counter() - t0), flush=True)

    ruta_csv = os.path.join(args.salida, "fase3_t%d.csv" % args.idt)
    hechas = filas_hechas(args.salida)
    if args.reanudar:
        limpiar_reservas(args.salida, hechas)
    sin_sitio = set()
    # La semilla por FUERA del bucle de configuraciones: así, si hay que parar a
    # media tanda, están las doce con una medida cada una —que ya ordena algo—
    # en vez de cuatro con tres y ocho sin tocar.
    for semilla in SEMILLAS:
        for cid, nota2, params in tareas:
            if ((cid, semilla) in hechas or (cid, semilla) in sin_sitio
                    or not _reservar(args.salida, cid, semilla)):
                continue
            c = dict(params)
            c["epocas"] = args.epocas
            # Lo único que cambia entre las tres pasadas: los pesos iniciales,
            # el dropout y el orden de los lotes salen todos de aquí.
            torch.manual_seed(semilla)
            torch.cuda.manual_seed_all(semilla)
            np.random.seed(semilla)
            ini = time.perf_counter()
            try:
                V, media, escala = datos.vista(c["n_vecinos"], c["horizonte"],
                                               c["h_pasado"],
                                               c.get("n_fourier", 0),
                                               c.get("n_rayos", 0))
                red, estado, val, eps = EN.entrenar_config(
                    V, datos, c, device, criba=None, enfasis=args.enfasis)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                # Se suelta la reserva para que la recoja una pasada con más
                # tarjeta por proceso, pero se apunta aquí para no volver a
                # intentarla en ESTA: si no cabe una vez, no va a caber luego
                # con los mismos vecinos, y los tres procesos se turnarían para
                # estrellarse con ella.
                _soltar(args.salida, cid, semilla)
                sin_sitio.add((cid, semilla))
                print("[fase3] %s s%d: no cabe en memoria" % (cid, semilla),
                      flush=True)
                continue
            nota = EN.nota_de(red, estado, media, escala, device, flotas, opts, c)
            segs = round(time.perf_counter() - ini, 1)
            _anotar(ruta_csv, dict(
                {k: params[k] for k in EJES},
                id_config=cid, semilla=semilla, nota=round(float(nota), 5),
                nota_fase2=round(float(nota2), 5), val_mse=round(float(val), 5),
                epocas=int(eps), segundos=segs, dim_entrada=int(V.shape[1])))
            print("[fase3] %s s%d: nota %.4f (fase 2: %.4f) · %.1f min"
                  % (cid, semilla, nota, nota2, segs / 60.0), flush=True)
            EN.guardar_mejor(args.salida, "%s_s%d" % (cid, semilla),
                             (nota, val, dict(c), media.cpu().numpy(),
                              escala.cpu().numpy(),
                              {n: t.detach().cpu() for n, t in estado.items()},
                              V.shape[1]))
            del red, estado, V
            torch.cuda.empty_cache()
    print("[fase3] no queda nada por hacer en este proceso", flush=True)


if __name__ == "__main__":
    main()
