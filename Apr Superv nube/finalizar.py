#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Entrena y evalúa el modelo definitivo después de la fase 6.

La configuración se elige por la media de sus tres semillas independientes de
fase 6. Después se entrena desde cero con la semilla fija 4 y se abre el conjunto
de test una sola vez. El script se niega a empezar si falta alguna medida o si
el fichero final ya contiene una nota de test.

Uso:
    python finalizar.py
    python finalizar.py --en-cpu
"""

import argparse
import csv
import glob
import os
import statistics
import time

import numpy as np

import comun
from comun import MODELOS_DIR, MUESTRAS_DIR, asegurar
import escenarios as esc
import fase6
import politica as pol
import vectorizado as vec
from entrenar import _puntuar_vehiculo


SEMILLA_FINAL = 4


def elegir_configuracion(carpeta):
    """Devuelve (nombre, config, media, desviación, notas).

    Solo acepta la matriz completa definida en fase6: cuatro candidatas por sus
    tres semillas. No usa la semilla 0 con la que se escogieron las candidatas.
    """
    resultados = {}
    patron = os.path.join(carpeta, fase6.PATRON)
    for ruta in sorted(glob.glob(patron)):
        with open(ruta, encoding="utf-8", newline="") as fh:
            for fila in csv.DictReader(fh):
                try:
                    clave = (fila["id_config"], int(fila["semilla"]))
                    nota = float(fila["nota"])
                except (KeyError, TypeError, ValueError):
                    continue
                if clave in resultados and resultados[clave] != nota:
                    raise SystemExit("Resultado duplicado y distinto: %s s%d"
                                     % clave)
                resultados[clave] = nota

    configs = dict(fase6.configs())
    esperados = {(nombre, semilla) for nombre in configs
                 for semilla in fase6.SEMILLAS}
    faltan = sorted(esperados - set(resultados))
    if faltan:
        detalle = ", ".join("%s s%d" % x for x in faltan)
        raise SystemExit("Fase 6 incompleta; faltan %d medidas: %s"
                         % (len(faltan), detalle))

    resumen = []
    for nombre in configs:
        notas = [resultados[(nombre, s)] for s in fase6.SEMILLAS]
        resumen.append((statistics.mean(notas), statistics.pstdev(notas),
                        nombre, notas))
    resumen.sort(reverse=True)
    print("[final] fase 6 completa; orden por media:")
    for media, desv, nombre, notas in resumen:
        print("  %s · media %.5f · desv %.5f · %s"
              % (nombre, media, desv,
                 " / ".join("%.5f" % n for n in notas)))
    media, desv, nombre, notas = resumen[0]
    return nombre, dict(configs[nombre]), media, desv, notas


def reentrenar(hp, muestras, device, en_cpu, max_muestras, semilla):
    import torch
    import entrenar as ent
    from entrenar_nube import Datos, entrenar_config

    torch.manual_seed(semilla)
    torch.cuda.manual_seed_all(semilla)
    np.random.seed(semilla)

    t0 = time.perf_counter()
    datos = Datos(muestras, device, fraccion=1.0,
                  max_muestras=max_muestras, en_cpu=en_cpu)
    print("[final] entrenando con %s muestras (%s de entrenamiento) · carga %.1f s"
          % (format(len(datos.X), ","), format(len(datos.idx_tr), ","),
             time.perf_counter() - t0), flush=True)
    V, media, escala = datos.vista(
        hp["n_vecinos"], hp["horizonte"], hp["h_pasado"],
        hp.get("n_fourier", 0), hp.get("n_rayos", 0))
    red, estado, val, eps = entrenar_config(V, datos, hp, device,
                                            criba=None)
    print("[final] entrenamiento terminado: %d épocas · val %.5f · %.1f min"
          % (eps, val, (time.perf_counter() - t0) / 60.0), flush=True)

    pol.configurar_representacion(
        hp["n_vecinos"], hp["horizonte"], hp["h_pasado"],
        hp.get("n_fourier", 0), hp.get("lambda_fina"),
        hp.get("n_rayos", 0))
    arq = {
        "oculto": hp["oculto"],
        "n_capas": hp["n_capas"],
        "dropout": hp["dropout"],
        "activacion": hp["activacion"],
        "normalizacion": hp.get("normalizacion", "no"),
        "residual": hp.get("residual", False),
    }
    cfg = ent._cfg(arq, None, val)
    cfg["hiperparametros"] = dict(hp)
    cfg["muestras_finales"] = len(datos.X)
    cfg["epocas_hechas"] = int(eps)
    cfg["semilla_final"] = int(semilla)
    return {
        "config": cfg,
        "media": media.cpu().numpy().tolist(),
        "escala": escala.cpu().numpy().tolist(),
        "state_dict": {n: t.detach().cpu() for n, t in estado.items()},
    }


def evaluar_test(punto, escenarios_carpeta, n=None):
    import torch
    from politica import Politica, crear_red

    cfg = punto["config"]
    pol.configurar_representacion(
        cfg["n_vecinos"], cfg["horizonte"], cfg["h_pasado"],
        cfg.get("n_fourier", 0), cfg.get("lambda_fina"),
        cfg.get("n_rayos", 0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    red = crear_red(
        cfg["dim_entrada"], cfg["oculto"], cfg["n_pred"],
        cfg["n_capas"], cfg["dropout"], cfg["activacion"],
        cfg.get("normalizacion", "no"), cfg.get("residual", False))
    red.load_state_dict(punto["state_dict"])
    red.to(device).eval()
    politica = Politica(
        red, torch.tensor(punto["media"], dtype=torch.float32, device=device),
        torch.tensor(punto["escala"], dtype=torch.float32, device=device),
        device)

    flotas, opts = esc.flotas(esc.cargar("test", escenarios_carpeta), limite=n)
    obstaculos, mundo = esc.obstaculos()
    llegados = vec.rollout(
        flotas, politica, opts=opts, n_vec=cfg["n_vecinos"],
        horizonte=cfg["horizonte"], h_pasado=cfg["h_pasado"],
        obstaculos=obstaculos, mundo=mundo,
        n_fourier=cfg.get("n_fourier", 0), n_rayos=cfg.get("n_rayos", 0))
    vehiculos = [v for f in flotas for v in f]
    nota = sum(_puntuar_vehiculo(v) for v in vehiculos) / len(vehiculos)
    choques = sum(bool(getattr(v, "choque", False)) for v in vehiculos)
    return nota, sum(llegados), choques, len(vehiculos), len(flotas)


def construir_parser():
    ap = argparse.ArgumentParser(
        description="Entrena el ganador de fase 6 y mide test una vez.")
    ap.add_argument("--fase6", default=fase6.CARPETA,
                    help="carpeta con los CSV de fase 6")
    ap.add_argument("--muestras", default=MUESTRAS_DIR)
    ap.add_argument("--escenarios", default=esc.CARPETA)
    ap.add_argument("--salida", default=os.path.join(MODELOS_DIR,
                                                      "politica.pt"))
    ap.add_argument("--semilla", type=int, default=SEMILLA_FINAL)
    ap.add_argument("--max-muestras", dest="max_muestras", type=int,
                    default=100_000_000)
    ap.add_argument("--en-cpu", dest="en_cpu", action="store_true",
                    help="mantener las muestras en RAM en vez de VRAM")
    return ap


def main():
    args = construir_parser().parse_args()
    import torch

    if os.path.exists(args.salida):
        try:
            anterior = torch.load(args.salida, map_location="cpu",
                                  weights_only=False)
            if anterior.get("config", {}).get("nota_test") is not None:
                raise SystemExit("El modelo final ya tiene nota de test; no se "
                                 "vuelve a abrir el conjunto.")
        except (OSError, RuntimeError, EOFError):
            pass
        raise SystemExit("Ya existe %s; indica otra --salida para no pisarlo."
                         % args.salida)

    nombre, hp, media_sel, desv_sel, notas_sel = elegir_configuracion(args.fase6)
    print("[final] ganadora: %s · semilla final %d" % (nombre, args.semilla))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    punto = reentrenar(hp, args.muestras, device, args.en_cpu,
                       args.max_muestras, args.semilla)
    punto["config"].update(
        id_config_fase6=nombre,
        nota_seleccion_media=media_sel,
        nota_seleccion_desv=desv_sel,
        notas_seleccion=notas_sel,
    )

    nota, llegados, choques, n_veh, n_esc = evaluar_test(
        punto, args.escenarios)
    punto["config"].update(
        nota_test=nota,
        llegadas_test="%d/%d" % (llegados, n_veh),
        choques_test="%d/%d" % (choques, n_veh),
        escenarios_test=n_esc,
    )
    print("[final] TEST: %d escenarios · %d vehículos · nota %.5f · "
          "llegan %d/%d (%.1f %%) · chocan %d/%d (%.1f %%)"
          % (n_esc, n_veh, nota, llegados, n_veh,
             100.0 * llegados / max(1, n_veh), choques, n_veh,
             100.0 * choques / max(1, n_veh)))

    carpeta_salida = os.path.dirname(os.path.abspath(args.salida))
    asegurar(carpeta_salida)
    torch.save(punto, args.salida)
    print("[final] modelo definitivo → %s" % args.salida)
    print("[final] para usarlo en la interfaz, cópialo a "
          "«Apr Superv local/modelos/politica.pt»")

    destino = comun.bucket_env("modelos")
    if destino:
        comun.sincronizar(carpeta_salida, destino)


if __name__ == "__main__":
    main()
