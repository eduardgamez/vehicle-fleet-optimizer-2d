#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FASE 3 · Modelo definitivo: reentreno con TODOS los datos y medida honesta.

Cada tarea del barrido deja su registro (barrido_tNNN.csv) y su mejor red
(mejor_tNNN.pt). Aquí se juntan todos y se coge la configuración ganadora por
nota de SELECCIÓN. Después:

  · se REENTRENA esa configuración con el dataset completo (durante el barrido
    se usa solo una parte, porque para ordenar configuraciones entre sí no hace
    falta todo y sale carísimo);
  · se le pasa —una sola vez— el conjunto de TEST, que no ha intervenido en
    ninguna decisión.

Esa separación importa: elegir la mejor de miles de configuraciones por su nota
en un conjunto ya sesga esa nota al alza (se está eligiendo, en parte, a quien
tuvo suerte con esos escenarios). La cifra que vale es la del test.

Uso:
    python finalizar.py --modelos datos/modelos              (reentrena)
    python finalizar.py --sin-reentrenar                     (solo medir)
"""

import argparse
import csv
import glob
import os
import time

import comun
from comun import MODELOS_DIR, MUESTRAS_DIR, asegurar
import escenarios as esc
import politica as pol
import vectorizado as vec
from entrenar import _puntuar_vehiculo


def juntar_registros(carpeta):
    """Une los CSV de todas las tareas en barrido.csv y devuelve las filas."""
    partes = sorted(glob.glob(os.path.join(carpeta, "barrido_t*.csv")))
    filas, campos = [], None
    for p in partes:
        with open(p, encoding="utf-8", newline="") as f:
            lector = csv.DictReader(f)
            campos = campos or lector.fieldnames
            filas.extend(lector)
    if filas:
        destino = os.path.join(carpeta, "barrido.csv")
        with open(destino, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=campos)
            w.writeheader()
            w.writerows(filas)
        print(f"[final] {len(filas)} configuraciones de {len(partes)} tareas "
              f"→ {destino}")
    return filas


def mejor_modelo(carpeta):
    """(ruta, punto) del .pt con mejor nota de selección de entre los que ha
    dejado cada tarea."""
    import torch
    mejor = None
    for ruta in sorted(glob.glob(os.path.join(carpeta, "mejor_t*.pt"))):
        punto = torch.load(ruta, map_location="cpu", weights_only=False)
        nota = punto["config"].get("nota_rollout")
        if nota is not None and (mejor is None or nota > mejor[2]):
            mejor = (ruta, punto, nota)
    if mejor is None:
        raise SystemExit(f"No hay ningún mejor_t*.pt en {carpeta}.")
    return mejor


def evaluar(punto, escenarios_carpeta, conjunto, n=None):
    """Nota media por vehículo del modelo sobre un conjunto de escenarios."""
    import torch
    from politica import Politica, crear_red
    cfg = punto["config"]
    pol.configurar_representacion(cfg["n_vecinos"], cfg["horizonte"],
                                  cfg["h_pasado"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    red = crear_red(cfg["dim_entrada"], cfg["oculto"], cfg["n_pred"],
                    cfg["n_capas"], cfg["dropout"], cfg["activacion"],
                    cfg.get("normalizacion", "no"))
    red.load_state_dict(punto["state_dict"])
    red.to(device).eval()
    politica = Politica(
        red, torch.tensor(punto["media"], dtype=torch.float32, device=device),
        torch.tensor(punto["escala"], dtype=torch.float32, device=device),
        device)
    flotas, opts = esc.flotas(esc.cargar(conjunto, escenarios_carpeta), limite=n)
    llegados = vec.rollout(flotas, politica, opts=opts,
                                   n_vec=cfg["n_vecinos"],
                                   horizonte=cfg["horizonte"],
                                   h_pasado=cfg["h_pasado"])
    total = sum(_puntuar_vehiculo(v) for f in flotas for v in f)
    n_veh = sum(len(f) for f in flotas)
    return (total / n_veh, sum(llegados), n_veh, len(flotas))


def reentrenar(hp, muestras, device, en_cpu, max_muestras):
    """Entrena la configuración ganadora con el dataset COMPLETO y devuelve un
    'punto' con el mismo formato que los .pt del barrido."""
    import torch
    from entrenar_nube import Datos, entrenar_config, guardar_mejor  # noqa: F401
    import entrenar as ent

    t0 = time.perf_counter()
    datos = Datos(muestras, device, fraccion=1.0, max_muestras=max_muestras,
                  en_cpu=en_cpu)
    print(f"[final] reentrenando con {len(datos.X):,} muestras "
          f"({len(datos.idx_tr):,} de entrenamiento) · carga "
          f"{time.perf_counter() - t0:.0f} s", flush=True)
    V, media, escala = datos.vista(hp["n_vecinos"], hp["horizonte"],
                                   hp["h_pasado"])
    red, estado, val, eps = entrenar_config(V, datos, hp, device)
    print(f"[final] reentrenada: {eps} épocas · val {val:.5f} · "
          f"{(time.perf_counter() - t0) / 60:.1f} min", flush=True)

    pol.configurar_representacion(hp["n_vecinos"], hp["horizonte"],
                                  hp["h_pasado"])
    arq = {"oculto": hp["oculto"], "n_capas": hp["n_capas"],
           "dropout": hp["dropout"], "activacion": hp["activacion"],
           "normalizacion": hp.get("normalizacion", "no")}
    cfg = ent._cfg(arq, None, val)
    cfg["hiperparametros"] = dict(hp)
    cfg["muestras_finales"] = len(datos.X)
    return {"config": cfg, "media": media.cpu().numpy().tolist(),
            "escala": escala.cpu().numpy().tolist(),
            "state_dict": {n: t.detach().cpu() for n, t in estado.items()}}


def main():
    ap = argparse.ArgumentParser(
        description="Reentrena la configuración ganadora con todos los datos y "
                    "la mide con escenarios nuevos.")
    ap.add_argument("--modelos", default=MODELOS_DIR)
    ap.add_argument("--muestras", default=MUESTRAS_DIR)
    ap.add_argument("--escenarios", default=esc.CARPETA)
    ap.add_argument("--salida", default=None,
                    help="ruta del modelo final (def. <modelos>/politica.pt)")
    ap.add_argument("--sin-reentrenar", dest="sin_reentrenar",
                    action="store_true",
                    help="usa los pesos del barrido tal cual, sin volver a "
                         "entrenar con el dataset completo")
    ap.add_argument("--max-muestras", dest="max_muestras", type=int,
                    default=100_000_000)
    ap.add_argument("--en-cpu", dest="en_cpu", action="store_true",
                    help="datos en memoria del ordenador en vez de en la GPU")
    args = ap.parse_args()

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    juntar_registros(args.modelos)
    ruta, punto, nota_sel = mejor_modelo(args.modelos)
    cfg = punto["config"]
    hp = cfg.get("hiperparametros")
    print(f"[final] ganador: {os.path.basename(ruta)} · nota de selección "
          f"{nota_sel:.4f} · val {cfg.get('val_mse')}")
    print(f"[final] hiperparámetros: {hp or cfg}")

    if not args.sin_reentrenar and hp:
        punto = reentrenar(hp, args.muestras, device, args.en_cpu,
                           args.max_muestras)
        punto["config"]["nota_seleccion_barrido"] = nota_sel

    nota, llegados, n_veh, n_esc = evaluar(punto, args.escenarios, "test")
    print(f"[final] TEST ({n_esc} escenarios nuevos, {n_veh} vehículos): "
          f"nota {nota:.4f} · llegan {llegados}/{n_veh} "
          f"({100.0 * llegados / max(1, n_veh):.1f} %)")

    cfg["nota_test"] = nota
    cfg["llegadas_test"] = f"{llegados}/{n_veh}"
    salida = args.salida or os.path.join(args.modelos, "politica.pt")
    asegurar(os.path.dirname(salida))
    torch.save(punto, salida)
    print(f"[final] modelo final → {salida}")
    print("[final] cópialo a «Apr Superv local/modelos/politica.pt» para usarlo "
          "en la interfaz.")

    destino = comun.bucket_env("modelos")
    if destino:
        comun.sincronizar(args.modelos, destino)


if __name__ == "__main__":
    main()
