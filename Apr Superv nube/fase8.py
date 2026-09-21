#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FASE 8 Y FINAL: evalúa una sola vez en test el mejor modelo de fase 6.

La configuración se elige por la media de sus tres semillas. Dentro de la
configuración ganadora se publica la instancia con mayor nota de selección; no
se vuelve a entrenar. El test queda protegido con un cerrojo permanente para
evitar repetirlo accidentalmente.

Uso:
    python fase8.py
    python fase8.py --resumen
"""

import argparse
import csv
import glob
import hashlib
import json
import os
import shutil
import statistics
import time

import comun
from comun import MODELOS_DIR, RAIZ_LOCAL, asegurar
import escenarios as esc
import fase6
import politica as pol
import vectorizado as vec
from entrenar import _puntuar_vehiculo


RESULTADO = "fase8_test.json"
CERROJO = "fase8_test.lock"


def _resultados_fase6(carpeta):
    resultados = {}
    for ruta in sorted(glob.glob(os.path.join(carpeta, fase6.PATRON))):
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
    return resultados


def elegir_modelo(carpeta):
    """Elige configuración por media y, dentro de ella, la mejor instancia."""
    resultados = _resultados_fase6(carpeta)
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
    print("[fase8] fase 6 completa; orden por media:")
    for media, desv, nombre, notas in resumen:
        print("  %s · media %.5f · desv %.5f · %s"
              % (nombre, media, desv,
                 " / ".join("%.5f" % n for n in notas)))

    media, desv, nombre, notas = resumen[0]
    semilla = max(fase6.SEMILLAS,
                  key=lambda s: resultados[(nombre, s)])
    nota = resultados[(nombre, semilla)]
    ruta = os.path.join(carpeta, "mejor_%s_s%d.pt" % (nombre, semilla))
    if not os.path.isfile(ruta):
        raise SystemExit("Falta el modelo elegido: %s" % ruta)
    return nombre, dict(configs[nombre]), media, desv, notas, semilla, nota, ruta


def _sha256(ruta):
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1024 * 1024), b""):
            h.update(bloque)
    return h.hexdigest()


def evaluar_test(punto, escenarios_carpeta):
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

    escenarios = esc.cargar("test", escenarios_carpeta)
    flotas, opts = esc.flotas(escenarios)
    if len(flotas) != len(escenarios):
        raise RuntimeError("Solo %d/%d escenarios de test son válidos"
                           % (len(flotas), len(escenarios)))
    obstaculos, mundo = esc.obstaculos()
    llegados = vec.rollout(
        flotas, politica, opts=opts, n_vec=cfg["n_vecinos"],
        horizonte=cfg["horizonte"], h_pasado=cfg["h_pasado"],
        obstaculos=obstaculos, mundo=mundo,
        n_fourier=cfg.get("n_fourier", 0), n_rayos=cfg.get("n_rayos", 0))
    vehiculos = [v for flota in flotas for v in flota]
    nota = sum(_puntuar_vehiculo(v) for v in vehiculos) / len(vehiculos)
    choques = sum(bool(getattr(v, "choque", False)) for v in vehiculos)
    return nota, sum(llegados), choques, len(vehiculos), len(flotas)


def _guardar_torch_atomico(punto, ruta):
    import torch
    asegurar(os.path.dirname(os.path.abspath(ruta)))
    temporal = ruta + ".tmp-%d" % os.getpid()
    try:
        torch.save(punto, temporal)
        os.replace(temporal, ruta)
    finally:
        if os.path.exists(temporal):
            os.remove(temporal)


def _copiar_atomico(origen, destino):
    asegurar(os.path.dirname(os.path.abspath(destino)))
    temporal = destino + ".tmp-%d" % os.getpid()
    try:
        shutil.copy2(origen, temporal)
        os.replace(temporal, destino)
    finally:
        if os.path.exists(temporal):
            os.remove(temporal)


def _mostrar_resultado(ruta):
    if not os.path.isfile(ruta):
        print("[fase8] el test todavía no se ha ejecutado")
        return
    with open(ruta, encoding="utf-8") as f:
        r = json.load(f)
    print("[fase8] TEST ya medido: nota %.5f · llegan %d/%d · chocan %d/%d"
          % (r["nota_test"], r["llegados"], r["vehiculos"],
             r["choques"], r["vehiculos"]))


def construir_parser():
    ap = argparse.ArgumentParser(
        description="Publica el mejor modelo de fase 6 y mide test una vez.")
    ap.add_argument("--fase6", default=fase6.CARPETA)
    ap.add_argument("--escenarios", default=esc.CARPETA)
    ap.add_argument("--salida",
                    default=os.path.join(MODELOS_DIR, "politica.pt"))
    ap.add_argument("--destino-local", default=os.path.join(
        RAIZ_LOCAL, "modelos", "politica.pt"))
    ap.add_argument("--resumen", action="store_true")
    return ap


def main():
    args = construir_parser().parse_args()
    import torch

    carpeta_salida = os.path.dirname(os.path.abspath(args.salida))
    resultado = os.path.join(carpeta_salida, RESULTADO)
    cerrojo = os.path.join(carpeta_salida, CERROJO)
    if args.resumen:
        _mostrar_resultado(resultado)
        return
    if os.path.exists(resultado) or os.path.exists(args.salida):
        raise SystemExit("La fase 8 ya tiene resultado; el test no se repite.")

    asegurar(carpeta_salida)
    try:
        fd = os.open(cerrojo, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise SystemExit("El test ya se abrió o está ejecutándose; no se repite.")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write("pid %d · %s\n" % (os.getpid(), time.strftime("%Y-%m-%d %H:%M:%S")))

    (nombre, hp, media_sel, desv_sel, notas_sel, semilla, nota_sel,
     origen) = elegir_modelo(args.fase6)
    punto = torch.load(origen, map_location="cpu", weights_only=False)
    guardados = punto.get("config", {}).get("hiperparametros", {})
    if any(guardados.get(k) != v for k, v in hp.items()):
        raise SystemExit("El .pt elegido no coincide con la configuración ganadora")
    huella = _sha256(origen)
    punto["config"].update(
        id_config_fase6=nombre,
        semilla_modelo=semilla,
        nota_seleccion_modelo=nota_sel,
        nota_seleccion_media=media_sel,
        nota_seleccion_desv=desv_sel,
        notas_seleccion=notas_sel,
        modelo_origen=os.path.basename(origen),
        sha256_origen=huella,
    )
    print("[fase8] modelo: %s s%d · selección %.5f" %
          (nombre, semilla, nota_sel), flush=True)

    nota, llegados, choques, n_veh, n_esc = evaluar_test(
        punto, args.escenarios)
    punto["config"].update(
        nota_test=nota,
        llegadas_test="%d/%d" % (llegados, n_veh),
        choques_test="%d/%d" % (choques, n_veh),
        escenarios_test=n_esc,
    )
    registro = {
        "fase": 8,
        "modelo_origen": os.path.abspath(origen),
        "sha256_origen": huella,
        "id_config_fase6": nombre,
        "semilla_modelo": semilla,
        "nota_seleccion_modelo": nota_sel,
        "nota_seleccion_media": media_sel,
        "nota_seleccion_desv": desv_sel,
        "notas_seleccion": notas_sel,
        "escenarios_test": n_esc,
        "vehiculos": n_veh,
        "nota_test": nota,
        "llegados": llegados,
        "choques": choques,
        "fecha": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    print("[fase8] TEST: %d escenarios · %d vehículos · nota %.5f · "
          "llegan %d/%d (%.1f %%) · chocan %d/%d (%.1f %%)"
          % (n_esc, n_veh, nota, llegados, n_veh,
             100.0 * llegados / max(1, n_veh), choques, n_veh,
             100.0 * choques / max(1, n_veh)), flush=True)

    _guardar_torch_atomico(punto, args.salida)
    if args.destino_local:
        _copiar_atomico(args.salida, args.destino_local)
    temporal_json = resultado + ".tmp-%d" % os.getpid()
    try:
        with open(temporal_json, "w", encoding="utf-8") as f:
            json.dump(registro, f, ensure_ascii=False, indent=2)
        os.replace(temporal_json, resultado)
    finally:
        if os.path.exists(temporal_json):
            os.remove(temporal_json)
    print("[fase8] modelo definitivo → %s" % args.salida)
    print("[fase8] copia para la interfaz → %s" % args.destino_local)

    destino = comun.bucket_env("modelos")
    if destino:
        comun.sincronizar(carpeta_salida, destino)


if __name__ == "__main__":
    main()
