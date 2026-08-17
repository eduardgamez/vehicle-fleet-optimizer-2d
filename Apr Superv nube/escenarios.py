#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Escenarios de EVALUACIÓN: situaciones iniciales nuevas, sin rutas.

Para puntuar una red no hace falta que el planificador clásico resuelva el
escenario: lo que se mide es si la red lleva su flota a las metas, no si copia
una ruta concreta. Así que un escenario de evaluación es solo su situación de
partida (poses, metas, tamaños, límites, grupo/prioridad y modo de
optimización), y generarlo cuesta milisegundos en vez de minutos.

Se generan DOS conjuntos con semillas disjuntas entre sí y del entrenamiento:

  · selección — con el que el barrido decide qué configuración gana. Se consulta
    miles de veces, así que la ganadora está algo «ajustada» a él por el simple
    hecho de haber sido elegida como la mejor de miles.
  · test      — no interviene en ninguna decisión. Se usa UNA vez, al final,
    sobre el modelo ganador: es la cifra honesta que se puede publicar.

Uso:
    python escenarios.py --n 400            (crea los dos conjuntos)
    python escenarios.py --n 400 --conjunto test
"""

import argparse
import json
import os
import random

import comun
from comun import DATOS_DIR, MAPA_ENTRENAMIENTO, asegurar

from nucleo import (Entorno, cargar_mapa_en, espec_a_vehiculo,
                    parsear_especificaciones)
from generador import aleatorios_spec
from generador_nube import MEZCLA, elegir_modo

# Rangos de semilla reservados. El entrenamiento arranca en 0 y crece hacia
# arriba; estos están tan lejos que no se alcanzan ni con cientos de millones de
# escenarios, así que ningún caso de evaluación puede haberse entrenado.
BASES = {"seleccion": 900_000_000, "test": 950_000_000}

CARPETA = os.path.join(DATOS_DIR, "escenarios")


def ruta_conjunto(conjunto, carpeta=None):
    return os.path.join(carpeta or CARPETA, f"{conjunto}.json")


# --------------------------------------------------------------------------- #
# Generación
# --------------------------------------------------------------------------- #
def generar(n, conjunto, veh_min=2, veh_max=6, opt="mixto",
            mapa=MAPA_ENTRENAMIENTO, mezcla=MEZCLA):
    """Lista de escenarios {semilla, opt, especs}. El sorteo es EL MISMO que el
    del generador de entrenamiento (mismo mapa, mismos rangos de parámetros,
    mismo reparto de nº de vehículos y misma mezcla de modos), solo cambia el
    rango de semillas: los casos de evaluación son nuevos, pero no son más
    fáciles ni más difíciles."""
    env = Entorno()
    cargar_mapa_en(env, mapa)
    base = BASES[conjunto]
    fuera = []
    sem = base
    while len(fuera) < n:
        random.seed(sem)
        nv = random.randint(veh_min, veh_max)
        modo = elegir_modo(sem, opt, mezcla)
        especs = aleatorios_spec(env, nv)
        sem += 1
        if especs is None:
            continue
        fuera.append({"semilla": sem - 1, "opt": modo, "especs": especs})
    return fuera


def guardar(escenarios, conjunto, carpeta=None):
    ruta = ruta_conjunto(conjunto, carpeta)
    asegurar(os.path.dirname(ruta))
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(escenarios, f)
    return ruta


def cargar(conjunto, carpeta=None):
    ruta = ruta_conjunto(conjunto, carpeta)
    if not os.path.exists(ruta):
        raise SystemExit(f"No existe {ruta}. Créalo con: python escenarios.py")
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# De escenario a flota de Vehiculo (lo que consume el rollout)
# --------------------------------------------------------------------------- #
def obstaculos(mapa=MAPA_ENTRENAMIENTO):
    """(esquinas, círculos, tamaño del mundo) de los obstáculos fijos del mapa,
    en el formato que consume la comprobación de choques del simulador rápido.

    Se saca una sola vez y se reutiliza en todas las evaluaciones: el mapa no
    cambia nunca (la red se entrena y se usa siempre sobre el mismo)."""
    from nucleo import W as ANCHO_MUNDO, H as ALTO_MUNDO
    env = Entorno()
    cargar_mapa_en(env, mapa)
    return (env.np_c, env.np_bb), (ANCHO_MUNDO, ALTO_MUNDO)


def flotas(escenarios, mapa=MAPA_ENTRENAMIENTO, limite=None):
    """(lista de flotas, lista de modos de optimización). Los vehículos se crean
    una sola vez y se reutilizan en todas las evaluaciones: el rollout solo
    sobrescribe su trayectoria y su bandera de misión."""
    env = Entorno()
    cargar_mapa_en(env, mapa)
    out, opts = [], []
    for esc in escenarios[:limite]:
        normal = parsear_especificaciones(repr(esc["especs"]))
        try:
            flota = [espec_a_vehiculo(e, i, env) for i, e in enumerate(normal)]
        except ValueError:
            continue            # pose no válida en este mapa: se descarta
        out.append(flota)
        opts.append(esc["opt"])
    return out, opts


def main():
    ap = argparse.ArgumentParser(
        description="Genera los escenarios de evaluación (sin rutas).")
    ap.add_argument("--n", type=int, default=400,
                    help="escenarios por conjunto (def. 400)")
    ap.add_argument("--conjunto", default="ambos",
                    choices=("ambos", "seleccion", "test"))
    ap.add_argument("--veh", default="2-6")
    ap.add_argument("--opt", default="mixto",
                    choices=("secuencial", "global", "prioridades", "mixto"))
    ap.add_argument("--mapa", default=MAPA_ENTRENAMIENTO)
    ap.add_argument("--carpeta", default=CARPETA)
    args = ap.parse_args()

    veh_min, _, veh_max = args.veh.partition("-")
    veh_min, veh_max = max(1, int(veh_min)), int(veh_max or veh_min)
    conjuntos = (("seleccion", "test") if args.conjunto == "ambos"
                 else (args.conjunto,))
    for c in conjuntos:
        escs = generar(args.n, c, veh_min, veh_max, args.opt, args.mapa)
        ruta = guardar(escs, c, args.carpeta)
        veh = sum(len(e["especs"]) for e in escs)
        print(f"[esc] {c}: {len(escs)} escenarios · {veh} vehículos → {ruta}")

    destino = comun.bucket_env("escenarios")
    if destino:
        comun.sincronizar(args.carpeta, destino)


if __name__ == "__main__":
    main()
