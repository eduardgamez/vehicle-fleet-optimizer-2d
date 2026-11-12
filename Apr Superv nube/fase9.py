#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FASE 9 · Reválida tras arreglar el giro de la marcha atrás.

Todas las fases anteriores entrenaron con el giro de la marcha atrás desplazado
±π (ver `entrenar.corregir_giro`): el 18 % de las filas llevaba un objetivo de
giro de ~±4,9 en unidades normalizadas, y esas filas eran el 97 % de la energía
de la etiqueta. La pérdida estaba dominada por ellas, y la historia de controles
que entra a la red (h_pasado) llevaba el mismo valor roto. O sea que el ranking
de configuraciones de las fases 2-6 pudo salir torcido: no basta con reentrenar
la campeona.

Aquí se reentrenan DOCE configuraciones con los datos corregidos
(`datos/muestras_giro`), tres semillas y 320 épocas, como en la fase 3:

  · la campeona de la fase 6 (5x2048) y la ganadora de la fase 3 (4x2048);
  · seis de las 40 mejores de la fase 2, elegidas para ser lo más DISTINTAS
    posible entre sí (máxima distancia mínima en nº de ejes diferentes; se
    dejan fuera las de anchura 4096, que tardan el triple y ya iban peor);
  · cuatro variantes de la campeona en los ejes que el fallo tocaba de lleno:
    la historia de controles (3 y 10), el dropout y la tasa de aprendizaje.

El reparto de trabajo, las reservas y el CSV son los de la fase 3, tal cual. La
columna 'nota_fase2' lleva aquí la nota ANTERIOR al arreglo que se tenga de esa
configuración (media de la fase 3 o 6 si existe; si no, la de la fase 2; 0 si
es una variante nueva).

Uso (tres procesos, como en la fase 3):
    python fase9.py --idt 0 --frac-vram 0.27 --reanudar
    python fase9.py --idt 1 --frac-vram 0.27
    python fase9.py --idt 2 --frac-vram 0.27
    python fase9.py --resumen
"""

import os
import sys

from comun import DATOS_DIR, MODELOS_DIR
import fase3

CARPETA = os.path.join(MODELOS_DIR, "fase9")
MUESTRAS = os.path.join(DATOS_DIR, "muestras_giro")

CAMPEONA = dict(n_capas=5, oculto=2048, dropout=0.2, activacion="gelu",
                lr=0.003, lote=2048, n_vecinos=3, h_pasado=1, horizonte=20,
                weight_decay=0.01, normalizacion="layernorm",
                optimizador="adamw", mezcla="raros_x4", n_fourier=0,
                n_rayos=18)


def _c(**cambios):
    c = dict(CAMPEONA)
    c.update(cambios)
    return c


# (nota anterior al arreglo, configuración)
CONFIGS = [
    (0.1251, _c()),                                   # campeona fase 6 (a 640)
    (0.1130, _c(n_capas=4)),                          # 2fcab7e2, ganadora fase 3
    # Seis de las 40 mejores de la fase 2, lo más distintas posible.
    (0.0971, _c(n_capas=3, activacion="silu", lr=0.001, n_vecinos=4,
                h_pasado=2, horizonte=10, weight_decay=0.0, n_rayos=32)),
    (0.1002, _c(n_capas=3, lote=512, n_vecinos=5, horizonte=10,
                weight_decay=0.0, n_fourier=12, n_rayos=12)),
    (0.1005, _c(n_capas=4, activacion="relu", lr=0.001, n_vecinos=5,
                horizonte=10, weight_decay=0.0, mezcla="natural")),
    (0.0806, _c(n_capas=3, oculto=512, h_pasado=2, weight_decay=0.0,
                n_fourier=12)),
    (0.1086, _c(n_capas=3, lr=0.008, n_vecinos=7, horizonte=10, n_fourier=6)),
    (0.1004, _c(n_capas=4, dropout=0.1, activacion="relu", h_pasado=2,
                weight_decay=0.0, mezcla="cap_rec")),
    # Variantes de la campeona en los ejes que el fallo distorsionaba.
    (0.0, _c(h_pasado=3)),
    (0.0, _c(h_pasado=10)),
    (0.0, _c(dropout=0.35)),
    (0.0, _c(lr=0.001)),
]


def tareas(_estudio=None, _n=None):
    return [(fase3.id_config(c), nota, c) for nota, c in CONFIGS]


if __name__ == "__main__":
    fase3.mejores_configs = tareas
    if "--salida" not in sys.argv:
        sys.argv += ["--salida", CARPETA]
    if "--muestras" not in sys.argv:
        sys.argv += ["--muestras", MUESTRAS]
    fase3.main()
