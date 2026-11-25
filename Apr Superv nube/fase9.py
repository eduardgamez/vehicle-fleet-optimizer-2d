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
                weight_decay=0.0, n_fourier=12)),    # era n_rayos=12, que
    #                         ya no está en el superset (18/22/27/32): va con 18
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


def ruta_checkpoint(carpeta, cid, semilla):
    return os.path.join(carpeta, "checkpoints",
                        "%s_s%d.checkpoint.pt" % (cid, semilla))


def main():
    """El bucle de la fase 3, con un añadido: cada entrenamiento guarda su
    estado completo cada pocas épocas. El ordenador puede apagarse en cualquier
    momento; al relanzar (fase9.bat), cada tarea a medias continúa por la época
    siguiente a la última guardada, con los mismos números aleatorios, en vez de
    empezar de cero."""
    import time
    import numpy as np
    import torch
    import entrenar_nube as EN
    import escenarios as esc

    ap = fase3.construir_parser()
    ap.add_argument("--cada-checkpoint", dest="cada_checkpoint", type=int,
                    default=5, help="épocas entre guardados (def. 5)")
    ap.set_defaults(salida=CARPETA, muestras=MUESTRAS)
    args = ap.parse_args()
    fase3.asegurar(args.salida)
    if args.resumen:
        fase3.resumen(args.salida)
        return

    lista = tareas()
    print("[fase9] %d configuraciones x %d semillas x %d épocas"
          % (len(lista), len(fase3.SEMILLAS), args.epocas), flush=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and 0 < args.frac_vram < 1.0:
        torch.cuda.set_per_process_memory_fraction(args.frac_vram)

    datos = EN.Datos(args.muestras, device, 1.0, None, en_cpu=args.en_cpu)
    flotas, opts = esc.flotas(esc.cargar("seleccion", args.escenarios),
                              limite=args.n_escenarios)
    ruta_csv = os.path.join(args.salida, "fase3_t%d.csv" % args.idt)
    hechas = fase3.filas_hechas(args.salida)
    if args.reanudar:
        fase3.limpiar_reservas(args.salida, hechas)
    sin_sitio = set()
    for semilla in fase3.SEMILLAS:
        for cid, nota_prev, params in lista:
            if ((cid, semilla) in hechas or (cid, semilla) in sin_sitio
                    or not fase3._reservar(args.salida, cid, semilla)):
                continue
            c = dict(params)
            c["epocas"] = args.epocas
            torch.manual_seed(semilla)
            torch.cuda.manual_seed_all(semilla)
            np.random.seed(semilla)
            ini = time.perf_counter()
            checkpoint = ruta_checkpoint(args.salida, cid, semilla)
            try:
                V, media, escala = datos.vista(c["n_vecinos"], c["horizonte"],
                                               c["h_pasado"], c["n_fourier"],
                                               c["n_rayos"])
                red, estado, val, eps = EN.entrenar_config(
                    V, datos, c, device, criba=None, enfasis=args.enfasis,
                    checkpoint=checkpoint,
                    cada_checkpoint=args.cada_checkpoint)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                fase3._soltar(args.salida, cid, semilla)
                sin_sitio.add((cid, semilla))
                print("[fase9] %s s%d: no cabe en memoria" % (cid, semilla),
                      flush=True)
                continue
            fin = time.perf_counter()
            nota = EN.nota_de(red, estado, media, escala, device, flotas, opts, c)
            # El tiempo cuenta también las sesiones anteriores al apagón.
            segs = round(getattr(red, "_segundos_entrenamiento", fin - ini)
                         + time.perf_counter() - fin, 1)
            # Primero el modelo y luego la fila: la fila es lo que da la tarea
            # por hecha, así que un corte entre las dos repite solo la nota.
            EN.guardar_mejor(args.salida, "%s_s%d" % (cid, semilla),
                             (nota, val, dict(c), media.cpu().numpy(),
                              escala.cpu().numpy(),
                              {n: t.detach().cpu() for n, t in estado.items()},
                              V.shape[1]))
            fase3._anotar(ruta_csv, dict(
                {k: params[k] for k in fase3.EJES},
                id_config=cid, semilla=semilla, nota=round(float(nota), 5),
                nota_fase2=round(float(nota_prev), 5),
                val_mse=round(float(val), 5), epocas=int(eps), segundos=segs,
                dim_entrada=int(V.shape[1])))
            for parcial in (checkpoint, checkpoint + ".json",
                            checkpoint + ".prev"):
                try:
                    os.remove(parcial)
                except FileNotFoundError:
                    pass
            print("[fase9] %s s%d: nota %.4f (antes: %.4f) · %.1f min"
                  % (cid, semilla, nota, nota_prev, segs / 60.0), flush=True)
            del red, estado, V
            torch.cuda.empty_cache()
    print("[fase9] no queda nada por hacer en este proceso", flush=True)


if __name__ == "__main__":
    main()
