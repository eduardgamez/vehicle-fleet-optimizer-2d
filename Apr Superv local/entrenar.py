#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entrenamiento de la política neuronal a partir de los CSV de rutas.

Lee TODOS los .csv bajo la carpeta rutas/ (recursivo: incluye los clasificados
por nº de vehículos que escribe generador.py y el dataset acumulado de la GUI),
reconstruye por cada instante el vector de entrada definido en politica.py
(estado propio + meta + controles pasados + vecinos) y el objetivo (los
próximos N_PRED pares (a, giro) normalizados), y entrena el MLP con MSE.

La validación se separa POR RUN (escenario completo), no por fila, para no
contaminar la validación con instantes vecinos de las mismas rutas.

Requiere:   pip install torch numpy
Ejecutar:   python entrenar.py                  (usa rutas/ y 30 épocas)
            python entrenar.py --epocas 60 --oculto 1024
"""

import argparse
import ast
import glob
import itertools
import math
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
import sys
import time
import zlib

# La consola de Windows suele ser cp1252 y no puede imprimir algunos símbolos;
# forzar UTF-8 en la salida evita que un print tumbe el entrenamiento.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import numpy as np

from nucleo import DT, RUTAS_DIR
# 'pol' se usa para leer DIM_ENTRADA / H_PASADO / N_PRED de forma DINÁMICA: el
# barrido reconfigura la representación (nº de vecinos, historia…) en caliente y
# esos globales cambian, así que hay que consultarlos por el módulo, no fijarlos
# en el import.
import politica as pol
from politica import MODELO_PT, vector_entrada, crear_red


# --------------------------------------------------------------------------- #
# Lectura de los CSV
# --------------------------------------------------------------------------- #
def _parsear_condiciones(linea):
    """De la línea '# run=... condiciones_iniciales=[...]' devuelve el dict
    id → parámetros del vehículo (ángulos ya en RADIANES, meta como tupla)."""
    lista = ast.literal_eval(linea.split("condiciones_iniciales=", 1)[1])
    conds = {}
    for d in lista:
        ang = d.get("angulo_llegada")
        conds[d["id"]] = {
            "meta": (float(d["meta"][0]), float(d["meta"][1])),
            "meta_th": (None if ang in (None, "libre")
                        else math.radians(float(ang))),
            "largo": float(d["largo"]), "ancho": float(d["ancho"]),
            "v_max": float(d["v_max"]), "a_max": float(d["a_max"]),
            "giro_max": math.radians(float(d["giro_max"])),
            "grupo": int(d.get("grupo", 1)),
            "prioridad": int(d.get("prioridad", 1)),
        }
    return conds


def leer_runs(carpeta):
    """Recorre los CSV y devuelve una lista de runs:
    (clave_run, condiciones, modo_opt, {vid: array (n, 6) de [x, y, th, v, a, giro]})."""
    archivos = sorted(glob.glob(os.path.join(carpeta, "**", "*.csv"),
                                recursive=True))
    runs = []
    for arch in archivos:
        conds, filas = None, {}
        run_actual = None
        opt_actual = "secuencial"

        def cerrar():
            if conds and filas:
                datos = {vid: np.array(v, dtype=np.float64)
                         for vid, v in filas.items()}
                runs.append((f"{os.path.basename(arch)}::{run_actual}",
                             conds, opt_actual, datos))

        with open(arch, encoding="utf-8") as f:
            for linea in f:
                linea = linea.strip()
                if not linea or linea.startswith("run,"):
                    continue
                if linea.startswith("#"):
                    cerrar()                      # fin del run anterior
                    conds, filas = None, {}
                    if "condiciones_iniciales=" in linea:
                        try:
                            conds = _parsear_condiciones(linea)
                            run_actual = linea.split("run=", 1)[1].split()[0]
                            opt_actual = ("secuencial" if "opt=" not in linea
                                          else linea.split("opt=", 1)[1].split()[0])
                            opt_actual = ALIAS_ETIQUETA.get(opt_actual,
                                                            opt_actual)
                        except Exception:
                            conds = None          # comentario corrupto: se salta
                    continue
                if conds is None:
                    continue
                c = linea.split(",")
                vid = int(c[1])
                x1, y1, x2, y2 = map(float, c[3:7])
                v, th, a, giro = map(float, c[7:11])
                # Estado por instante: centro del OBB (mitad de las esquinas
                # opuestas), rumbo, velocidad y el control aplicado.
                filas.setdefault(vid, []).append(
                    ((x1 + x2) / 2.0, (y1 + y2) / 2.0, th, v, a, giro))
        cerrar()
    return _capar_recuperados(runs)


# Como MUCHO esta fracción de los runs secuenciales puede venir de órdenes
# rescatados de casos de prioridades. Son gratis (su ruta ya estaba calculada al
# comparar órdenes), así que sin tope acabarían siendo mayoría y la red
# aprendería "secuencial" sobre todo de flotas que en realidad se generaron
# para otra cosa.
FRAC_MAX_RECUPERADOS = 0.4

# Etiquetas de los CSV que se unifican al leer. Las 'secuencial_par' son las
# GEMELAS del sistema viejo (una situación pensada para global/grupos, resuelta
# TAMBIÉN en secuencial). Se desmanteló porque los órdenes descartados que ahora
# se reaprovechan dan lo mismo gratis, y quedaron 10 runs sueltos.
#
# Cuentan como secuencial PURA, no como reutilizadas: sus rutas se planificaron
# en secuencial de principio a fin, igual que cualquier otra secuencial. Lo
# único que tienen de particular es de dónde salió la situación de partida, y
# eso no sesga las rutas. Las reutilizadas, en cambio, sí llegan sesgadas (solo
# se puede rescatar un orden que respete grupo y prioridad), y por eso a ellas
# sí se les baja el peso.
ALIAS_ETIQUETA = {"secuencial_par": "secuencial"}


def _capar_recuperados(runs):
    """Limita los runs 'secuencial_rec' a FRAC_MAX_RECUPERADOS del total de
    secuenciales. Al descartar el sobrante se barajan primero de forma estable
    (por su clave, no por el orden de lectura): así los que sobreviven quedan
    repartidos entre escenarios de origen distintos en vez de amontonarse en los
    primeros ficheros leídos."""
    rec = [r for r in runs if r[2] == "secuencial_rec"]
    otros = [r for r in runs if r[2] != "secuencial_rec"]
    if not rec:
        return runs
    n_sec = sum(1 for r in otros if r[2].startswith("secuencial"))
    # n_rec / (n_sec + n_rec) <= f  →  n_rec <= f/(1-f) · n_sec
    f = FRAC_MAX_RECUPERADOS
    tope = int(n_sec * f / (1.0 - f)) if f < 1.0 else len(rec)
    if len(rec) > tope:
        rec.sort(key=lambda r: zlib.crc32(r[0].encode()))
        rec = rec[:tope]
    return otros + rec


# --------------------------------------------------------------------------- #
# Submuestreo del DATASET (pasadas con el 35 % / 70 % / 100 % de los datos)
# --------------------------------------------------------------------------- #
def _n_veh_de(run):
    """Tamaño de la flota de un run (los vehículos con condiciones iniciales)."""
    _, conds, _, datos = run
    return sum(1 for vid in datos if vid in conds)


def _orden_submuestreo(clave):
    """Orden pseudoaleatorio ESTABLE de un run dentro de su estrato. El prefijo
    lo desacopla del hash que decide la validación (es_validacion), que también
    es un crc32 de la clave: sin él, quedarse con los primeros sesgaría el
    reparto train/val."""
    return zlib.crc32(("submuestreo:" + clave).encode())


def submuestrear_runs(runs, frac=1.0, frac_modos=None, verbose=True):
    """Se queda con una fracción de los runs de ENTRENAMIENTO (los de validación
    pasan enteros, para que el val_mse siga siendo comparable entre pasadas).

    El recorte es ESTRATIFICADO por (etiqueta de modo, nº de vehículos): dentro
    de cada combinación se conserva la misma proporción, así que el subconjunto
    mantiene la mezcla de modos Y el reparto de tamaños de flota del dataset
    completo. Dentro de cada estrato la elección es pseudoaleatoria pero fija
    (hash de la clave del run), de modo que:
      · queda uniforme en TODO lo demás (escenario de origen, fichero, semilla,
        grupos/prioridades…), porque no se ordena por ninguna de esas variables;
      · los subconjuntos ANIDAN: el 35 % es un subconjunto del 70 %, y este del
        100 %. Así la pasada fina reaprovecha lo visto en la amplia en vez de
        cambiar de datos a mitad del plan.

    'frac_modos' permite pedir una proporción distinta por etiqueta, p. ej.
    {"global": 1.0, "prioridades": 1.0}: los modos raros se conservan enteros y
    solo se recortan los secuenciales, que son los que sobran."""
    frac_modos = frac_modos or {}
    if frac >= 1.0 and not frac_modos:
        return runs
    va = [r for r in runs if es_validacion(r[0])]
    tr = [r for r in runs if not es_validacion(r[0])]

    estratos = {}
    for r in tr:
        estratos.setdefault((r[2], _n_veh_de(r)), []).append(r)

    elegidos, resumen = [], {}
    for (modo, n_veh), grupo in estratos.items():
        f = min(1.0, max(0.0, frac_modos.get(modo, frac)))
        grupo.sort(key=lambda r: _orden_submuestreo(r[0]))
        # Se redondea al alza (y nunca a cero si se pide algo) para que los
        # estratos pequeños —las flotas grandes, que son las escasas— no
        # desaparezcan del subconjunto.
        n = 0 if f <= 0 else max(1, int(math.ceil(f * len(grupo))))
        elegidos.extend(grupo[:n])
        d = resumen.setdefault(modo, [0, 0])
        d[0] += n
        d[1] += len(grupo)

    if verbose:
        tot_n = sum(v[0] for v in resumen.values())
        tot_d = sum(v[1] for v in resumen.values()) or 1
        det = " · ".join(f"{m} {v[0]}/{v[1]}" for m, v in sorted(resumen.items()))
        print(f"[datos] submuestreo train {tot_n}/{tot_d} runs "
              f"({100.0 * tot_n / tot_d:.0f} %) · {det} · "
              f"validación entera ({len(va)} runs)")
    return elegidos + va


def parsear_frac_modos(texto):
    """'global=1,prioridades=1,secuencial=0.35' → dict. Devuelve {} si es None."""
    if not texto:
        return {}
    out = {}
    for trozo in texto.split(","):
        trozo = trozo.strip()
        if not trozo:
            continue
        if "=" not in trozo:
            raise SystemExit(f"--frac-modos: falta '=' en '{trozo}'")
        k, v = trozo.split("=", 1)
        k = ALIAS_ETIQUETA.get(k.strip(), k.strip())
        try:
            out[k] = float(v)
        except ValueError:
            raise SystemExit(f"--frac-modos: '{v}' no es un número")
    return out


# --------------------------------------------------------------------------- #
# Construcción de muestras (una por vehículo e instante)
# --------------------------------------------------------------------------- #
# La ETIQUETA guardada en el CSV no siempre es un modo de planificación real.
# 'secuencial_par' es el gemelo secuencial de un escenario pensado para
# global/prioridades: se planificó COMO secuencial, así que en la ENTRADA de la
# red le toca el one-hot de 'secuencial'. Sin esta traducción, vector_entrada no
# reconoce la etiqueta y deja el one-hot de modo a CERO — un código de contexto
# que en inferencia no se da nunca, porque el usuario siempre elige uno de los
# tres modos reales. La etiqueta se conserva aparte, en M, que es lo que usa
# pesos_de_modos para darle la sensibilidad intermedia.
MODO_ENTRADA = {"secuencial_par": "secuencial",
                "secuencial_rec": "secuencial"}


def construir_muestras(runs):
    """Devuelve (X, Y, M): entradas, objetivos y la ETIQUETA de cada muestra
    ('secuencial'/'secuencial_par'/'global'/'prioridades'), para poder ponderar
    en el entrenamiento los modos raros (ver `pesos_de_modos`). Ojo: en X va el
    modo de PLANIFICACIÓN (ver MODO_ENTRADA), en M la etiqueta."""
    X, Y, M = [], [], []
    for _, conds, opt, datos in runs:
        opt_x = MODO_ENTRADA.get(opt, opt)      # el que ve la red
        vids = [vid for vid in datos if vid in conds]
        n_veh = len(vids)               # tamaño de la flota vista en este run
        for vid in vids:
            arr = datos[vid]
            p = conds[vid]
            n = len(arr)
            for t in range(n):
                pasado = [(arr[k, 4], arr[k, 5])
                          for k in range(max(0, t - pol.H_PASADO), t)]
                ego = {"x": arr[t, 0], "y": arr[t, 1], "th": arr[t, 2],
                       "v": arr[t, 3], "largo": p["largo"], "ancho": p["ancho"],
                       "v_max": p["v_max"], "a_max": p["a_max"],
                       "giro_max": p["giro_max"], "grupo": p["grupo"],
                       "prioridad": p["prioridad"], "meta": p["meta"],
                       "meta_th": p["meta_th"], "pasado": pasado,
                       "opt": opt_x, "n_veh": n_veh}
                otros = []
                for ov in vids:
                    if ov == vid:
                        continue
                    oa, op = datos[ov], conds[ov]
                    if t < len(oa):
                        ox, oy, oth, ovel = oa[t, 0], oa[t, 1], oa[t, 2], oa[t, 3]
                    else:               # ya aparcado: última pose, v = 0
                        ox, oy, oth, ovel = oa[-1, 0], oa[-1, 1], oa[-1, 2], 0.0
                    otros.append({"x": ox, "y": oy, "th": oth, "v": ovel,
                                  "largo": op["largo"], "ancho": op["ancho"],
                                  "v_max": op["v_max"],
                                  "grupo": op["grupo"],
                                  "prioridad": op["prioridad"]})
                X.append(vector_entrada(ego, otros))
                obj = np.zeros(2 * pol.N_PRED, dtype=np.float32)
                for k in range(pol.N_PRED):
                    if t + k < n:       # más allá del final: aparcado (0, 0)
                        obj[2 * k] = arr[t + k, 4] / max(p["a_max"], 1e-6)
                        obj[2 * k + 1] = (arr[t + k, 5]
                                          / max(p["giro_max"], 1e-6))
                Y.append(obj)
                M.append(opt)
    return (np.asarray(X, dtype=np.float32),
            np.asarray(Y, dtype=np.float32),
            np.asarray(M))


def es_validacion(clave_run):
    """~10 % de los runs a validación, de forma estable entre ejecuciones."""
    return zlib.crc32(clave_run.encode()) % 10 == 0


# --------------------------------------------------------------------------- #
# Puntuación por ROLLOUT en bucle cerrado (la métrica que decide "la mejor red")
# --------------------------------------------------------------------------- #
# Escalas de la recompensa parcial: a qué distancia (m) y a qué error de ángulo
# (rad) la puntuación cae a 1/e. Fijan cómo de "generosa" es la nota al que no
# llega pero se queda cerca.
ESCALA_DIST = 2.0
ESCALA_ANG = math.radians(30.0)

# Techo de la nota de un vehículo que ha chocado. Queda por debajo del 0,5 que
# como mucho saca uno que no llega pero conduce limpio, así que chocar nunca
# puede salir a cuenta.
FACTOR_CHOQUE = 0.1


def _puntuar_vehiculo(veh):
    """Nota POSITIVA de un vehículo tras el rollout:
      · llega sin chocar (mision_ok)     → 1 + ang        (rango 1..2, ALTA),
      · no llega, sin chocar             → 0.5·prox·(0.5+0.5·ang)   (0..0,5)
      · CHOCA, llegue o no               → 0.1·prox·(0.5+0.5·ang)   (0..0,1)

    Los tres tramos no se solapan, así que chocar es SIEMPRE peor que cualquier
    resultado limpio, aunque el que choca llegue y el limpio no. Eso es lo
    innegociable: como el simulador no altera la física al chocar, un vehículo
    que atraviesa a otro llegaría igual de bien a su plaza, y sin esto esas
    redes ganarían el barrido.

    Lo que no se hace es anular la nota del todo. Al principio de una búsqueda
    casi todas las redes chocan con casi todo, y con un cero plano no habría
    forma de distinguir a la que se estrella al salir de la que casi llega:
    dentro del tramo de choque se sigue premiando acercarse, que es la pista
    que necesita el buscador para ir mejorando.
    donde 'prox' = e^(−d/ESCALA_DIST) premia acabar cerca de la meta y 'ang' =
    e^(−|Δθ|/ESCALA_ANG) premia acabar con el ángulo de llegada exigido (1.0 si el
    ángulo es libre). Así "llegar en buen ángulo" domina y el resto puntúa poco
    pero de forma continua según cuánto se acerca en posición y en ángulo."""
    from nucleo import ang_norm
    if not veh.traj:
        return 0.0
    fx, fy, fth = veh.traj[-1]
    mx, my = veh.meta
    d = math.hypot(fx - mx, fy - my)
    # Una red divergida (NaN en los pesos) produce poses no finitas: no puntúa.
    if not (math.isfinite(d) and math.isfinite(fth)):
        return 0.0
    prox = math.exp(-d / ESCALA_DIST)
    if veh.meta_th is None:
        ang = 1.0
    else:
        ang = math.exp(-abs(ang_norm(fth - veh.meta_th)) / ESCALA_ANG)
    if getattr(veh, "choque", False):
        return FACTOR_CHOQUE * prox * (0.5 + 0.5 * ang)
    if veh.mision_ok:
        return 1.0 + ang
    return 0.5 * prox * (0.5 + 0.5 * ang)


def _opt_de_run(arch, rid):
    """Modo de optimización guardado en la línea '# run=<rid> opt=<modo> …'."""
    with open(arch, encoding="utf-8") as f:
        for linea in f:
            if linea.startswith("# run="):
                if linea.split("run=", 1)[1].split()[0] == rid:
                    return ("secuencial" if "opt=" not in linea
                            else linea.split("opt=", 1)[1].split()[0])
    return "secuencial"


def escenarios_eval(carpeta):
    """Lista de (archivo, run_id, modo_opt) para reproducir cada escenario."""
    from nucleo import listar_runs
    return [(arch, rid, _opt_de_run(arch, rid))
            for arch, rid, _ in listar_runs(carpeta)]


_PLANTILLAS = {}        # (archivo, run) -> vehículos base, para no reparsear el CSV


def _flota_de(arch, rid):
    """Vehículos frescos de un escenario (el rollout muta traj, así que cada
    evaluación necesita copias propias)."""
    import copy
    from nucleo import cargar_run
    clave = (arch, rid)
    if clave not in _PLANTILLAS:
        vehs = cargar_run(arch, rid)
        for v in vehs:
            v.traj = []                      # la del CSV no hace falta y pesa
        _PLANTILLAS[clave] = vehs
    return copy.deepcopy(_PLANTILLAS[clave])


def evaluar_rollout(politica, escenarios):
    """Media de la nota por vehículo sobre todos los escenarios, ejecutando la
    política en bucle cerrado desde el estado inicial. Los escenarios se simulan
    A LA VEZ para agrupar las consultas a la red. Devuelve (nota, n_veh)."""
    from politica import rollout_multiflota
    flotas = [_flota_de(arch, rid) for arch, rid, _ in escenarios]
    opts = [opt for _, _, opt in escenarios]
    rollout_multiflota(flotas, politica, opts=opts)
    total, n = 0.0, 0
    for vehs in flotas:
        for veh in vehs:
            total += _puntuar_vehiculo(veh)
            n += 1
    return (total / n if n else 0.0), n


# --------------------------------------------------------------------------- #
# Entrenamiento de UNA red (reutilizado por el modo simple y por el barrido)
# --------------------------------------------------------------------------- #
def _a_tensores(Xtr, Ytr, Wtr, Xva, Yva, media, escala, device):
    import torch
    return (torch.from_numpy((Xtr - media) / escala).to(device),
            torch.from_numpy(Ytr).to(device),
            torch.from_numpy(np.asarray(Wtr, dtype=np.float32)).to(device),
            torch.from_numpy((Xva - media) / escala).to(device),
            torch.from_numpy(Yva).to(device))


FACTOR_LR_SGD = 10.0    # SGD necesita un paso mayor que Adam para el mismo rango
#                         de lr, si no la comparación entre ambos es falsa


def entrenar_red(red, tensores, device, epocas, lr, lote, weight_decay=0.01,
                 optimizador="adamw", verbose=False):
    """Entrena 'red' y devuelve (mejor_state_dict, mejor_val). Guarda en memoria
    el estado de la época con menor val (no toca disco)."""
    import copy
    import torch
    Xtr_t, Ytr_t, Wtr_t, Xva_t, Yva_t = tensores
    if optimizador == "sgd":
        opt = torch.optim.SGD(red.parameters(), lr=lr * FACTOR_LR_SGD,
                              momentum=0.9, weight_decay=weight_decay)
    else:
        opt = torch.optim.AdamW(red.parameters(), lr=lr,
                                weight_decay=weight_decay)
    plani = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epocas)
    perdida = torch.nn.MSELoss()
    perdida_muestra = torch.nn.MSELoss(reduction="none")   # para ponderar
    mejor_val, mejor_estado = float("inf"), None
    for ep in range(1, epocas + 1):
        t0 = time.perf_counter()
        red.train()
        orden = torch.randperm(len(Xtr_t), device=device)
        tot = 0.0
        for i in range(0, len(orden), lote):
            idx = orden[i:i + lote]
            opt.zero_grad()
            # MSE ponderada por muestra: los modos raros (global/prioridades)
            # pesan más (ver pesos_de_modos). Media por dims de salida y luego
            # media ponderada sobre las muestras del lote.
            err = perdida_muestra(red(Xtr_t[idx]), Ytr_t[idx]).mean(dim=1)
            w = Wtr_t[idx]
            l = (err * w).sum() / w.sum()
            l.backward()
            opt.step()
            tot += l.item() * len(idx)
        plani.step()
        red.eval()
        with torch.no_grad():
            lv = sum(perdida(red(Xva_t[i:i + lote]),
                             Yva_t[i:i + lote]).item() * len(Xva_t[i:i + lote])
                     for i in range(0, len(Xva_t), lote)) / len(Xva_t)
        # Solo cuenta una época con pérdida FINITA: con tasas altas (sobre todo
        # con SGD) el entrenamiento puede divergir a NaN, y NaN < x siempre es
        # falso, así que sin esta guarda no se guardaría ningún estado.
        if math.isfinite(lv) and lv < mejor_val:
            mejor_val = lv
            mejor_estado = copy.deepcopy(red.state_dict())
        if verbose:
            marca = "  <- mejor" if lv <= mejor_val else ""
            print(f"[train] época {ep:3d}/{epocas} · train {tot / len(Xtr_t):.5f}"
                  f" · val {lv:.5f} · {time.perf_counter() - t0:.1f} s{marca}")
    if mejor_estado is None:        # divergió en TODAS las épocas
        mejor_estado = copy.deepcopy(red.state_dict())
    return mejor_estado, mejor_val


def _cfg(arq, nota=None, val=None):
    """Diccionario 'config' que se guarda en el .pt (arquitectura + representación),
    suficiente para reconstruir y desplegar la red con Politica.cargar."""
    return {"dim_entrada": pol.DIM_ENTRADA, "oculto": arq["oculto"],
            "n_pred": pol.N_PRED, "h_pasado": pol.H_PASADO,
            "n_capas": arq["n_capas"], "dropout": arq["dropout"],
            "activacion": arq["activacion"], "n_vecinos": pol.N_VECINOS,
            "horizonte": pol.HORIZONTE_VECINO,
            "normalizacion": arq.get("normalizacion", "no"),
            "nota_rollout": nota, "val_mse": val}


def _guardar(ruta, cfg, media, escala, state):
    import torch
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    torch.save({"config": cfg, "media": media.tolist(),
                "escala": escala.tolist(), "state_dict": state}, ruta)


#   secuencial pura           → sensibilidad BASE (nunca se sube). Incluye las
#     (y las gemelas)             gemelas, ver ALIAS_ETIQUETA
#   secuencial_rec            → BASE, y además rebajada a la mitad por
#     (reutilizadas)              PESO_SUB_MODO: llegan sesgadas hacia flotas
#                                 con jerarquías planas
#   global / prioridades      → sensibilidad ALTA (todo 'enfasis')
NIVEL_MODO = {"secuencial": 0.0, "secuencial_rec": 0.0,
              "global": 1.0, "prioridades": 1.0}

# Modos que pesan MENOS que la base. NIVEL_MODO solo sabe subir la sensibilidad
# por encima del secuencial puro; esto es el mando para bajarla.
#
# 'secuencial_rec' son órdenes descartados de casos global/grupos que se
# reaprovechan. Son datos buenos y gratis, pero llegan sesgados: un orden solo
# se puede reaprovechar si respeta grupo y prioridad, y eso ocurre mucho más en
# flotas con pocos grupos/prioridades distintos (más empates → más órdenes
# válidos). O sea que sobre-representan las jerarquías planas. Al valer la
# mitad, aportan sin arrastrar la red hacia ese tipo de flota.
PESO_SUB_MODO = {"secuencial_rec": 0.5}


def pesos_de_modos(modos, enfasis=1.0):
    """Peso por muestra según su modo/etiqueta, en TRES niveles de sensibilidad
    (ver NIVEL_MODO): la secuencial pura no se toca; su gemela (una situación
    pensada para global/prioridades pero resuelta en secuencial) recibe una
    sensibilidad intermedia; global y prioridades reciben la sensibilidad alta.

    Dentro de cada nivel activo se usa el inverso de la frecuencia (para que
    dentro de "alta" ni global ni prioridades se coman entre sí si una es más
    rara que la otra), elevado a 'enfasis · nivel'. 'enfasis' global sigue
    siendo el mando único: 0 = sin ponderar nada, 1 = valores por defecto de
    NIVEL_MODO, >1 exagera. Se normaliza a media 1 para no alterar el paso
    efectivo del optimizador."""
    modos = np.asarray(modos)
    pesos = np.ones(len(modos), dtype=np.float32)
    if enfasis <= 0 or len(modos) == 0:
        return pesos
    n = len(modos)
    unicos, cuentas = np.unique(modos, return_counts=True)
    frec = dict(zip(unicos, cuentas))
    # Nº de modos PRESENTES, no de entradas de la tabla: si se contase la tabla,
    # añadirle una etiqueta nueva (aunque sea de nivel base, que ni entra en el
    # bucle) movería los pesos de todos los demás sin motivo.
    n_niveles = len(unicos)
    for m, c in frec.items():
        nivel = NIVEL_MODO.get(m, 1.0)   # modo desconocido: se trata como "alta"
        if nivel <= 0:
            continue                     # base: se queda en 1
        pesos[modos == m] = (n / (n_niveles * c)) ** (enfasis * nivel)
    for m, sub in PESO_SUB_MODO.items():
        # Se aplica DESPUÉS de los niveles y ANTES de normalizar, para que
        # rebajar un modo no altere la proporción entre los demás.
        pesos[modos == m] *= sub ** enfasis
    pesos *= len(pesos) / pesos.sum()   # media 1
    return pesos.astype(np.float32)


def _preparar_datos(runs):
    """Divide en train/val por run y construye muestras con la representación
    ACTUAL (pol.*). Devuelve (Xtr, Ytr, Mtr, Xva, Yva); 'Mtr' es el modo de cada
    muestra de train (para ponderar). La validación NO se pondera, para que la
    selección de modelo siga siendo una cifra honesta."""
    r_tr = [r for r in runs if not es_validacion(r[0])]
    r_va = [r for r in runs if es_validacion(r[0])]
    if not r_va:                        # pocos runs: aparta el último
        r_va = [r_tr.pop()]
    Xtr, Ytr, Mtr = construir_muestras(r_tr)
    Xva, Yva, _ = construir_muestras(r_va)
    return Xtr, Ytr, Mtr, Xva, Yva


# --------------------------------------------------------------------------- #
# Modo SIMPLE: entrena una sola red con los hiperparámetros dados
# --------------------------------------------------------------------------- #
def main(args):
    import torch

    print(f"[datos] leyendo CSVs de {args.rutas} …")
    runs = leer_runs(args.rutas)
    if not runs:
        raise SystemExit("No hay runs en la carpeta de rutas: genera datos "
                         "primero con generador.py (o con la GUI).")
    runs = submuestrear_runs(runs, args.frac_datos,
                             parsear_frac_modos(args.frac_modos))
    pol.configurar_representacion(args.n_vecinos, args.horizonte, args.h_pasado)
    Xtr, Ytr, Mtr, Xva, Yva = _preparar_datos(runs)
    Wtr = pesos_de_modos(Mtr, getattr(args, "enfasis_modos", 1.0))
    print(f"[datos] {len(runs)} runs · {len(Xtr):,} muestras train / "
          f"{len(Xva):,} val (dim entrada {pol.DIM_ENTRADA})")

    media = Xtr.mean(axis=0)
    escala = np.maximum(Xtr.std(axis=0), 1e-6)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] dispositivo: {device}")
    tensores = _a_tensores(Xtr, Ytr, Wtr, Xva, Yva, media, escala, device)

    arq = {"oculto": args.oculto, "n_capas": args.n_capas,
           "dropout": args.dropout, "activacion": args.activacion}
    red = crear_red(pol.DIM_ENTRADA, arq["oculto"], pol.N_PRED,
                    arq["n_capas"], arq["dropout"], arq["activacion"]).to(device)
    estado, mejor_val = entrenar_red(red, tensores, device, args.epocas,
                                     args.lr, args.lote, verbose=True)
    _guardar(args.salida, _cfg(arq), media, escala, estado)

    # Nota de rollout del modelo guardado (métrica de verdad).
    from politica import Politica
    red.load_state_dict(estado)
    politica = Politica(red.eval(),
                        torch.tensor(media, dtype=torch.float32, device=device),
                        torch.tensor(escala, dtype=torch.float32, device=device),
                        device)
    nota, nveh = evaluar_rollout(politica, escenarios_eval(args.rutas))
    print(f"[train] mejor val {mejor_val:.5f} · nota rollout {nota:.4f} "
          f"({nveh} veh.) · modelo en {args.salida}")


# --------------------------------------------------------------------------- #
# Modo BARRIDO: prueba muchas configs y se queda con la de mejor nota de rollout
# --------------------------------------------------------------------------- #
# Espacio de búsqueda de la exploración amplia. Dos topes vienen del propio
# problema, no de la red: los escenarios tienen 2-6 vehículos, así que un vehículo
# nunca ve más de 5 vecinos (valores mayores serían bloques siempre a cero); y el
# horizonte del filtro debe ser >= N_PRED (10), los pasos que la red compromete en
# bucle abierto.
#
# El techo de TAMAÑO de red es alto a propósito (hasta ~80 M de parámetros). La
# GPU va sobrada con las redes pequeñas —rinde a un 10 % de su capacidad—, así
# que agrandar sale casi gratis, y el mapa no se codifica en la entrada: la red
# tiene que aprendérselo de las coordenadas, y eso pide capacidad. El límite real
# no es la tarjeta sino los datos (612 k muestras), y de eso se encargan
# 'dropout' y 'weight_decay', que por eso llegan más arriba que antes.
#
# Hubo un recorte (fuera 'sgd' y las redes de 6 capas) basado en la primera
# fase aleatoria. Se ha DESHECHO: aquella nota no miraba las colisiones, así
# que medía otra cosa y no sirve para descartar nada. Además, el mapa no entra
# como dato en la red —se entrena y se usa siempre sobre el mismo, así que
# tiene que memorizarlo—, y memorizar pide parámetros: las redes profundas son
# justo las que no convenía quitar.
ESPACIO = {
    "n_capas":      [2, 3, 4, 6],
    "oculto":       [256, 512, 1024, 2048, 4096],
    "dropout":      [0.0, 0.1, 0.2],
    "activacion":   ["relu", "gelu", "silu"],
    "lr":           [1e-3, 3e-3, 8e-3],
    "lote":         [512, 2048, 8192, 16384],
    "n_vecinos":    [2, 3, 5],
    "h_pasado":     [3, 5, 10],
    "horizonte":    [10, 15, 20],
    # Presupuesto único. Dejó de ser un eje de búsqueda: era a la vez algo que
    # el muestreador elegía y algo que los peajes de la fase 2 recortaban, y eso
    # hacía que dos candidatas idénticas salvo por el presupuesto compitieran
    # sin que la diferencia significase nada.
    "epocas":       [80],
    "weight_decay": [0.0, 0.01, 0.1],
    "normalizacion": ["no", "layernorm"],
    "optimizador":  ["adamw", "sgd"],
}


CAMPOS_EXTRA = ["dim_entrada", "n_muestras", "val_mse", "nota_rollout",
                "segundos"]


def _clave_config(c, claves):
    """Identificador textual de una config, con el mismo formato que se escribe
    en el CSV, para poder reconocer las ya evaluadas al reanudar."""
    return tuple(str(c[k]) for k in claves)


def _ya_evaluadas(log_path, claves):
    """Configs presentes en un registro previo (reanudación tras un corte)."""
    import csv
    if not os.path.exists(log_path):
        return set()
    hechas = set()
    with open(log_path, encoding="utf-8", newline="") as f:
        for fila in csv.DictReader(f):
            if all(k in fila for k in claves):
                hechas.add(tuple(fila[k] for k in claves))
    return hechas


def _evaluar_config(c, runs, escenarios, device, epocas, cache):
    """Entrena y evalúa UNA config. Devuelve (nota, val, media, escala, estado,
    dim_entrada, n_muestras). 'cache' guarda las muestras por representación."""
    import torch
    from politica import Politica
    clave = (c["n_vecinos"], c["horizonte"], c["h_pasado"])
    # Fijar siempre los globales: el rollout posterior los consulta.
    pol.configurar_representacion(*clave)
    enf = c.get("enfasis_modos", 1.0)
    clave_cache = clave + (enf,)          # el peso por modo depende del énfasis
    if clave_cache not in cache:
        Xtr, Ytr, Mtr, Xva, Yva = _preparar_datos(runs)
        Wtr = pesos_de_modos(Mtr, enf)
        media = Xtr.mean(axis=0)
        escala = np.maximum(Xtr.std(axis=0), 1e-6)
        cache[clave_cache] = (
            _a_tensores(Xtr, Ytr, Wtr, Xva, Yva, media, escala, device),
            media, escala, len(Xtr))
    tensores, media, escala, n_muestras = cache[clave_cache]

    red = crear_red(pol.DIM_ENTRADA, c["oculto"], pol.N_PRED,
                    c["n_capas"], c["dropout"], c["activacion"],
                    c.get("normalizacion", "no")).to(device)
    estado, val = entrenar_red(
        red, tensores, device, int(c.get("epocas", epocas)), c["lr"], c["lote"],
        weight_decay=c.get("weight_decay", 0.01),
        optimizador=c.get("optimizador", "adamw"))
    red.load_state_dict(estado)
    politica = Politica(
        red.eval(), torch.tensor(media, dtype=torch.float32, device=device),
        torch.tensor(escala, dtype=torch.float32, device=device), device)
    nota, _ = evaluar_rollout(politica, escenarios)
    return nota, val, media, escala, estado, pol.DIM_ENTRADA, n_muestras


def _trabajar_configs(tarea):
    """Worker de barrido: evalúa su lote de configs, escribe su propio CSV y
    devuelve solo la MEJOR que ha encontrado (devolver todas sería tirar
    gigabytes de pesos entre procesos)."""
    import csv
    import torch
    wid, configs, claves, rutas, epocas, log_w, frac, frac_modos = tarea
    runs = submuestrear_runs(leer_runs(rutas), frac, frac_modos, verbose=False)
    escenarios = escenarios_eval(rutas)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache, mejor = {}, None
    with open(log_w, "w", newline="", encoding="utf-8") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=claves + CAMPOS_EXTRA)
        w.writeheader()
        for k, c in enumerate(configs, 1):
            t0 = time.perf_counter()
            nota, val, media, escala, estado, dim, n_m = _evaluar_config(
                c, runs, escenarios, device, epocas, cache)
            fila = dict(c)
            fila.update({"dim_entrada": dim, "n_muestras": n_m,
                         "val_mse": round(val, 5), "nota_rollout": round(nota, 4),
                         "segundos": round(time.perf_counter() - t0, 1)})
            w.writerow(fila)
            fcsv.flush()
            if mejor is None or nota > mejor[0]:
                # Los pesos viajan en CPU: el proceso padre los reubica.
                estado_cpu = {n: t.detach().cpu() for n, t in estado.items()}
                mejor = (nota, val, dict(c), media, escala, estado_cpu, dim)
            print(f"[w{wid}] {k}/{len(configs)} · nota {nota:.4f} · "
                  f"mejor {mejor[0]:.4f}", flush=True)
    return wid, mejor, len(configs)


def barrido(args):
    import csv
    import random as _rnd
    import torch

    print(f"[datos] leyendo CSVs de {args.rutas} …")
    runs = leer_runs(args.rutas)
    if not runs:
        raise SystemExit("No hay runs: genera datos primero con generador.py.")
    frac_modos = parsear_frac_modos(args.frac_modos)
    runs = submuestrear_runs(runs, args.frac_datos, frac_modos)
    # El rollout se hace SIEMPRE con todos los escenarios: es la métrica que
    # decide el ganador y tiene que ser comparable entre las tres pasadas.
    escenarios = escenarios_eval(args.rutas)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Espacio de búsqueda: el JSON de --espacio (fase de grid fino) o el ESPACIO
    # por defecto. Debe contener las mismas claves que ESPACIO.
    if args.espacio:
        import json
        with open(args.espacio, encoding="utf-8") as fj:
            espacio = json.load(fj)
    else:
        espacio = ESPACIO
    claves = list(ESPACIO.keys())              # orden fijo de columnas

    _rnd.seed(args.semilla)
    todas = list(itertools.product(*(espacio[k] for k in claves)))
    _rnd.shuffle(todas)
    # Exhaustivo: TODAS las combinaciones (grid fino ya acotado). Si no, muestreo
    # aleatorio sin repetición de n_configs (búsqueda amplia).
    if not args.exhaustivo:
        # --fraccion pide un PORCENTAJE del espacio total (p. ej. 0.2 = 20 %);
        # si no se da, se usa el número fijo de --n-configs.
        n = (int(round(args.fraccion * len(todas))) if args.fraccion
             else args.n_configs)
        todas = todas[:max(1, n)]
    configs = [dict(zip(claves, vals)) for vals in todas]

    # Cada pasada (35 % / 70 % / 100 %) lleva su PROPIO registro: las notas de
    # una no son comparables con las de otra, y compartir fichero haría que la
    # reanudación diese por evaluadas configs entrenadas con menos datos.
    suf = "" if args.frac_datos >= 1.0 else f"_f{int(round(args.frac_datos * 100))}"
    log_path = args.log or os.path.join(os.path.dirname(args.salida),
                                        f"barrido{suf}.csv")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    campos = claves + CAMPOS_EXTRA

    # REANUDACIÓN: si ya hay registro previo, no se repite lo evaluado.
    hechas = _ya_evaluadas(log_path, claves)
    if hechas:
        antes = len(configs)
        configs = [c for c in configs
                   if _clave_config(c, claves) not in hechas]
        print(f"[barrido] reanudando: {antes - len(configs)} configs ya "
              f"evaluadas en {os.path.basename(log_path)}, faltan {len(configs)}")

    procesos = max(1, min(args.procesos, len(configs) or 1))
    print(f"[barrido] {len(runs)} runs · {len(escenarios)} escenarios de rollout "
          f"· {device} · {len(configs)} configs · {procesos} procesos"
          f"{' (exhaustivo)' if args.exhaustivo else ''}")
    if not configs:
        raise SystemExit("[barrido] no queda ninguna config por evaluar.")

    t_ini = time.perf_counter()
    mejor = None

    if procesos == 1:
        cache = {}
        nuevo = not hechas
        with open(log_path, "a" if hechas else "w", newline="",
                  encoding="utf-8") as fcsv:
            w = csv.DictWriter(fcsv, fieldnames=campos)
            if nuevo:
                w.writeheader()
            for k, c in enumerate(configs, 1):
                t0 = time.perf_counter()
                nota, val, media, escala, estado, dim, n_m = _evaluar_config(
                    c, runs, escenarios, device, args.epocas, cache)
                fila = dict(c)
                fila.update({"dim_entrada": dim, "n_muestras": n_m,
                             "val_mse": round(val, 5),
                             "nota_rollout": round(nota, 4),
                             "segundos": round(time.perf_counter() - t0, 1)})
                w.writerow(fila)
                fcsv.flush()
                if mejor is None or nota > mejor[0]:
                    mejor = (nota, val, dict(c), media, escala, estado, dim)
                transc = time.perf_counter() - t_ini
                print(f"[barrido] {k}/{len(configs)} · nota {nota:.4f} · "
                      f"val {val:.4f} · faltan "
                      f"~{transc / k * (len(configs) - k) / 60:.0f} min · "
                      f"mejor {mejor[0]:.4f}", flush=True)
    else:
        # Reparto por BLOQUES CONTIGUOS tras ordenar por representación: así cada
        # worker toca pocas ternas distintas y su caché de muestras acierta casi
        # siempre (reconstruirlas es lo más caro).
        configs.sort(key=lambda c: (c["n_vecinos"], c["horizonte"],
                                    c["h_pasado"]))
        trozos = [configs[i::procesos] for i in range(procesos)]
        tareas = [(i, t, claves, args.rutas, args.epocas,
                   f"{log_path}.w{i}", args.frac_datos, frac_modos)
                  for i, t in enumerate(trozos) if t]
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=len(tareas), mp_context=ctx) as ex:
            for wid, m, n in ex.map(_trabajar_configs, tareas):
                print(f"[barrido] worker {wid} terminado: {n} configs · "
                      f"mejor {m[0]:.4f}" if m else f"worker {wid}: sin datos")
                if m and (mejor is None or m[0] > mejor[0]):
                    mejor = m
        # Fusiona los CSV parciales en el registro principal y los borra.
        with open(log_path, "a" if hechas else "w", newline="",
                  encoding="utf-8") as fcsv:
            w = csv.DictWriter(fcsv, fieldnames=campos)
            if not hechas:
                w.writeheader()
            for i, _ in enumerate(tareas):
                parcial = f"{log_path}.w{i}"
                if not os.path.exists(parcial):
                    continue
                with open(parcial, encoding="utf-8", newline="") as fp:
                    for fila in csv.DictReader(fp):
                        w.writerow(fila)
                os.remove(parcial)

    print(f"[barrido] {len(configs)} configs en "
          f"{(time.perf_counter() - t_ini) / 60:.1f} min")

    nota, val, c, media, escala, estado, _ = mejor

    # El mejor GLOBAL puede estar en el registro PREVIO (reanudación tras un
    # corte), cuyos pesos ya no existen en memoria. Se detecta por IDENTIDAD de
    # config, NO por el número: la nota del CSV está redondeada y puede quedar un
    # pelín por encima de la de memoria, lo que antes disparaba un reentrenamiento
    # espurio que TIRABA los pesos buenos del ganador de esta misma tanda.
    mejor_log, fila_log = -1.0, None
    with open(log_path, encoding="utf-8", newline="") as f:
        for fila in csv.DictReader(f):
            try:
                n = float(fila["nota_rollout"])
            except (KeyError, ValueError):
                continue
            if n > mejor_log:
                mejor_log, fila_log = n, fila
    c_log = {k: (fila_log[k] if isinstance(ESPACIO[k][0], str)
                 else type(ESPACIO[k][0])(float(fila_log[k])))
             for k in claves}
    if _clave_config(c_log, claves) != _clave_config(c, claves):
        # El mejor del registro es OTRA config: sus pesos no están en memoria, así
        # que se reentrena esa (una sola vez) para poder guardarla.
        print(f"[barrido] el mejor global ({mejor_log:.4f}) es de una tanda "
              f"anterior sin pesos: reentrenando esa config · {c_log}")
        nota, val, media, escala, estado, _, _ = _evaluar_config(
            c_log, runs, escenarios, device, args.epocas, {})
        c = c_log
        print(f"[barrido] reentrenada: nota {nota:.4f} (val {val:.5f})")

    # Reconfigura la representación de la mejor config antes de guardar (por si la
    # última probada era otra) y vuelca el modelo ganador a la salida.
    pol.configurar_representacion(c["n_vecinos"], c["horizonte"], c["h_pasado"])
    arq = {"oculto": c["oculto"], "n_capas": c["n_capas"],
           "dropout": c["dropout"], "activacion": c["activacion"],
           "normalizacion": c.get("normalizacion", "no")}
    _guardar(args.salida, _cfg(arq, nota, val), media, escala, estado)
    print(f"\n[barrido] MEJOR nota {nota:.4f} (val {val:.5f}) · {c}")
    print(f"[barrido] modelo ganador → {args.salida} · registro → {log_path}")


def construir_parser():
    ap = argparse.ArgumentParser(description="Entrena la política neuronal.")
    ap.add_argument("--rutas", default=RUTAS_DIR,
                    help="carpeta raíz con los CSV (def. rutas/)")
    ap.add_argument("--epocas", type=int, default=40)
    ap.add_argument("--lote", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--oculto", type=int, default=512,
                    help="anchura de las capas ocultas (def. 512)")
    ap.add_argument("--n-capas", dest="n_capas", type=int, default=3,
                    help="nº de capas ocultas (def. 3)")
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--activacion", default="relu",
                    choices=("relu", "gelu", "silu"))
    ap.add_argument("--n-vecinos", dest="n_vecinos", type=int,
                    default=pol.N_VECINOS)
    ap.add_argument("--horizonte", type=int, default=pol.HORIZONTE_VECINO)
    ap.add_argument("--h-pasado", dest="h_pasado", type=int, default=pol.H_PASADO)
    ap.add_argument("--enfasis-modos", dest="enfasis_modos", type=float,
                    default=1.0,
                    help="peso de los modos raros (global/prioridades) en el "
                         "entrenamiento: 0 = sin ponderar, 1 = influencia "
                         "equilibrada entre modos, >1 = los raros dominan")
    ap.add_argument("--frac-datos", dest="frac_datos", type=float, default=1.0,
                    help="fracción del DATASET de entrenamiento a usar "
                         "(0.35 = 35 %%). El recorte es estratificado por modo "
                         "y por tamaño de flota, y los subconjuntos anidan "
                         "(el 35 %% está dentro del 70 %%). La validación va "
                         "siempre entera")
    ap.add_argument("--frac-modos", dest="frac_modos", default=None,
                    help="fracción distinta por tipo de escenario, p. ej. "
                         "'global=1,prioridades=1,secuencial=0.3'. Lo que no "
                         "se nombre usa --frac-datos. Etiquetas: secuencial, "
                         "secuencial_rec, global, prioridades")
    ap.add_argument("--salida", default=MODELO_PT)
    ap.add_argument("--barrido", action="store_true",
                    help="prueba muchas configs y guarda la de mejor nota rollout")
    ap.add_argument("--n-configs", dest="n_configs", type=int, default=24,
                    help="configs a probar en el barrido (def. 24)")
    ap.add_argument("--semilla", type=int, default=0,
                    help="semilla del muestreo de configs del barrido")
    ap.add_argument("--espacio", default=None,
                    help="JSON con el espacio de búsqueda (mismas claves que "
                         "ESPACIO); por defecto usa el ESPACIO interno")
    ap.add_argument("--exhaustivo", action="store_true",
                    help="prueba TODAS las combinaciones del espacio (grid fino)")
    ap.add_argument("--fraccion", type=float, default=None,
                    help="fracción del espacio total a muestrear (0.2 = 20 %%); "
                         "tiene prioridad sobre --n-configs")
    ap.add_argument("--procesos", type=int, default=6,
                    help="procesos en paralelo del barrido (def. 6). Bájalo si "
                         "estás usando la GPU para otra cosa")
    ap.add_argument("--log", default=None,
                    help="ruta del CSV de registro (def. modelos/barrido.csv)")
    return ap


if __name__ == "__main__":
    args = construir_parser().parse_args()
    if args.barrido:
        barrido(args)
    else:
        main(args)
