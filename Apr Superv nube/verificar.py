#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verificación de EQUIVALENCIA entre el pipeline local y su versión de nube.

Las adaptaciones de nube (muestras vectorizadas, superset con recorte de
columnas y rollout vectorizado) solo valen si producen exactamente los mismos
números que el código local, que es la referencia. Este script lo comprueba
sobre los CSV que haya en una carpeta de rutas:

  1. superset + recorte de columnas  ==  entrenar.construir_muestras
     para VARIAS representaciones (nº de vecinos, historia, horizonte).
  2. rollout vectorizado             ==  politica.rollout_multiflota
     (solo si hay torch instalado; usa una red sin entrenar, que es suficiente
      para comparar las dos simulaciones paso a paso).

Uso:   python verificar.py [--rutas <carpeta>]
"""

import argparse
import os
import sys

import numpy as np

import comun                                   # añade «Apr Superv local» al path
import politica as pol
import entrenar as ent
import vectorizado as vec


def _muestras_referencia(runs, n_vec, horizonte, h_pasado):
    pol.configurar_representacion(n_vec, horizonte, h_pasado)
    return ent.construir_muestras(runs)


def _muestras_nube(runs, n_vec, horizonte, h_pasado, n_vec_max, h_max, hor_max):
    Xs, TCs, Ys = [], [], []
    for _, conds, opt, datos in runs:
        X, TC, Y = vec.superset_run(conds, opt, datos, n_vec_max, h_max, hor_max)
        Xs.append(X)
        TCs.append(TC)
        Ys.append(Y)
    X = np.concatenate(Xs)
    TC = np.concatenate(TCs)
    Y = np.concatenate(Ys)
    return recortar(X, TC, n_vec, horizonte, h_pasado, n_vec_max, h_max), Y


def recortar(X, TC, n_vec, horizonte, h_pasado, n_vec_max, h_max):
    """Vista de una representación concreta a partir del superset: recorte de
    columnas + puesta a cero de los vecinos que el horizonte descarta."""
    base_vec = vec.DIM_EGO + vec.DIM_META + 2 * h_max
    vivos = vec.mascara_horizonte(TC[:, :n_vec], horizonte)
    V = X[:, vec.columnas_vista(n_vec, h_pasado, n_vec_max, h_max)].copy()
    ini = vec.DIM_EGO + vec.DIM_META + 2 * h_pasado
    bloques = V[:, ini:ini + vec.DIM_VECINO * n_vec].reshape(
        len(V), n_vec, vec.DIM_VECINO)
    bloques *= vivos[:, :, None]
    V[:, ini:ini + vec.DIM_VECINO * n_vec] = bloques.reshape(len(V), -1)
    assert base_vec >= 0
    return V


def comparar(nombre, a, b, tol=2e-4):
    if a.shape != b.shape:
        print(f"  [FALLO] {nombre}: formas distintas {a.shape} vs {b.shape}")
        return False
    d = np.abs(a - b)
    peor = d.max() if d.size else 0.0
    ok = peor <= tol
    print(f"  [{'ok ' if ok else 'FALLO'}] {nombre}: dif. máx. {peor:.3e} "
          f"({a.shape[0]:,} filas × {a.shape[1]} col.)")
    if not ok:
        i, j = np.unravel_index(np.argmax(d), d.shape)
        print(f"          peor en fila {i}, columna {j}: "
              f"local {b[i, j]:.6f} vs nube {a[i, j]:.6f}")
    return ok


def verificar_muestras(runs, n_vec_max, h_max, hor_max):
    print("[1] muestras: superset + recorte  vs  entrenar.construir_muestras")
    todo_ok = True
    casos = [(n_vec_max, hor_max, h_max),      # el propio superset
             (2, 15, 3), (3, 10, 5), (1, 20, 2), (5, 15, 10)]
    for n_vec, horizonte, h_pasado in casos:
        if n_vec > n_vec_max or h_pasado > h_max or horizonte > hor_max:
            continue
        Xr, Yr = _muestras_referencia(runs, n_vec, horizonte, h_pasado)
        (Xn, Yn) = _muestras_nube(runs, n_vec, horizonte, h_pasado,
                                  n_vec_max, h_max, hor_max)
        etiqueta = f"n_vecinos={n_vec} horizonte={horizonte} h_pasado={h_pasado}"
        todo_ok &= comparar(f"X · {etiqueta}", Xn, Xr)
        todo_ok &= comparar(f"Y · {etiqueta}", Yn, Yr)
    return todo_ok


class PoliticaFalsa:
    """Red de mentira, determinista y en numpy: para comparar los dos rollouts
    basta con que ambos reciban la MISMA función de control, no hace falta una
    red entrenada (y así la comprobación tampoco necesita torch)."""

    def __init__(self, dim, semilla=0):
        rng = np.random.default_rng(semilla)
        self.W = rng.normal(scale=0.05, size=(dim, 2 * pol.N_PRED))

    def predecir_lote(self, obs):
        y = np.tanh(np.asarray(obs, dtype=np.float64) @ self.W)
        return y.reshape(len(obs), pol.N_PRED, 2)


def verificar_rollout(rutas, n_vec, horizonte, h_pasado):
    print("[2] rollout vectorizado  vs  politica.rollout_multiflota")
    import vectorizado as vec
    from politica import rollout_multiflota

    pol.configurar_representacion(n_vec, horizonte, h_pasado)
    escenarios = ent.escenarios_eval(rutas)[:12]
    if not escenarios:
        print("  [--] no hay escenarios en la carpeta de rutas")
        return True
    politica = PoliticaFalsa(pol.DIM_ENTRADA)

    flotas = [ent._flota_de(a, r) for a, r, _ in escenarios]
    opts = [o for _, _, o in escenarios]
    rollout_multiflota(flotas, politica, opts=opts)
    ref = np.array([[v.traj[-1][0], v.traj[-1][1], v.traj[-1][2],
                     float(v.mision_ok)]
                    for f in flotas for v in f])

    flotas2 = [ent._flota_de(a, r) for a, r, _ in escenarios]
    vec.rollout(flotas2, politica, opts=opts, n_vec=n_vec,
                        horizonte=horizonte, h_pasado=h_pasado)
    nue = np.array([[v.traj[-1][0], v.traj[-1][1], v.traj[-1][2],
                     float(v.mision_ok)]
                    for f in flotas2 for v in f])
    return comparar("pose final y misión", nue, ref, tol=1e-4)


def main():
    ap = argparse.ArgumentParser(description="Comprueba que la versión de nube "
                                             "calcula lo mismo que la local.")
    ap.add_argument("--rutas", default=os.path.join(comun.RAIZ_LOCAL, "rutas"),
                    help="carpeta con CSV de rutas (def. los del pipeline local)")
    ap.add_argument("--n-vec-max", dest="n_vec_max", type=int, default=7)
    ap.add_argument("--h-max", dest="h_max", type=int, default=10)
    ap.add_argument("--hor-max", dest="hor_max", type=int, default=20)
    args = ap.parse_args()

    print(f"[datos] leyendo {args.rutas} …")
    runs = ent.leer_runs(args.rutas)
    if not runs:
        raise SystemExit("No hay CSV de rutas con los que comparar.")
    print(f"[datos] {len(runs)} runs")

    ok = verificar_muestras(runs, args.n_vec_max, args.h_max, args.hor_max)
    ok &= verificar_rollout(args.rutas, 3, 15, 5)
    print("\nRESULTADO:", "todo equivalente" if ok else "HAY DIFERENCIAS")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
