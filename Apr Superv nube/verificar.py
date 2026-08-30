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
import escenarios as esc
import vectorizado as vec


def _muestras_referencia(runs, n_vec, horizonte, h_pasado, n_fourier=0,
                         n_rayos=0):
    pol.configurar_representacion(n_vec, horizonte, h_pasado, n_fourier, None,
                                  n_rayos)
    return ent.construir_muestras(runs)


def _muestras_nube(runs, n_vec, horizonte, h_pasado, n_vec_max, h_max, hor_max,
                   n_fourier=0, n_f_max=0, n_rayos=0, con_rayos=False):
    mapa_r = esc.obstaculos() if con_rayos else None
    Xs, TCs, Ys = [], [], []
    for _, conds, opt, datos in runs:
        X, TC, Y = vec.superset_run(conds, opt, datos, n_vec_max, h_max, hor_max,
                                    n_f_max, mapa_r)
        Xs.append(X)
        TCs.append(TC)
        Ys.append(Y)
    X = np.concatenate(Xs)
    TC = np.concatenate(TCs)
    Y = np.concatenate(Ys)
    return (recortar(X, TC, n_vec, horizonte, h_pasado, n_vec_max, h_max,
                     n_fourier, n_f_max, n_rayos, con_rayos), Y)


def recortar(X, TC, n_vec, horizonte, h_pasado, n_vec_max, h_max,
             n_fourier=0, n_f_max=0, n_rayos=0, con_rayos=False):
    """Vista de una representación concreta a partir del superset: recorte de
    columnas + puesta a cero de los vecinos que el horizonte descarta."""
    base_vec = vec.DIM_EGO + vec.DIM_META + 2 * h_max
    vivos = vec.mascara_horizonte(TC[:, :n_vec], horizonte)
    V = X[:, vec.columnas_vista(n_vec, h_pasado, n_vec_max, h_max,
                                n_fourier, n_f_max, n_rayos,
                                con_rayos)].copy()
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


def verificar_muestras(runs, n_vec_max, h_max, hor_max, n_f_max=0,
                      con_rayos=False):
    print("[1] muestras: superset + recorte  vs  entrenar.construir_muestras")
    todo_ok = True
    # El último valor es n_fourier: hay que cubrir el 0 (sin el bloque), el tope
    # y un recorte intermedio, que es donde se vería si las ondas se quedan
    # desordenadas o desplazadas al recortar columnas.
    # El ultimo valor es n_rayos: hay que cubrir TODOS los conjuntos, porque el
    # superset los guarda seguidos y una vista se queda con UN tramo; si los
    # desplazamientos estuvieran mal, una configuracion recibiria los rayos de
    # otra sin que nada fallara.
    import rayos
    conj = list(rayos.CONJUNTOS) if con_rayos else []
    r_max = conj[-1] if conj else 0
    casos = [(n_vec_max, hor_max, h_max, n_f_max, r_max),  # el propio superset
             (2, 15, 3, 0, 0), (3, 10, 5, 0, 0), (1, 20, 2, 0, 0),
             (5, 15, 10, 0, 0),
             (3, 15, 5, min(4, n_f_max), 0), (2, 10, 3, min(1, n_f_max), 0)]
    # Uno por conjunto, alternando el resto de la vista para que ningun tramo se
    # compruebe siempre en el mismo sitio del vector.
    for i, n_r in enumerate(conj):
        n_f = (0, min(6, n_f_max), n_f_max)[i % 3]
        casos.append(((3, 2, 5)[i % 3], (15, 20, 10)[i % 3], (5, 3, 10)[i % 3],
                      n_f, n_r))
    for n_vec, horizonte, h_pasado, n_f, n_r in casos:
        if n_vec > n_vec_max or h_pasado > h_max or horizonte > hor_max:
            continue
        # construir_muestras devuelve (X, Y, M): la M es la etiqueta de modo de
        # cada fila, que aquí no se compara (no forma parte de la entrada).
        Xr, Yr = _muestras_referencia(runs, n_vec, horizonte, h_pasado, n_f,
                                      n_r)[:2]
        (Xn, Yn) = _muestras_nube(runs, n_vec, horizonte, h_pasado,
                                  n_vec_max, h_max, hor_max, n_f, n_f_max,
                                  n_r, con_rayos)
        etiqueta = (f"n_vecinos={n_vec} horizonte={horizonte} "
                    f"h_pasado={h_pasado} n_fourier={n_f} n_rayos={n_r}")
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


def verificar_rollout(rutas, n_vec, horizonte, h_pasado, n_fourier=0,
                      n_rayos=0):
    print("[2] rollout vectorizado  vs  politica.rollout_multiflota")
    import vectorizado as vec
    from politica import rollout_multiflota

    pol.configurar_representacion(n_vec, horizonte, h_pasado, n_fourier, None,
                                  n_rayos)
    obst_r, mundo_r = esc.obstaculos() if n_rayos else (None, None)
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
                        horizonte=horizonte, h_pasado=h_pasado,
                        n_fourier=n_fourier, n_rayos=n_rayos,
                        obstaculos=obst_r, mundo=mundo_r)
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
    ap.add_argument("--n-f-max", dest="n_f_max", type=int,
                    default=pol.N_FOURIER_MAX)
    ap.add_argument("--sin-rayos", dest="con_rayos", action="store_false",
                    default=True)
    args = ap.parse_args()

    print(f"[datos] leyendo {args.rutas} …")
    runs = ent.leer_runs(args.rutas)
    if not runs:
        raise SystemExit("No hay CSV de rutas con los que comparar.")
    print(f"[datos] {len(runs)} runs")

    ok = verificar_muestras(runs, args.n_vec_max, args.h_max, args.hor_max,
                            args.n_f_max, args.con_rayos)
    ok &= verificar_rollout(args.rutas, 3, 15, 5)
    # El rollout, otra vez con las ondas activas: la entrada la construyen dos
    # trozos de código distintos (politica.vector_entrada y
    # vectorizado.entradas_instante) y el bloque nuevo tiene que salir igual en
    # los dos, o la nota del barrido no mediría la red que se entrenó.
    ok &= verificar_rollout(args.rutas, 3, 15, 5, args.n_f_max)
    if args.con_rayos:
        # Y con RAYOS: el rollout los traza contra el mapa en cada paso y el
        # dataset los saca de las poses guardadas. Son dos caminos distintos
        # hacia el mismo numero, y tienen que dar exactamente el mismo.
        ok &= verificar_rollout(args.rutas, 3, 15, 5, 0, 12)
    print("\nRESULTADO:", "todo equivalente" if ok else "HAY DIFERENCIAS")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
