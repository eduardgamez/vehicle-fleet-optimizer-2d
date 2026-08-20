#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAYOS al obstáculo: cuánto espacio libre tiene el vehículo en cada dirección.

La red recibe su posición y tiene que deducir de ahí dónde están los muros,
porque el mapa no es una entrada. Eso funciona en teoría —el mapa es siempre el
mismo, así que la información está— pero en la práctica no: 241 configuraciones
del barrido y el 98 % de choques no se movió ni con más capacidad, ni dando la
posición en senos y cosenos, ni bajando el error de imitación.

Los rayos atacan otra cosa. No recodifican lo que la red ya tiene: le dan una
magnitud que no tenía. La distancia libre es lo que decide si chocas, y hasta
ahora había que deducirla de (x, y) más un mapa aprendido en los pesos. Con
esto llega hecha: "pared a 15 cm a la derecha".

CÓMO SE CALCULA. Trazar los rayos contra los 56 polígonos del mapa en cada
instante sale carísimo: el rollout de una candidata pasaría de 5 s a casi un
minuto, y el barrido entero se multiplicaría por diez. Pero el mapa NO CAMBIA
NUNCA, así que se traza una sola vez y se guarda en una tabla:

    TABLA[iy, ix, ia] = distancia libre desde el centro de la celda (ix, iy)
                        en la dirección ia

Después, cada consulta es leer una posición de memoria. La tabla se cachea en
disco y se rehace sola si cambia el mapa o la resolución.

LA MISMA TABLA LA USAN LOS TRES SITIOS que construyen la entrada de la red
(`politica.vector_entrada` para el despliegue, `vectorizado.superset_run` para
el dataset y `vectorizado.entradas_instante` para el rollout). Tiene que ser
así: si el entrenamiento y la evaluación midieran los rayos de forma distinta
—uno exacto y otro aproximado—, la nota estaría midiendo una red que ve otra
cosa. `verificar.py` lo comprueba exigiendo diferencia CERO entre los tres.

La discretización mete un error de hasta media celda en posición y de medio
paso en ángulo. Es aceptable porque esto es una PISTA para la red, no una
comprobación de choque: los choques se siguen decidiendo con SAT exacto sobre
las esquinas reales, aquí no se toca nada de eso.
"""

import hashlib
import math
import os

import numpy as np

from nucleo import W, H

# Resolución de la tabla. 0,1 de celda deja el error de posición en 5
# centésimas y 3 grados de paso angular dan unos 5 cm de desvío lateral a un
# metro de distancia, ambos por debajo de la holgura con la que conduce el
# planificador (10-20 cm). Subir la resolución crece en memoria al cubo y no
# cambia nada: esto alimenta una red, no un detector de colisiones.
CELDA = 0.1
N_ANGULOS = 120

# Más allá de esto un obstáculo ya no condiciona la maniobra, y saturar evita
# que la entrada tenga un rango enorme por los rayos que apuntan al vacío. Son
# unos 8 largos de vehículo.
D_MAX = 8.0

# Números de rayos que puede usar una configuración. El superset guarda LOS
# TRES conjuntos, uno detrás de otro, y cada punto del barrido se queda con el
# suyo. No se puede recortar como se hace con las ondas —quedarse con los
# primeros—: ocho rayos repartidos por la circunferencia no son un subconjunto
# de dieciocho, son direcciones distintas.
CONJUNTOS = (8, 12, 18)

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modelos",
                     "tabla_rayos.npz")

_TABLA = None
_FIRMA = None


def _firma(oc, mundo):
    """Huella del mapa y de la resolución: si cambia cualquiera, la tabla
    guardada ya no vale y hay que rehacerla."""
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(oc, dtype=np.float64).tobytes())
    h.update(repr((tuple(mundo), CELDA, N_ANGULOS, D_MAX)).encode())
    return h.hexdigest()


def _distancias_a_segmentos(px, py, dx, dy, a, b, d_max):
    """Distancia del primer corte de cada rayo con cada segmento.

    'px, py' (P,) son los orígenes y 'dx, dy' (P,) las direcciones unitarias;
    'a' y 'b' (S, 2) los extremos de los segmentos. Devuelve (P,) con la
    distancia al corte más cercano, o d_max si el rayo no corta nada.

    Es la fórmula de siempre: se resuelve  o + t·d = a + u·(b − a)  y el corte
    vale solo si t >= 0 (hacia delante) y 0 <= u <= 1 (dentro del segmento)."""
    ex = b[:, 0] - a[:, 0]
    ey = b[:, 1] - a[:, 1]
    den = dx[:, None] * ey[None, :] - dy[:, None] * ex[None, :]
    # Paralelos: sin corte. Se marca con un denominador que no divida por cero.
    ok = np.abs(den) > 1e-12
    den = np.where(ok, den, 1.0)
    qx = a[None, :, 0] - px[:, None]
    qy = a[None, :, 1] - py[:, None]
    t = (qx * ey[None, :] - qy * ex[None, :]) / den
    u = (qx * dy[:, None] - qy * dx[:, None]) / den
    vale = ok & (t >= 0.0) & (u >= 0.0) & (u <= 1.0)
    t = np.where(vale, t, np.inf)
    return np.minimum(t.min(axis=1), d_max)


def _segmentos_del_mapa(oc, mundo):
    """(S, 2, 2) con todos los lados de los obstáculos más los cuatro bordes
    del mundo. Los polígonos vienen con vértices repetidos cuando tienen menos
    de cuatro lados; esos lados degenerados se caen solos porque no cortan."""
    seg = []
    for poly in np.asarray(oc, dtype=np.float64):
        for i in range(len(poly)):
            a, b = poly[i], poly[(i + 1) % len(poly)]
            if abs(a[0] - b[0]) > 1e-9 or abs(a[1] - b[1]) > 1e-9:
                seg.append((a, b))
    an, al = mundo
    for a, b in (((0.0, 0.0), (an, 0.0)), ((an, 0.0), (an, al)),
                 ((an, al), (0.0, al)), ((0.0, al), (0.0, 0.0))):
        seg.append((np.array(a), np.array(b)))
    return np.array(seg, dtype=np.float64)


def construir(oc, mundo, verbose=True):
    """Tabla (ny, nx, N_ANGULOS) de distancia libre. Tarda un rato y por eso se
    cachea: es el único momento en que se traza contra la geometría real."""
    seg = _segmentos_del_mapa(oc, mundo)
    a, b = seg[:, 0], seg[:, 1]
    an, al = mundo
    nx = int(math.ceil(an / CELDA))
    ny = int(math.ceil(al / CELDA))
    # Centros de celda: la consulta redondea a la celda que contiene el punto,
    # así que el valor guardado tiene que ser el de su centro.
    xs = (np.arange(nx) + 0.5) * CELDA
    ys = (np.arange(ny) + 0.5) * CELDA
    gx, gy = np.meshgrid(xs, ys)
    px, py = gx.ravel(), gy.ravel()

    tabla = np.empty((ny * nx, N_ANGULOS), dtype=np.float32)
    for ia in range(N_ANGULOS):
        th = 2.0 * math.pi * ia / N_ANGULOS
        dx = np.full(px.shape, math.cos(th))
        dy = np.full(px.shape, math.sin(th))
        tabla[:, ia] = _distancias_a_segmentos(px, py, dx, dy, a, b,
                                               D_MAX).astype(np.float32)
        if verbose and (ia + 1) % 20 == 0:
            print("[rayos] tabla %d/%d angulos" % (ia + 1, N_ANGULOS),
                  flush=True)
    return tabla.reshape(ny, nx, N_ANGULOS)


def tabla(oc, mundo, cache=CACHE, verbose=True):
    """Tabla de distancias, de memoria, de disco o recién construida."""
    global _TABLA, _FIRMA
    f = _firma(oc, mundo)
    if _TABLA is not None and _FIRMA == f:
        return _TABLA
    if cache and os.path.exists(cache):
        try:
            d = np.load(cache)
            if str(d["firma"]) == f:
                _TABLA, _FIRMA = d["tabla"], f
                return _TABLA
        except (OSError, KeyError, ValueError):
            pass                      # cache ilegible o de otro mapa: se rehace
    if verbose:
        print("[rayos] construyendo la tabla de distancias (una sola vez)…",
              flush=True)
    t = construir(oc, mundo, verbose=verbose)
    if cache:
        try:
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            np.savez_compressed(cache, tabla=t, firma=f)
        except OSError:
            pass
    _TABLA, _FIRMA = t, f
    return t


def distancias(x, y, th, n_rayos, oc, mundo, verbose=False):
    """(..., n_rayos) con la distancia libre NORMALIZADA en [0, 1].

    Los rayos salen del centro del vehículo y se reparten por igual alrededor,
    empezando por su morro: el rayo k apunta a th + 2πk/n. Van en el sistema
    del VEHÍCULO, no del mapa, así que 'el de delante' es siempre la misma
    columna gire hacia donde gire —que es lo que hace la señal aprovechable—.

    Se devuelve d/D_MAX y no la distancia cruda para que entre acotada en [0, 1]
    igual que el resto del vector, y para que los rayos que apuntan al vacío se
    queden todos en 1 en vez de disparar el rango de la entrada."""
    t = tabla(oc, mundo, verbose=verbose)
    ny, nx, na = t.shape
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    tha = np.asarray(th, dtype=np.float64)
    n = int(n_rayos)
    if n <= 0:
        return np.zeros(xa.shape + (0,), dtype=np.float32)

    ix = np.clip((xa / CELDA).astype(np.int64), 0, nx - 1)
    iy = np.clip((ya / CELDA).astype(np.int64), 0, ny - 1)
    k = np.arange(n)
    ang = tha[..., None] + (2.0 * math.pi) * k / n
    ia = np.mod(np.rint(ang * N_ANGULOS / (2.0 * math.pi)).astype(np.int64), na)
    d = t[iy[..., None], ix[..., None], ia]
    return (d / D_MAX).astype(np.float32)


def bloque_superset(x, y, th, oc, mundo, verbose=False):
    """Los tres conjuntos de rayos, uno detrás de otro (8, luego 12, luego 18).

    El superset los guarda todos porque no se pueden derivar unos de otros, y
    cada configuración del barrido se queda con el tramo que le toca."""
    partes = [distancias(x, y, th, n, oc, mundo, verbose) for n in CONJUNTOS]
    return np.concatenate(partes, axis=-1)


def ancho_superset():
    return sum(CONJUNTOS)


def tramo(n_rayos):
    """(inicio, fin) del conjunto de 'n_rayos' dentro del bloque del superset."""
    ini = 0
    for n in CONJUNTOS:
        if n == n_rayos:
            return ini, ini + n
        ini += n
    if n_rayos == 0:
        return 0, 0
    raise ValueError("n_rayos=%r no es uno de %r" % (n_rayos, CONJUNTOS))
