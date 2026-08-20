#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ENCADENA las fases sin que haya que estar delante.

Vigila el barrido de la fase 1 y, cuando termina, hace por su cuenta lo que
venía después: recoger las configuraciones que no cupieron en la tarjeta,
escribir el resumen de lo aprendido y arrancar la fase 2.

  1) espera a que la pasada 1 (3 procesos) llegue al objetivo o se estanque,
  2) PARA el panel de la fase 1 —si no, relanzaría trabajadores por su cuenta y
     se juntarían con los de la cascada: más de 3 procesos a la vez tumban el
     driver de la tarjeta—,
  3) pasadas 2 y 3: las que no cupieron, con más memoria por proceso y menos
     procesos a la vez. Se hacen SIEMPRE que quede alguna, aunque sea una: una
     configuración que no cabe NO se anota, así que sigue pendiente y la recoge
     la pasada siguiente, y las ya hechas no se repiten. Bajar de 3 procesos
     aquí no es por estabilidad —3 va bien— sino porque una red grande no cabe
     en su trozo de tarjeta y hay que darle más,
  4) escribe datos/modelos/resumen_fase1.txt con lo que respondía el barrido:
     choques por tamaño de red y por nº de ondas de la posición,
  5) arranca el panel de la fase 2, que lanza los trabajadores de Optuna.

Si se apaga el ordenador a media noche esto muere con él, y al encender vuelve
solo el panel de la fase 1 (el del acceso directo), no este vigilante: hay que
relanzarlo a mano.

Uso:
    python encadenar.py                 (por defecto: 240 configuraciones)
    python encadenar.py --sin-cascada   (salta las pasadas 2 y 3)
    python encadenar.py --sin-fase2     (solo vigila y resume)
"""

import argparse
import csv
import glob
import os
import subprocess
import sys
import time

import comun
from comun import MODELOS_DIR

RAIZ = os.path.dirname(os.path.abspath(__file__))
GUION_BARRIDO = os.path.join(RAIZ, "entrenar_nube.py")
PANEL_FASE1 = os.path.join(RAIZ, "panel_fase1.py")
PANEL_FASE2 = os.path.join(RAIZ, "panel.py")
TAREAS = 3


# --------------------------------------------------------------------------- #
# Estado del barrido
# --------------------------------------------------------------------------- #
def filas_hechas(carpeta):
    """Configuraciones ya evaluadas en todos los registros de la fase 1."""
    n = 0
    for f in glob.glob(os.path.join(carpeta, "barrido_t*.csv")):
        try:
            with open(f, encoding="utf-8", newline="") as fh:
                n += sum(1 for _ in csv.DictReader(fh))
        except OSError:
            pass
    return n


def procesos(patron):
    """Procesos de python vivos cuya línea de comandos menciona 'patron'.

    Se descarta el hijo cuando su padre también aparece: el python.exe del
    entorno virtual es un lanzador y cada proceso saldría por duplicado."""
    import psutil
    enc = {}
    yo = os.getpid()
    for p in psutil.process_iter(["pid", "ppid", "name", "cmdline"]):
        if p.info["pid"] == yo:
            continue
        if not (p.info.get("name") or "").lower().startswith("python"):
            continue
        cmd = " ".join(str(a) for a in (p.info.get("cmdline") or []))
        if "encadenar.py" in cmd:
            continue
        if patron in cmd:
            enc[p.info["pid"]] = p
    return [p for pid, p in enc.items() if p.info.get("ppid") not in enc]


def parar(patron, espera=10):
    import psutil
    vivos = procesos(patron)
    for p in vivos:
        try:
            p.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    psutil.wait_procs(vivos, timeout=espera)
    for p in procesos(patron):
        try:
            p.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return len(vivos)


def esperar_fase1(carpeta, objetivo, estancamiento, latido=60):
    """Bloquea hasta que la pasada 1 llega al objetivo o deja de avanzar.

    Lo segundo hace falta porque el barrido puede terminar con MENOS del
    objetivo: las configuraciones que no caben en la tarjeta no se anotan, y
    entonces el contador se queda corto para siempre. El estancamiento tiene que
    ser holgado, que una red grande tarda sus buenos minutos en entrenarse."""
    ultimo, cambio = filas_hechas(carpeta), time.time()
    print("[enc] vigilando la fase 1: %d/%d" % (ultimo, objetivo), flush=True)
    while True:
        time.sleep(latido)
        n = filas_hechas(carpeta)
        if n >= objetivo:
            print("[enc] fase 1 al completo: %d/%d" % (n, objetivo), flush=True)
            return n
        if n != ultimo:
            ultimo, cambio = n, time.time()
            print("[enc] %d/%d" % (n, objetivo), flush=True)
        elif time.time() - cambio > estancamiento:
            print("[enc] la fase 1 lleva %.0f min sin avanzar en %d/%d: se da "
                  "por terminada. Faltan %d, que son las que no cupieron en la "
                  "tarjeta." % (estancamiento / 60.0, n, objetivo, objetivo - n),
                  flush=True)
            return n


# --------------------------------------------------------------------------- #
# Pasadas de la cascada
# --------------------------------------------------------------------------- #
def lanzar_barrido(idt, n_configs, frac, escenarios):
    cmd = [sys.executable, GUION_BARRIDO, "--n-configs", str(n_configs),
           "--semilla", "0", "--tarea", str(idt), "--tareas", str(TAREAS),
           "--criba=", "--n-escenarios", str(escenarios),
           "--frac-vram", str(frac)]
    sin_ventana = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return subprocess.Popen(cmd, cwd=RAIZ, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            creationflags=sin_ventana)


def pasada(a_la_vez, frac, n_configs, escenarios, nombre):
    """Ejecuta las 3 tareas en grupos de 'a_la_vez', con 'frac' de tarjeta cada
    una. Bloquea hasta que termina el último grupo."""
    print("[enc] %s: de %d en %d, %s de tarjeta cada una"
          % (nombre, a_la_vez, a_la_vez, frac), flush=True)
    for base in range(0, TAREAS, a_la_vez):
        grupo = [lanzar_barrido(i, n_configs, frac, escenarios)
                 for i in range(base, min(base + a_la_vez, TAREAS))]
        for p in grupo:
            p.wait()
    print("[enc] %s: terminada" % nombre, flush=True)


# --------------------------------------------------------------------------- #
# Resumen de la fase 1
# --------------------------------------------------------------------------- #
def _num(fila, clave, defecto=0.0):
    try:
        return float(fila[clave])
    except (KeyError, TypeError, ValueError):
        return defecto


def resumen_fase1(carpeta):
    """Texto con lo que el barrido tenía que responder: si chocar depende del
    TAMAÑO de la red y si las ONDAS de la posición cambian algo."""
    filas = []
    for f in sorted(glob.glob(os.path.join(carpeta, "barrido_t*.csv"))):
        try:
            with open(f, encoding="utf-8", newline="") as fh:
                filas += list(csv.DictReader(fh))
        except OSError:
            pass
    if not filas:
        return "No hay ninguna configuracion evaluada.", [], []

    def media(g, c):
        return sum(_num(x, c) for x in g) / len(g) if g else 0.0

    out = ["FASE 1 . %d configuraciones evaluadas" % len(filas), ""]

    out.append("Por NUMERO DE ONDAS de la posicion (n_fourier):")
    out.append("  %6s %4s %7s %8s %7s %12s"
               % ("ondas", "n", "nota", "%choque", "%llega", "seg tocando"))
    hay_ondas = False
    for v in sorted({x.get("n_fourier", "0") for x in filas},
                    key=lambda s: int(s or 0)):
        g = [x for x in filas if x.get("n_fourier", "0") == v]
        if not g:
            continue
        hay_ondas = True
        out.append("  %6s %4d %7.4f %8.1f %7.1f %12.1f"
                   % (v, len(g), media(g, "nota_seleccion"),
                      media(g, "pct_choque"), media(g, "pct_llegada"),
                      media(g, "seg_choque")))
    if not hay_ondas:
        out.append("  (el registro no trae la columna)")

    out.append("")
    out.append("Por TAMANO de la red (parametros ~ oculto^2 x (capas-1)):")
    out.append("  %8s %4s %7s %8s %7s %12s"
               % ("tamano", "n", "nota", "%choque", "%llega", "seg tocando"))
    tramos = [("<0.3M", 0, 3e5), ("0.3-2M", 3e5, 2e6),
              ("2-10M", 2e6, 1e7), (">10M", 1e7, float("inf"))]
    for nombre, lo, hi in tramos:
        g = []
        for x in filas:
            p = _num(x, "oculto") ** 2 * max(_num(x, "n_capas") - 1, 0)
            if lo <= p < hi:
                g.append(x)
        if not g:
            continue
        out.append("  %8s %4d %7.4f %8.1f %7.1f %12.1f"
                   % (nombre, len(g), media(g, "nota_seleccion"),
                      media(g, "pct_choque"), media(g, "pct_llegada"),
                      media(g, "seg_choque")))

    out.append("")
    out.append("Las 10 MEJORES por nota:")
    filas.sort(key=lambda x: -_num(x, "nota_seleccion"))
    out.append("  %7s %8s %7s %6s %6s %7s %7s %10s"
               % ("nota", "%choque", "%llega", "ondas", "capas", "oculto",
                  "lr", "mezcla"))
    for x in filas[:10]:
        out.append("  %7.4f %8.1f %7.1f %6s %6s %7s %7s %10s"
                   % (_num(x, "nota_seleccion"), _num(x, "pct_choque"),
                      _num(x, "pct_llegada"), x.get("n_fourier", "?"),
                      x.get("n_capas", "?"), x.get("oculto", "?"),
                      x.get("lr", "?"), x.get("mezcla", "?")))

    out.append("")
    out.append("COMO LEERLO: si la columna %choque sale igual con 0 ondas que")
    out.append("con 12, las ondas no eran el problema. Lo mismo con el tamano:")
    out.append("si una red de >10M choca como una de <0.3M, no falta capacidad.")

    texto_ejes, bordes, planos = analizar_ejes(filas)
    out.append("")
    out.append(texto_ejes)
    return "\n".join(out), bordes, planos


# --------------------------------------------------------------------------- #
# Ajuste del espacio para la fase 2
# --------------------------------------------------------------------------- #
# Ejes con un orden natural: en estos tiene sentido preguntarse si el mejor
# valor se ha quedado en un EXTREMO de la lista, porque entonces lo bueno puede
# estar más allá y el intervalo se queda corto. En los demás (activacion,
# mezcla…) no hay "más allá": los valores son nombres, no cantidades.
EJES_ORDENABLES = ("n_capas", "oculto", "dropout", "lr", "lote", "n_vecinos",
                   "h_pasado", "horizonte", "weight_decay", "n_fourier")

# Suelo NATURAL de algunos ejes: por debajo no hay nada que probar. Que lo mejor
# sea "sin dropout" o "sin weight_decay" no es que el intervalo se quede corto,
# es que apagar esa regularización es lo que conviene. Sin esto, cada barrido
# pediría ampliar hacia abajo un eje que ya está en su mínimo.
SUELO_NATURAL = {"dropout": 0.0, "weight_decay": 0.0, "n_fourier": 0.0}


def _espacio_actual():
    """Listas de valores de cada eje tal y como están AHORA en el código."""
    try:
        import entrenar_nube as EN
        return {k: v for k, v in EN.ESPACIO.items()}
    except Exception:
        return {}


ESPACIO_ACTUAL = _espacio_actual()


def analizar_ejes(filas):
    """Nota media por valor de cada eje, y dos avisos para la fase 2:

      · BORDE: el mejor valor de un eje ordenable es el primero o el último de
        su lista, y le saca al resto más que el ruido. Conviene ALARGAR la lista
        en esa dirección antes de empezar, porque el óptimo puede estar fuera.
      · PLANO: entre el mejor y el peor valor del eje no hay más diferencia que
        el ruido. Iterar sobre él es gastar pruebas en algo que no decide nada.

    El ruido se estima con el error estándar de la media de cada grupo: con
    pocas configuraciones por valor, diferencias pequeñas no significan nada y
    no hay que tocar el espacio por ellas."""
    import math

    def num(s):
        try:
            return float(s)
        except (TypeError, ValueError):
            return None

    out = ["AJUSTE DEL ESPACIO PARA LA FASE 2", ""]
    bordes, planos = [], []
    ejes = [k for k in filas[0] if k in EJES_ORDENABLES or k in
            ("activacion", "normalizacion", "optimizador", "mezcla")]
    for eje in ejes:
        grupos = {}
        for x in filas:
            grupos.setdefault(x.get(eje), []).append(_num(x, "nota_seleccion"))
        grupos = {k: v for k, v in grupos.items() if len(v) >= 3}
        if len(grupos) < 2:
            continue
        est = {}
        for k, v in grupos.items():
            m = sum(v) / len(v)
            var = sum((y - m) ** 2 for y in v) / max(len(v) - 1, 1)
            est[k] = (m, math.sqrt(var / len(v)), len(v))
        orden = sorted(est, key=lambda k: -est[k][0])
        mejor, peor = orden[0], orden[-1]
        dif = est[mejor][0] - est[peor][0]
        ruido = math.hypot(est[mejor][1], est[peor][1])

        etiqueta = ""
        if dif <= ruido:
            etiqueta = "PLANO"
            planos.append(eje)
        elif eje in EJES_ORDENABLES:
            # El extremo se mide contra el ESPACIO DE BUSQUEDA de ahora, no
            # contra los valores que aparecen en el registro. Si no, al ampliar
            # un eje el aviso no se apagaria nunca: los datos viejos siguen sin
            # tener los valores nuevos, asi que su maximo seguiria pareciendo
            # el tope aunque ya se haya alargado la lista.
            del_espacio = ESPACIO_ACTUAL.get(eje)
            if del_espacio:
                valores = [str(v) for v in sorted(del_espacio,
                                                  key=lambda v: float(v))]
            else:
                valores = sorted(grupos, key=lambda s: (num(s) is None, num(s)))
            suelo = SUELO_NATURAL.get(eje)
            en_suelo = suelo is not None and num(mejor) is not None and                 num(mejor) <= suelo
            if mejor == valores[0] and not en_suelo:
                etiqueta = "BORDE (por abajo)"
                bordes.append((eje, "abajo", mejor))
            elif mejor == valores[0]:
                etiqueta = "mejor en su minimo natural (nada que ampliar)"
            elif mejor == valores[-1]:
                etiqueta = "BORDE (por arriba)"
                bordes.append((eje, "arriba", mejor))

        out.append("%-14s mejor=%-9s (%.4f)  peor=%-9s (%.4f)  ruido=%.4f  %s"
                   % (eje, mejor, est[mejor][0], peor, est[peor][0], ruido,
                      etiqueta))

    out.append("")
    if bordes:
        out.append("HAY QUE AMPLIAR antes de la fase 2:")
        for eje, lado, val in bordes:
            out.append("  · %s: lo mejor esta en el extremo de %s (%s). "
                       "Alargar la lista en esa direccion." % (eje, lado, val))
    else:
        out.append("Ningun eje se sale por un extremo: los intervalos cubren "
                   "lo bueno.")
    if planos:
        out.append("")
        out.append("SIN EFECTO MEDIBLE (candidatos a fijar en un solo valor y "
                   "dejar de gastar pruebas en ellos):")
        out.append("  " + ", ".join(planos))
        out.append("  Ojo: 'sin efecto en la NOTA MEDIA' no es 'inutil'. Un eje "
                   "puede no mover la media y si importar en las mejores.")
    return "\n".join(out), bordes, planos


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="Encadena fase 1 -> cascada -> fase 2 sin supervision.")
    ap.add_argument("--registros", default=MODELOS_DIR)
    ap.add_argument("--n-configs", dest="n_configs", type=int, default=240)
    ap.add_argument("--n-escenarios", dest="n_escenarios", type=int, default=400)
    ap.add_argument("--estancamiento", type=int, default=1800,
                    help="segundos sin avanzar tras los que se da por acabada "
                         "la fase 1 (def. 30 min)")
    ap.add_argument("--sin-cascada", dest="sin_cascada", action="store_true",
                    help="no recoger las que no cupieron en la tarjeta")
    ap.add_argument("--sin-fase2", dest="sin_fase2", action="store_true",
                    help="parar tras el resumen, sin arrancar Optuna")
    ap.add_argument("--forzar-fase2", dest="forzar_fase2", action="store_true",
                    help="arrancar la fase 2 aunque algun eje se salga por un "
                         "extremo (por defecto se para para poder ampliarlo)")
    args = ap.parse_args()

    esperar_fase1(args.registros, args.n_configs, args.estancamiento)

    # El panel de la fase 1 relanza trabajadores caidos cada 20 s. Si sigue en
    # pie durante la cascada, sus 3 procesos se suman a los de la pasada y se
    # pasa del limite que aguanta el driver.
    n = parar("panel_fase1.py")
    print("[enc] panel de la fase 1 parado (%d)" % n, flush=True)
    parar("entrenar_nube.py")

    if not args.sin_cascada:
        pendientes = args.n_configs - filas_hechas(args.registros)
        if pendientes <= 0:
            print("[enc] no queda ninguna pendiente: no hace falta cascada",
                  flush=True)
        else:
            print("[enc] %d sin evaluar (no cupieron en la tarjeta): cascada"
                  % pendientes, flush=True)
            pasada(2, 0.44, args.n_configs, args.n_escenarios,
                   "Pasada 2 - las que no cupieron")
            if args.n_configs - filas_hechas(args.registros) > 0:
                pasada(1, 0.90, args.n_configs, args.n_escenarios,
                       "Pasada 3 - las mas grandes, una a una")

    texto, bordes, planos = resumen_fase1(args.registros)
    ruta = os.path.join(args.registros, "resumen_fase1.txt")
    with open(ruta, "w", encoding="utf-8") as f:
        f.write(texto + "\n")
    print(texto, flush=True)
    print("\n[enc] resumen en %s" % ruta, flush=True)

    if args.sin_fase2:
        print("[enc] hecho (sin arrancar la fase 2)", flush=True)
        return

    # Si algun eje se sale por un extremo, NO se arranca la fase 2. Ampliar el
    # espacio despues es peor que esperar: Optuna guarda con cada prueba la
    # lista de valores de cada eje, y cambiarla a mitad de estudio invalida lo
    # ya hecho o exige empezar otro. Mas vale ampliar ahora y arrancar una vez.
    if bordes and not args.forzar_fase2:
        print("[enc] FASE 2 EN ESPERA: %d eje(s) apuntan fuera del intervalo "
              "(%s). Ampliar entrenar_nube.ESPACIO y relanzar con "
              "--forzar-fase2." % (len(bordes), ", ".join(e for e, _, _ in bordes)),
              flush=True)
        return

    sin_ventana = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    suelto = getattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0)
    subprocess.Popen([sys.executable, PANEL_FASE2, "--sin-abrir"], cwd=RAIZ,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     creationflags=sin_ventana | suelto)
    print("[enc] fase 2 en marcha . panel en http://localhost:8770", flush=True)


if __name__ == "__main__":
    main()
