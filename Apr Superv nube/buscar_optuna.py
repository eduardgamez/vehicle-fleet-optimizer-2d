#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FASE 2 · Búsqueda INTELIGENTE de hiperparámetros (Optuna).

La fase 1 probó configuraciones al azar. Eso está bien para empezar —dice qué
ejes importan y no se deja ninguna zona sin mirar— pero es tonto: tira el mismo
esfuerzo en una zona que ya se sabe mala que en una prometedora. Aquí se cambian
las dos cosas que hacen que una búsqueda rinda:

  1. QUÉ se prueba. Un muestreador bayesiano (TPE) mira todo lo evaluado hasta
     ahora y propone la siguiente configuración donde es más probable que haya
     algo bueno. Arranca ya enseñado: las 207 configuraciones de la fase
     aleatoria se cargan en el estudio antes de empezar (ver `sembrar`), así que
     no hay que pagar otra vez el periodo en el que aún no sabe nada.

  2. CUÁNTO se gasta en cada una. Con ASHA, una configuración empieza con pocas
     épocas y solo se gana más si a los hitos va mejor que la mayoría de las que
     llegaron ahí. La que va la última a mitad de camino se abandona en vez de
     agotar su presupuesto. Explorar miles sale por lo que costaría entrenar
     unos cientos.

Los datos son los mismos que en la fase 1: el dataset COMPLETO (612 k muestras).
La fracción de dataset solo se toca si hace falta por tiempo; el recorte que
hace ASHA es en épocas, que es más limpio —todas las configuraciones se comparan
con los mismos datos y solo cambia cuánto tiempo se les da.

El estudio vive en una base de datos SQLite, así que se pueden lanzar VARIOS
procesos a la vez contra el mismo estudio: cada uno pide la siguiente
configuración y todos comparten lo aprendido. Y se puede apagar el ordenador
cuando sea: al relanzar, sigue donde estaba.

Uso:
    python buscar_optuna.py --n-pruebas 3500
    python buscar_optuna.py --n-pruebas 3500 --frac-vram 0.27   (uno de varios)
    python buscar_optuna.py --resumen                           (solo mirar)
"""

import argparse
import csv
import glob
import json
import os
import time

import numpy as np

import comun
from comun import MODELOS_DIR, MUESTRAS_DIR, asegurar
import entrenar_nube as EN
import escenarios as esc
import politica as pol

import entrenar as ent

# Ejes y valores: los del barrido aleatorio MENOS las épocas.
#
# Las épocas dejan de ser un eje de búsqueda a propósito. Eran a la vez algo que
# el muestreador elegía (40 u 80) y algo que los peajes recortaban, y eso es
# redundante: dos candidatas idénticas salvo por el presupuesto competían en el
# mismo peaje sin que la diferencia significase nada. Ahora todas nacen con el
# mismo máximo y son los peajes los que deciden hasta dónde llega cada una, que
# es para lo que están. Un eje menos que explorar y comparaciones limpias.
# 160 y no 80 desde el 20/08/2026. Con 80, las redes grandes se quedaban a
# medio entrenar: mirando los peajes de las pruebas ya hechas, las de más de 10M
# de parámetros SEGUÍAN mejorando en el último tramo el 78 % de las veces
# (+0,0109 de nota), frente al 55-66 % de las pequeñas. El récord del estudio
# llevaba nueve horas sin moverse y quinientas pruebas dando vueltas por el
# mismo sitio: el techo lo ponía el presupuesto, no la búsqueda.
#
# OJO al comparar: las pruebas anteriores a este cambio se midieron con 80. Sus
# notas siguen en el estudio a propósito —son el mapa que ya tiene TPE— pero
# están en desventaja, así que una zona vieja que parezca mala puede serlo solo
# por haber entrenado la mitad.
EPOCAS_MAX = 160

ESPACIO = {k: v for k, v in EN.ESPACIO.items() if k != "epocas"}

# Historial de estudios (cada cambio de reglas obliga a empezar uno nuevo,
# porque las notas de antes ya no son comparables):
#   tdr_flota    — notas de redes cortadas a medias, inservible.
#   tdr_flota_v2 — limpio, pero con las épocas todavía como eje de búsqueda.
#   tdr_flota_v3 — épocas fijas en EPOCAS_MAX y peajes decididos con la nota
#                  real (conduciendo las flotas) en vez de con el error de
#                  validación.
NOMBRE_ESTUDIO = "tdr_flota_v3"


# --------------------------------------------------------------------------- #
# Siembra: meter en el estudio lo que ya se sabe
# --------------------------------------------------------------------------- #
# Valor que se le supone a un eje cuando el registro NO trae su columna. Pasa
# con los ejes añadidos después de una tanda: las 241 configuraciones del primer
# barrido se evaluaron antes de que existieran los rayos, y su columna no está.
# Descartarlas seria tirar el grupo de CONTROL —justamente las que se corrieron
# sin rayos—, asi que se les pone el valor que de hecho tenian: ninguno.
NEUTRO = {"n_rayos": 0, "n_fourier": 0}


def _leer_barridos(carpeta):
    """Filas de los CSV de la fase aleatoria (una por configuración evaluada).

    Busca tambien en las SUBCARPETAS: las tandas viejas se apartan ahi para no
    mezclar registros con columnas distintas, pero sus notas siguen valiendo
    para sembrar."""
    filas = []
    patrones = [os.path.join(carpeta, "barrido_t*.csv"),
                os.path.join(carpeta, "*", "barrido_t*.csv")]
    for f in sorted(set(sum((glob.glob(pa) for pa in patrones), []))):
        # Una subcarpeta con este fichero guarda notas que YA NO VALEN (se
        # midieron con otra vara: sin mirar colisiones, o con la escala de
        # choque que se hundia a cero). Sembrar con ellas le ensenaria al
        # muestreador justo lo contrario de lo que debe.
        if os.path.exists(os.path.join(os.path.dirname(f), "NO_SEMBRAR.txt")):
            continue
        with open(f, encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                try:
                    r["_nota"] = float(r["nota_seleccion"])
                except (KeyError, TypeError, ValueError):
                    continue
                filas.append(r)
    return filas


def _valor_del_eje(eje, texto):
    """Convierte el texto del CSV al tipo que tiene ese eje en ESPACIO."""
    muestra = ESPACIO[eje][0]
    if isinstance(muestra, str):
        return texto
    if isinstance(muestra, bool):
        return texto.lower() in ("1", "true", "si", "sí")
    if isinstance(muestra, int):
        return int(float(texto))
    return float(texto)


def sembrar(estudio, carpeta, verbose=True):
    """Añade al estudio las configuraciones ya evaluadas en la fase aleatoria,
    con su nota. TPE las usa como experiencia previa: la búsqueda empieza
    apuntando a las zonas buenas en vez de a ciegas.

    Se saltan las que usan valores que ya no están en el espacio (la fase 1
    probó 'sgd' y redes de 6 capas, que se han descartado) y las repetidas."""
    import optuna
    from optuna.distributions import CategoricalDistribution
    from optuna.trial import TrialState

    dists = {k: CategoricalDistribution(v) for k, v in ESPACIO.items()}
    ya = set()
    for t in estudio.get_trials(deepcopy=False):
        ya.add(tuple(str(t.params.get(k)) for k in ESPACIO))

    nuevas, descartadas = 0, 0
    for fila in _leer_barridos(carpeta):
        try:
            params = {k: _valor_del_eje(k, fila[k]) if fila.get(k) is not None
                      else NEUTRO[k] for k in ESPACIO}
        except (KeyError, ValueError):
            descartadas += 1
            continue
        if any(params[k] not in ESPACIO[k] for k in ESPACIO):
            descartadas += 1          # usa un valor ya retirado del espacio
            continue
        # Solo sirven las entrenadas con el presupuesto que ahora es el único
        # (EPOCAS_MAX): una nota sacada con la mitad de épocas no es comparable.
        try:
            if int(float(fila.get("epocas", 0))) != EPOCAS_MAX:
                descartadas += 1
                continue
        except (TypeError, ValueError):
            descartadas += 1
            continue
        clave = tuple(str(params[k]) for k in ESPACIO)
        if clave in ya:
            continue
        ya.add(clave)
        estudio.add_trial(optuna.trial.create_trial(
            params=params, distributions=dists, value=fila["_nota"],
            state=TrialState.COMPLETE))
        nuevas += 1
    if verbose:
        print(f"[optuna] siembra: {nuevas} pruebas de la fase aleatoria "
              f"añadidas ({descartadas} descartadas por usar valores ya "
              f"retirados)", flush=True)
    return nuevas


def _tomar_cerrojo(carpeta, nombre="siembra.lock"):
    """True solo para el PRIMER proceso que lo pide. Ver el comentario en main.

    No se borra al terminar a propósito: el cerrojo marca "esta carpeta ya se
    sembró", así que una segunda tanda de trabajadores (tras un corte, o al
    añadir procesos) tampoco vuelve a meter la fase 1 encima de lo que ya está.
    Para volver a sembrar de cero hay que borrarlo a mano, que es una decisión
    lo bastante seria como para pedir ese gesto."""
    ruta = os.path.join(carpeta, nombre)
    try:
        fd = os.open(ruta, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    except OSError:
        return False
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write("sembrado por el pid %d" % os.getpid())
    return True


def sembrar_de_estudio(estudio, nombre_origen, almacen, verbose=True):
    """Trae de otro estudio SOLO las pruebas cuya nota es de fiar: las que
    entrenaron todas las épocas que pedía su configuración.

    Hace falta porque un estudio puede acumular notas de redes cortadas a media
    carrera (ver el comentario en `aviso`). Esas notas son bajas por no haber
    entrenado, no por ser malas configuraciones, y dejarlas dentro hace que el
    muestreador aprenda justo lo contrario de lo que debe."""
    import optuna
    from optuna.distributions import CategoricalDistribution
    from optuna.trial import TrialState

    try:
        viejo = optuna.load_study(study_name=nombre_origen, storage=almacen)
    except KeyError:
        print(f"[optuna] no existe el estudio '{nombre_origen}', nada que traer")
        return 0

    dists = {k: CategoricalDistribution(v) for k, v in ESPACIO.items()}
    # Aquí NO se quitan configuraciones repetidas, al contrario que en la
    # siembra desde los CSV: la misma configuración entrenada dos veces da notas
    # distintas (el arranque de la red es aleatorio) y esa discrepancia es
    # información buena — le dice al muestreador cuánto de lo que ve es ruido.
    # Lo que sí se evita es traer dos veces la MISMA prueba, anotando de dónde
    # salió, para que relanzar esto no duplique nada.
    ya = {t.user_attrs.get("origen")
          for t in estudio.get_trials(deepcopy=False)}

    traidas, cortadas = 0, 0
    for t in viejo.get_trials(deepcopy=False):
        if t.state != TrialState.COMPLETE or t.value is None:
            continue
        hechas = t.user_attrs.get("epocas_hechas")
        if hechas is not None and hechas < t.params.get("epocas", 0):
            cortadas += 1
            continue                  # nota de una red a medio entrenar
        if t.params.get("epocas", EPOCAS_MAX) != EPOCAS_MAX:
            cortadas += 1             # entrenada con otro presupuesto
            continue
        if any(t.params.get(k) not in ESPACIO[k] for k in ESPACIO):
            continue
        origen = f"{nombre_origen}#{t.number}"
        if origen in ya:
            continue
        ya.add(origen)
        estudio.add_trial(optuna.trial.create_trial(
            params={k: t.params[k] for k in ESPACIO}, distributions=dists,
            value=t.value, state=TrialState.COMPLETE,
            user_attrs={"origen": origen,
                        "epocas_hechas": t.user_attrs.get("epocas_hechas")}))
        traidas += 1
    if verbose:
        print(f"[optuna] del estudio '{nombre_origen}': {traidas} pruebas "
              f"válidas traídas, {cortadas} descartadas por estar a medio "
              f"entrenar", flush=True)
    return traidas


def reencolar_abandonadas(estudio, verbose=True):
    """Vuelve a poner en la cola las candidatas que se abandonaron a medias.

    Una abandonada costó sus épocas y no dejó ninguna nota: es esfuerzo tirado.
    Rematarlas sale barato comparado con probar una candidata nueva, porque de
    aquellas ya se sabe que apuntaban a algo (el muestreador las eligió), y
    encima cada una que se remata es una nota más con la que aprender.

    Se encolan, no se ejecutan aquí: Optuna reparte la cola entre los procesos
    que haya en marcha, y cada uno coge lo encolado ANTES de ponerse a proponer
    candidatas nuevas. Así se vacían en paralelo y sin duplicados."""
    from optuna.trial import TrialState

    ts = estudio.get_trials(deepcopy=False)
    # Lo que ya está hecho o esperando, para no encolar dos veces lo mismo.
    hechas = {tuple(sorted(t.params.items()))
              for t in ts if t.state in (TrialState.COMPLETE, TrialState.WAITING)}
    pendientes, encoladas = [], 0
    for t in ts:
        if t.state != TrialState.PRUNED:
            continue
        if any(t.params.get(k) not in ESPACIO[k] for k in ESPACIO):
            continue                  # usa un valor ya retirado del espacio
        clave = tuple(sorted(t.params.items()))
        if clave in hechas:
            continue
        hechas.add(clave)
        pendientes.append({k: t.params[k] for k in ESPACIO})
    for p in pendientes:
        estudio.enqueue_trial(p, skip_if_exists=True)
        encoladas += 1
    if verbose:
        print(f"[optuna] {encoladas} candidatas abandonadas vuelven a la cola; "
              f"se rematarán antes de proponer nada nuevo", flush=True)
    return encoladas


# --------------------------------------------------------------------------- #
# Una prueba
# --------------------------------------------------------------------------- #
class Evaluador:
    """Entrena y puntúa una configuración. Guarda en memoria el dataset y los
    escenarios para no releerlos en cada prueba."""

    def __init__(self, args, device):
        import torch
        self.args = args
        self.device = device
        self.datos = EN.Datos(args.muestras, device, args.fraccion_datos,
                              args.max_muestras, en_cpu=args.en_cpu)
        self.flotas, self.opts = esc.flotas(
            esc.cargar("seleccion", args.escenarios), limite=args.n_escenarios)
        self.mejor = None
        self.torch = torch

    def _peajes(self):
        """Épocas en las que se decide si una candidata sigue. Son las mismas
        que usa el podador de Optuna: 12, 24, 48… hasta el máximo."""
        peajes, e = [], self.args.min_epocas
        while e < EPOCAS_MAX:
            peajes.append(e)
            e *= max(2, self.args.reduccion)
        return set(peajes)

    def __call__(self, trial):
        import optuna
        torch = self.torch
        c = {k: trial.suggest_categorical(k, v) for k, v in ESPACIO.items()}
        c["epocas"] = EPOCAS_MAX          # ya no se sortea: lo fija el peaje
        t0 = time.perf_counter()
        peajes = self._peajes()
        media = escala = None

        def aviso(ep, mejor_val, red, estado):
            """Decide si esta candidata sigue, PUNTUÁNDOLA de verdad.

            En cada peaje se pone la red a conducir las 400 flotas y se informa
            de esa nota, que es la misma cifra con la que se la juzgará al
            final. Antes se informaba del error de validación, que solo es un
            indicio: se cortaba por sospechas. Se puede hacer así porque en
            este problema lo caro es entrenar (minutos) y medir es casi gratis
            (1-3 segundos), justo al revés de lo habitual.

            Y de paso, una candidata cortada deja de irse de vacío: su nota
            queda anotada aunque no llegue al final."""
            if not np.isfinite(mejor_val):
                trial.set_user_attr("fallo", "la red divergió")
                raise optuna.TrialPruned("divergió")
            if ep not in peajes or estado is None:
                return False
            # Puntuar carga en la red el mejor estado guardado, así que hay que
            # devolverle sus pesos vivos o el entrenamiento seguiría desde otro
            # sitio.
            vivos = {k: v.detach().clone() for k, v in red.state_dict().items()}
            nota = EN.nota_de(red, estado, media, escala, self.device,
                              self.flotas, self.opts, c)
            red.load_state_dict(vivos)
            red.train()
            trial.report(float(nota), ep)
            trial.set_user_attr(f"nota_epoca_{ep}", round(float(nota), 4))
            # Cortar tiene que ocurrir AQUÍ MISMO, con una excepción. Antes esto
            # devolvía True y fuera se volvía a preguntar `should_prune()`: como
            # esa segunda respuesta puede ser distinta (otros procesos van
            # terminando y la referencia se mueve), redes cortadas a las 5
            # épocas acababan puntuando COMO SI hubieran entrenado enteras.
            if trial.should_prune():
                raise optuna.TrialPruned()
            return False

        try:
            # OJO: los mismos ejes de representación que use la vista tienen
            # que ir luego al rollout de la nota (ver EN.nota_de). Si aquí falta
            # uno, la red se entrena con menos columnas de las que recibe al
            # evaluarla y revienta con un desajuste de tamaños.
            V, media, escala = self.datos.vista(c["n_vecinos"], c["horizonte"],
                                                c["h_pasado"],
                                                c.get("n_fourier", 0),
                                                c.get("n_rayos", 0))
            red, estado, val, eps = EN.entrenar_config(
                V, self.datos, c, self.device, criba=None,
                enfasis=self.args.enfasis, aviso=aviso)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            # No cabe con el tope de memoria de este proceso. No es un resultado
            # malo, es una prueba que no se ha hecho: se marca como fallida para
            # que Optuna no aprenda de ella una lección falsa.
            trial.set_user_attr("fallo", "no cabe en memoria")
            raise optuna.TrialPruned("no cabe en memoria")
        except optuna.TrialPruned:
            torch.cuda.empty_cache()      # abandonada a media carrera: se suelta
            raise

        nota = EN.nota_de(red, estado, media, escala, self.device,
                          self.flotas, self.opts, c)
        trial.set_user_attr("val_mse", round(float(val), 5))
        trial.set_user_attr("epocas_hechas", int(eps))
        trial.set_user_attr("dim_entrada", int(V.shape[1]))
        trial.set_user_attr("segundos", round(time.perf_counter() - t0, 1))

        if self.mejor is None or nota > self.mejor:
            self.mejor = nota
            EN.guardar_mejor(self.args.salida, self.args.idt,
                             (nota, val, dict(c), media.cpu().numpy(),
                              escala.cpu().numpy(),
                              {n: t.detach().cpu() for n, t in estado.items()},
                              V.shape[1]))
        del red, estado, V
        torch.cuda.empty_cache()
        return nota


# --------------------------------------------------------------------------- #
def construir_parser():
    ap = argparse.ArgumentParser(
        description="Búsqueda inteligente de hiperparámetros con Optuna.")
    ap.add_argument("--muestras", default=MUESTRAS_DIR)
    ap.add_argument("--escenarios", default=esc.CARPETA)
    ap.add_argument("--salida", default=MODELOS_DIR)
    ap.add_argument("--n-pruebas", dest="n_pruebas", type=int, default=3500,
                    help="pruebas que hace ESTE proceso (def. 3500)")
    ap.add_argument("--estudio", default=NOMBRE_ESTUDIO)
    ap.add_argument("--bd", default=None,
                    help="fichero SQLite del estudio (def. modelos/optuna.db)")
    ap.add_argument("--idt", type=int, default=0,
                    help="número de este proceso, solo para nombrar su .pt")
    ap.add_argument("--fraccion-datos", dest="fraccion_datos", type=float,
                    default=1.0)
    ap.add_argument("--max-muestras", dest="max_muestras", type=int,
                    default=12_000_000)
    ap.add_argument("--en-cpu", dest="en_cpu", action="store_true")
    ap.add_argument("--enfasis", type=float, default=1.0)
    ap.add_argument("--n-escenarios", dest="n_escenarios", type=int, default=400)
    ap.add_argument("--frac-vram", dest="frac_vram", type=float, default=1.0,
                    help="parte de la tarjeta que puede usar este proceso "
                         "(0.27 con tres a la vez)")
    ap.add_argument("--sin-podar", dest="sin_podar", action="store_true",
                    help="que toda candidata entrene sus épocas enteras. Por "
                         "defecto SÍ se cortan pronto las que van peor que las "
                         "demás: se exploran muchas más configuraciones por el "
                         "mismo tiempo, a cambio de que solo las que sobreviven "
                         "dejan nota. Cortar pronto NO es un fallo: la "
                         "candidata simplemente no llega a hacer todas sus "
                         "épocas")
    ap.add_argument("--min-epocas", dest="min_epocas", type=int, default=12,
                    help="épocas que se le regalan a toda configuración antes "
                         "de poder abandonarla (def. 12). Con 5 se abandonaba "
                         "al 96 %%, y con sesgo: las redes grandes arrancan "
                         "despacio y se las mataba por lentas, no por malas "
                         "(ancho medio de las abandonadas 1225 frente a 853 "
                         "de las que sobrevivían)")
    ap.add_argument("--reduccion", type=int, default=2,
                    help="de cada N que llegan a un hito, sobrevive 1 (def. 2).\n"
                         "Con 3 sobrevivía 1 de cada 9 y se abandonaba al 95 %%: "
                         "la cuenta es la correcta, pero aquí es demasiado. La "
                         "nota que guía la búsqueda solo se puede medir al final "
                         "(hay que conducir las flotas), así que abandonar tanto "
                         "deja al muestreador con una nota de verdad cada veinte "
                         "pruebas. Con 2 llega entera una de cada cuatro y sigue "
                         "ahorrando más de la mitad del tiempo")
    ap.add_argument("--sin-siembra", dest="sin_siembra", action="store_true",
                    help="no cargar los resultados de la fase aleatoria")
    ap.add_argument("--rematar", action="store_true",
                    help="poner en cola las candidatas que quedaron a medias "
                         "para terminarlas antes de proponer nada nuevo")
    ap.add_argument("--sembrar-de", dest="sembrar_de", default=None,
                    help="nombre de otro estudio del que traer sus pruebas "
                         "válidas (las que entrenaron todas sus épocas)")
    ap.add_argument("--resumen", action="store_true",
                    help="solo imprime cómo va el estudio y sale")
    ap.add_argument("--tf32", action="store_true", default=True)
    return ap


def imprimir_resumen(estudio):
    from optuna.trial import TrialState
    ts = estudio.get_trials(deepcopy=False)
    hechas = [t for t in ts if t.state == TrialState.COMPLETE]
    podadas = [t for t in ts if t.state == TrialState.PRUNED]
    print(f"[optuna] {len(ts)} pruebas · {len(hechas)} completas · "
          f"{len(podadas)} abandonadas pronto")
    if not hechas:
        return
    mejores = sorted(hechas, key=lambda t: -t.value)[:10]
    print(f"[optuna] mejor nota: {mejores[0].value:.4f}")
    print("\nLas 10 mejores:")
    for t in mejores:
        p = t.params
        print(f"  {t.value:.4f}  {p['n_capas']} capas x {p['oculto']:<5} "
              f"lote {p['lote']:<6} lr {p['lr']:<7} vec {p['n_vecinos']} "
              f"h {p['h_pasado']} hor {p['horizonte']} {p['activacion']:<5} "
              f"drop {p['dropout']} wd {p['weight_decay']} "
              f"{p['normalizacion']:<9} {p['mezcla']}")


def main():
    args = construir_parser().parse_args()
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    asegurar(args.salida)
    bd = args.bd or os.path.join(args.salida, "optuna.db")
    almacen = f"sqlite:///{bd}"

    # 'maximize' porque la nota de rollout es mejor cuanto más alta. El
    # muestreador arranca con unas pocas al azar solo si NO hay siembra; con las
    # 207 de la fase 1 dentro, ya propone con criterio desde la primera.
    estudio = optuna.create_study(
        study_name=args.estudio, storage=almacen, load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=(10 if args.sin_siembra else 0),
            multivariate=True, group=True, constant_liar=True),
        pruner=(optuna.pruners.NopPruner() if args.sin_podar
                else optuna.pruners.SuccessiveHalvingPruner(
                    min_resource=args.min_epocas,
                    reduction_factor=args.reduccion,
                    min_early_stopping_rate=0)))

    # La siembra la hace UN SOLO trabajador. Todos arrancan a la vez y todos
    # llamarían a sembrar(); su comprobación de "esta ya está" mira los trials
    # que hay en ese momento, así que con tres procesos leyendo el estudio vacío
    # al mismo tiempo, los tres añaden la fase 1 entera. Pasó: 484 pruebas para
    # 241 configuraciones, alguna repetida cinco veces. Y no es inofensivo: TPE
    # cuenta pruebas, así que una zona duplicada le parece el triple de
    # explorada de lo que está y concentra ahí lo que le queda.
    #
    # El cerrojo es un fichero creado con O_EXCL, que en Windows y en POSIX es
    # atómico: lo consigue exactamente uno. Los demás siguen sin sembrar, que es
    # justo lo que se quiere —la siembra ya la está haciendo otro—.
    if not args.sin_siembra and _tomar_cerrojo(args.salida):
        sembrar(estudio, args.salida)
    if args.sembrar_de:
        sembrar_de_estudio(estudio, args.sembrar_de, almacen)
    if args.rematar:
        reencolar_abandonadas(estudio)

    if args.resumen:
        imprimir_resumen(estudio)
        return

    import torch
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and 0 < args.frac_vram < 1.0:
        torch.cuda.set_per_process_memory_fraction(args.frac_vram)

    t0 = time.perf_counter()
    ev = Evaluador(args, device)
    n_veh = sum(len(f) for f in ev.flotas)
    print(f"[optuna] proceso {args.idt} · {device} · "
          f"{len(ev.datos.X):,} muestras "
          f"({len(ev.datos.idx_tr):,} train / {len(ev.datos.idx_va):,} val) · "
          f"{len(ev.flotas)} escenarios ({n_veh} vehículos) · "
          f"carga {time.perf_counter() - t0:.1f} s", flush=True)

    def traza(estudio, trial):
        from optuna.trial import TrialState
        hechas = sum(1 for t in estudio.get_trials(deepcopy=False)
                     if t.state == TrialState.COMPLETE)
        estado = ("abandonada" if trial.state == TrialState.PRUNED
                  else f"nota {trial.value:.4f}"
                  if trial.value is not None else str(trial.state))
        try:
            mejor = f"{estudio.best_value:.4f}"
        except ValueError:
            mejor = "—"
        print(f"[optuna] prueba {trial.number} · {estado} · "
              f"{hechas} completas · mejor {mejor}", flush=True)

    estudio.optimize(ev, n_trials=args.n_pruebas, callbacks=[traza],
                     gc_after_trial=True)
    imprimir_resumen(estudio)


if __name__ == "__main__":
    main()
