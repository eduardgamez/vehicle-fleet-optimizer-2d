import os
import math
import random
import json
import tempfile
import threading
import logging
from flask import Flask, request, jsonify, send_from_directory

import multi_v_evo as mve

app = Flask(__name__, static_folder="web")


# El navegador sondea el progreso cada ~0.7s con GET /api/*/estado, lo que
# llenaría la consola de líneas de acceso. Se filtran esas (y solo esas) del log
# de werkzeug, conservando el resto de peticiones.
class _FiltroSondeo(logging.Filter):
    def filter(self, record):
        return "/estado" not in record.getMessage()


logging.getLogger("werkzeug").addFilter(_FiltroSondeo())

env_actual = mve.Entorno()
env_actual.generar(0.0)
vehiculos_actuales = []
warmed_up = False

# La simulación muta estado GLOBAL (env_actual, vehiculos_actuales, veh.traj).
# Si llegan dos peticiones a la vez (p. ej. el navegador reintenta una que tarda
# mucho), en el servidor de desarrollo —multihilo— se ejecutarían en paralelo y
# corromperían ese estado (se veía cada vehículo planificado DOS veces). Este
# cerrojo serializa: si ya hay una simulación en curso, la segunda se rechaza
# con un mensaje claro en vez de pisar la primera.
_sim_lock = threading.Lock()

# Estado del trabajo de planificación en curso. El navegador ya NO espera de
# forma síncrona (lo cortaba a los ~60s): lanza el cálculo, este corre en un hilo
# de fondo, y el navegador consulta /api/<tarea>/estado cada segundo para mover la
# barra de progreso y recoger el resultado al terminar. Así se puede tardar lo que
# haga falta (como el escritorio) sin que el navegador dé "Load failed".
_job = {"estado": "idle", "progreso": "", "resultado": None, "error": None}
_job_lock = threading.Lock()


def _set_job(**kw):
    with _job_lock:
        _job.update(kw)


def _set_progreso(texto):
    with _job_lock:
        _job["progreso"] = texto


def _leer_job():
    with _job_lock:
        return dict(_job)


def _payload_simular(motivos):
    raw_frames = mve.construir_frames(vehiculos_actuales)
    frames_json = []
    for fr in raw_frames:
        fdict = {}
        for v in vehiculos_actuales:
            if v.idx in fr:
                pose = fr[v.idx]
                fdict[str(v.vid)] = [round(float(pose[0]), 3), round(float(pose[1]), 3), round(float(pose[2]), 4)]
        frames_json.append(fdict)
    salida = []
    for idx, v in enumerate(vehiculos_actuales):
        puntos = [[float(p[0]), float(p[1]), float(p[2])] for p in v.traj] if v.traj else []
        salida.append({"id": v.vid, "mision_ok": v.mision_ok, "traj": puntos})
    return {"ok": True, "trayectorias": salida, "frames": frames_json,
            "motivos": {str(k): str(v) for k, v in motivos.items()}}


def _worker_simular(calidad, modo_opt, max_cand):
    try:
        indices = list(range(len(vehiculos_actuales)))
        base = mve.Reservas()
        trays, motivos = _ejecutar_planificacion(
            env_actual, vehiculos_actuales, indices, base, calidad, modo_opt, max_cand,
            reubicable=True, progreso=_set_progreso)
        for idx, v in enumerate(vehiculos_actuales):
            traj_data = trays.get(idx)
            if traj_data is not None:
                v.traj = traj_data
                v.mision_ok = True
            else:
                v.traj = []
                v.mision_ok = False
        _set_job(estado="done", progreso="Completado",
                 resultado=_payload_simular(motivos), error=None)
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        _set_job(estado="error", error=f"{type(e).__name__}: {e}")
    finally:
        _sim_lock.release()


@app.route("/api/simular/estado", methods=["GET"])
def planificacion_estado():
    return jsonify(_leer_job())

def asegurar_warmup():
    global warmed_up
    if not warmed_up:
        print("Realizando warmup de núcleos C/Numba...")
        mve.warmup()
        warmed_up = True

def _muestrear(env, length, width, otros, sep):
    for _ in range(4000):
        x = random.uniform(length, mve.W - length)
        y = random.uniform(length, mve.H - length)
        th = random.uniform(0, 2 * math.pi)
        if not env.libre(x, y, th, length, width, margen=0.3):
            continue
        if all(math.hypot(x - ox, y - oy) >= sep for ox, oy in otros):
            return (x, y, th)
    return None

def _orientaciones_libres_meta(env, mx, my, length, width, n=48):
    """Barrido denso de orientaciones en la meta: devuelve la lista de ángulos
    (rad) en los que el vehículo cabe. Sustituye al muestreo de 25 ángulos al
    azar, que en plazas con una banda de orientaciones estrecha fallaba a
    menudo y acababa marcando la llegada como "libre" justo donde MENOS margen
    hay."""
    libres = []
    for i in range(n):
        c = 2 * math.pi * i / n
        if env.libre(mx, my, c, length, width, margen=0.25):
            libres.append(c)
    return libres

# Una meta se considera "holgada" (apta para llegada libre) si admite al menos
# esta fracción de orientaciones del barrido. Por debajo de eso, la plaza es
# demasiado ajustada para dejar el ángulo al azar del planificador: se le fija
# un ángulo de llegada concreto y factible para que la ruta sea fiable.
FRACCION_HOLGADA = 0.25
PROB_LLEGADA_LIBRE = 0.4

def _elegir_llegada(env, mx, my, length, width, th_fallback, n=48):
    """Decide el ángulo de llegada de una meta:
      · Si la plaza es holgada (muchas orientaciones caben), con probabilidad
        PROB_LLEGADA_LIBRE se deja LIBRE (None) — ahí el planificador siempre
        puede cuadrar una orientación.
      · Si es ajustada, se fija un ángulo concreto y factible (nunca "libre"),
        que es justo el caso que antes fallaba.
    'th_fallback' es la orientación con la que se muestreó la meta (garantizada
    factible), usada si el barrido no encontrara ninguna por discretización."""
    libres = _orientaciones_libres_meta(env, mx, my, length, width, n)
    if not libres:
        return th_fallback
    holgada = len(libres) >= max(1, int(FRACCION_HOLGADA * n))
    if holgada and random.random() < PROB_LLEGADA_LIBRE:
        return None
    return random.choice(libres)

def generar_especs_aleatorias(env, n):
    especs = []
    # OJO: se comprueba contra TODOS los puntos ya colocados (inicios Y metas
    # de TODOS los vehículos), no solo contra los de su misma categoría. La
    # meta de cualquier vehículo se bloquea como obstáculo permanente para
    # los demás (bloqueos_metas), así que si el inicio de uno cae encima de
    # la meta de otro, ese vehículo nace "atrapado" y no puede dar ni un
    # paso: el planificador falla al instante (motivo="sin_ruta") aunque el
    # mapa esté prácticamente vacío.
    puntos = []
    for i in range(n):
        largo = random.uniform(0.7, 1.2)
        ancho = largo * random.uniform(0.50, 0.62)
        sep = largo * 1.6
        ini = _muestrear(env, largo, ancho, puntos, sep)
        if ini is None:
            return None
        ix, iy, ith = ini
        puntos.append((ix, iy))
        met = _muestrear(env, largo, ancho, puntos, sep)
        if met is None:
            return None
        mx, my, mth = met
        puntos.append((mx, my))
        thm = _elegir_llegada(env, mx, my, largo, ancho, mth)
        th0 = ith
        especs.append({
            "id": i + 1,
            "inicio": (round(ix, 2), round(iy, 2)),
            "giro_inicial": round(math.degrees(th0), 1),
            "meta": (round(mx, 2), round(my, 2)),
            "angulo_llegada": (round(math.degrees(thm), 1)
                               if thm is not None else "libre"),
            "largo": round(largo, 2),
            "ancho": round(ancho, 2),
            "v_max": round(random.uniform(1.8, 3.2), 2),
            "a_max": round(random.uniform(0.8, 1.4), 2),
            "giro_max": round(random.uniform(28.0, 40.0), 1),
            "grupo": random.randint(1, 2),
            "prioridad": random.randint(1, 3),
        })
    # Numeración por PRIORIDAD: se ordena por (grupo, prioridad) ascendente y se
    # reasignan los ids 1..n en ese orden, de modo que un vehículo con más
    # prioridad (grupo/prioridad menor) siempre lleve un número más pequeño que
    # otro con menos prioridad. Entre vehículos de igual grupo y prioridad
    # (p. ej. lo que optimiza el modo global) el orden numérico es indiferente,
    # así que se deja el de generación (sorted es estable).
    especs.sort(key=lambda e: (e["grupo"], e["prioridad"]))
    for i, e in enumerate(especs):
        e["id"] = i + 1
    return especs

def _veh_a_spec(v):
    return {
        "id": v.vid,
        "inicio": (round(v.inicio[0], 2), round(v.inicio[1], 2)),
        "giro_inicial": round(math.degrees(v.inicio[2]), 1),
        "meta": (round(v.meta[0], 2), round(v.meta[1], 2)),
        "angulo_llegada": round(math.degrees(v.meta_th), 1) if v.meta_th is not None else "libre",
        "largo": round(v.length, 2),
        "ancho": round(v.width, 2),
        "v_max": round(v.v_max, 2),
        "a_max": round(v.a_max, 2),
        # delta_max se guarda en RADIANES; el texto de especificaciones va en
        # GRADOS. Sin la conversión, el texto exportado decía "giro_max: 0.6"
        # y al re-parsearlo fallaba la validación (mínimo 5 grados).
        "giro_max": round(math.degrees(v.delta_max), 1),
        "grupo": getattr(v, "grupo", 1),
        "prioridad": getattr(v, "prioridad", 1),
    }

def obtener_estado_json(texto_salida=None):
    obs_json = [[[float(pt[0]), float(pt[1])] for pt in pol] for pol in env_actual.obstaculos]
    vehs_json = []
    for idx, v in enumerate(vehiculos_actuales):
        vehs_json.append({
            "id": v.vid,
            "inicio": [float(v.inicio[0]), float(v.inicio[1]), float(v.inicio[2])],
            "meta": [float(v.meta[0]), float(v.meta[1])],
            "meta_th": float(v.meta_th) if v.meta_th is not None else None,
            "largo": float(v.length),
            "ancho": float(v.width),
            "grupo": int(getattr(v, "grupo", 1)),
            "prioridad": int(getattr(v, "prioridad", 1)),
            "color": mve.PALETA[idx % len(mve.PALETA)]
        })
    if texto_salida is None:
        texto_salida = mve.especs_a_texto([_veh_a_spec(v) for v in vehiculos_actuales]) if vehiculos_actuales else mve.TEXTO_EJEMPLO
    return {
        "ok": True,
        "vehiculos": vehs_json,
        "texto": texto_salida,
        "obstaculos": obs_json,
        "mundo": {"W": mve.W, "H": mve.H}
    }

@app.errorhandler(Exception)
def _errores_como_json(e):
    """Devuelve SIEMPRE JSON en las rutas /api, incluso ante errores no
    controlados. Así el frontend puede mostrar el motivo real en vez de
    fallar al parsear la página HTML de error de Flask (que en Safari da
    'The string did not match the expected pattern')."""
    from werkzeug.exceptions import HTTPException
    if request.path.startswith("/api/"):
        codigo = e.code if isinstance(e, HTTPException) else 500
        return jsonify({"ok": False, "error": str(e)}), codigo
    if isinstance(e, HTTPException):
        return e
    return "Error interno del servidor", 500

@app.route("/")
def index():
    return send_from_directory("web", "index.html")

@app.route("/<path:path>")
def static_files(path):
    return send_from_directory("web", path)

@app.route("/api/init", methods=["GET"])
def init_state():
    global env_actual
    if not env_actual.obstaculos:
        env_actual.generar(0.0)
    return jsonify(obtener_estado_json())

@app.route("/api/generar", methods=["POST"])
def generar():
    global env_actual, vehiculos_actuales
    data = request.json or {}
    modo = data.get("modo", "aleatorio")
    num_vehiculos = int(data.get("num_vehiculos", 4))
    texto_entrada = data.get("texto", "")

    if not env_actual.obstaculos:
        env_actual.generar(0.0)

    if modo in ("manual", "personalizado") and texto_entrada.strip():
        try:
            especs_norm = mve.parsear_especificaciones(texto_entrada)
            vehiculos_actuales = [
                mve.espec_a_vehiculo(e, i, env_actual)
                for i, e in enumerate(especs_norm)
            ]
            texto_salida = texto_entrada
        except ValueError as e:
            return jsonify({"ok": False, "error": str(e)}), 400
    else:
        especs = generar_especs_aleatorias(env_actual, num_vehiculos)
        if especs is None:
            return jsonify({"ok": False, "error": "No se encontraron posiciones libres. Reduce vehículos/obstáculos."}), 400
        texto_salida = mve.especs_a_texto(especs)
        try:
            especs_norm = mve.parsear_especificaciones(texto_salida)
            vehiculos_actuales = [
                mve.espec_a_vehiculo(e, i, env_actual)
                for i, e in enumerate(especs_norm)
            ]
        except ValueError as e:
            return jsonify({"ok": False, "error": str(e)}), 400

    return jsonify(obtener_estado_json(texto_salida))

@app.route("/api/mapa/aleatorio", methods=["POST"])
def mapa_aleatorio():
    global env_actual, vehiculos_actuales
    data = request.json or {}
    densidad = float(data.get("densidad", 0.0))
    env_actual.generar(densidad=densidad)
    vehiculos_actuales = []
    return jsonify(obtener_estado_json(mve.TEXTO_EJEMPLO))

@app.route("/api/mapa/importar", methods=["POST"])
def mapa_importar():
    global env_actual, vehiculos_actuales
    if "archivo" not in request.files:
        return jsonify({"ok": False, "error": "No se envió archivo"}), 400
    archivo = request.files["archivo"]
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(archivo.filename)[1]) as tmp:
        archivo.save(tmp.name)
        tmp_path = tmp.name
    try:
        polys = mve.imagen_a_poligonos(tmp_path)
        nombre = os.path.splitext(archivo.filename)[0]
        env_actual.desde_poligonos(polys, nombre)
        vehiculos_actuales = []
    except Exception as e:
        return jsonify({"ok": False, "error": f"No se pudo convertir la imagen: {e}"}), 400
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    return jsonify(obtener_estado_json(mve.TEXTO_EJEMPLO))

@app.route("/api/mapa/guardar", methods=["GET"])
def mapa_guardar():
    global env_actual
    data = {
        "origen": getattr(env_actual, "origen", "exportado"),
        "rects": getattr(env_actual, "rects", None)
    }
    if data["rects"] is None:
        data["obstaculos"] = [[[float(pt[0]), float(pt[1])] for pt in pol] for pol in env_actual.obstaculos]
    return jsonify(data)

@app.route("/api/mapa/cargar", methods=["POST"])
def mapa_cargar():
    global env_actual, vehiculos_actuales
    if "archivo" not in request.files:
        return jsonify({"ok": False, "error": "No se envió archivo"}), 400
    try:
        data = json.load(request.files["archivo"])
        if "rects" in data and data["rects"] is not None:
            env_actual.desde_rects(data["rects"], data.get("origen", "cargado"))
        elif "obstaculos" in data:
            # _empaquetar reconstruye también los arrays NumPy de colisión
            # (np_c/np_ax/np_bb); asignar solo .obstaculos dejaba al
            # planificador usando los OBB del mapa ANTERIOR.
            polys = [[(float(pt[0]), float(pt[1])) for pt in pol]
                     for pol in data["obstaculos"]]
            env_actual._empaquetar(polys)
            env_actual.origen = data.get("origen", "cargado")
            env_actual.rects = None
        vehiculos_actuales = []
    except Exception as e:
        return jsonify({"ok": False, "error": f"Error al cargar mapa JSON: {e}"}), 400
    return jsonify(obtener_estado_json(mve.TEXTO_EJEMPLO))

def _reubicar(env, veh, vehiculos):
    # Igual que en generar_especs_aleatorias: el nuevo inicio y la nueva meta
    # deben separarse de TODOS los puntos ajenos (inicios y metas), no solo
    # de su propia categoría, o el vehículo puede reaparecer atrapado contra
    # la meta bloqueada de otro.
    sep = veh.length * 1.6
    otros = []
    for v in vehiculos:
        if v.idx == veh.idx:
            continue
        otros.append(v.inicio[:2])
        otros.append(v.meta)
    ini = _muestrear(env, veh.length, veh.width, otros, sep)
    if ini is None:
        return False
    met = _muestrear(env, veh.length, veh.width, otros + [ini[:2]], sep)
    if met is None:
        return False
    ix, iy, ith = ini
    thm = _elegir_llegada(env, met[0], met[1], veh.length, veh.width, met[2])
    veh.inicio = (ix, iy, ith)
    veh.meta = (met[0], met[1])
    veh.meta_th = thm
    return True

#  Si se ejecuta en Render (donde se define la variable de entorno CAP_NODOS), se
#  limita la memoria para no agotar los 512 MB. En local (sin CAP_NODOS), se deja
#  en None para usar el límite nativo de la calidad elegida.
CAP_NODOS = os.environ.get("CAP_NODOS")
if CAP_NODOS is not None:
    CAP_NODOS = int(CAP_NODOS)


def _ruta_real(veh, traj):
    """¿'traj' es una ruta que de verdad LLEVA el vehículo a su meta, y no un
    apaño estático [p, p]?  El planificador, cuando no consigue avanzar, puede
    devolver una trayectoria de un solo punto que 'planificar_orden' envuelve
    como [p, p]: no es None, así que el conteo de fallos la daba por buena y el
    vehículo se quedaba CONGELADO en la animación aunque tuviera camino libre.
    Una ruta estática solo es legítima si el vehículo ya arranca en su meta."""
    if traj is None or len(traj) < 2:
        return False
    x0, y0 = traj[0][0], traj[0][1]
    xf, yf = traj[-1][0], traj[-1][1]
    if math.hypot(xf - x0, yf - y0) > 0.1:
        return True
    # No se ha movido: solo vale si ya nació esencialmente sobre la meta.
    return math.hypot(x0 - veh.meta[0], y0 - veh.meta[1]) <= 1.6


# Presupuesto de TIEMPO por vehículo al COMPARAR órdenes candidatos (fase de
# exploración de "global"/"grupos"). El orden GANADOR se replanifica luego SIN
# este límite (solo el tope de nodos): la exploración sirve para ELEGIR qué
# orden comprometer, no para producir la ruta definitiva. Igual que el
# escritorio (mve.PLAZO_VEH_CAND).
PLAZO_VEH_CAND = 5.0


def _planificar_una_pasada(env, vehiculos, orden, base, pl,
                           reubicable, progreso, etiqueta="", deadline_dur=None):
    """Una pasada cooperativa sobre 'orden' (índices): planifica vehículo a
    vehículo y RESERVA cada uno como obstáculo (móvil si logra ruta, estático si
    no) antes del siguiente. Devuelve (trays, motivos, fallos, coste), donde
    'coste' es el tiempo total de flota (suma de duraciones) más una penalización
    por cada vehículo sin ruta, para poder comparar órdenes entre sí.

    'deadline_dur' (s/vehículo), si se indica, acota el tiempo de cada búsqueda
    —se usa al comparar candidatos—; con None solo manda el tope de nodos.
    'reubicable': en modo aleatorio, a un vehículo sin salida se le buscan
    extremos nuevos (solo en la pasada DEFINITIVA, no al comparar)."""
    import time as _t
    reservas = base.copia()
    trays = {}
    motivos = {}
    coste = 0.0
    pref = (etiqueta + " · ") if etiqueta else ""
    for pos, i in enumerate(orden):
        veh = vehiculos[i]
        if CAP_NODOS is not None:
            pl.max_nodos = CAP_NODOS                  # presupuesto de nodos (memoria)
        pl.max_exp = 20_000_000                   # las expansiones no son el freno
        pl.deadline = (_t.perf_counter() + deadline_dur) if deadline_dur else None
        # pose0: un bloqueo de meta ajena que SOLAPE el arranque del vehículo
        # se omite; si no, nace atrapado y da "sin_ruta" al instante.
        pl.bloqueos = mve.bloqueos_metas(
            vehiculos, orden, excepto=i, pose0=veh.inicio)
        if progreso is not None:
            def _tick(expand, _p=pos, _vid=veh.vid, _n=len(orden), _pf=pref):
                progreso(f"{_pf}Planificando vehículo {_p + 1}/{_n} (id {_vid}) · "
                         f"{expand:,} nodos explorados")
            pl.tick = _tick
        else:
            pl.tick = None

        _t0 = _t.perf_counter()
        traj = pl.planificar(veh, reservas)
        _dt = _t.perf_counter() - _t0
        motivos[i] = pl.motivo
        ok = _ruta_real(veh, traj)
        # REUBICACIÓN (modo aleatorio, como el escritorio): si el vehículo no
        # tiene salida donde nació, se le buscan extremos nuevos.
        intentos = 0
        while (not ok and reubicable
               and pl.motivo == "sin_ruta" and intentos < 12):
            if not _reubicar(env, veh, vehiculos):
                break
            pl.deadline = (_t.perf_counter() + deadline_dur) if deadline_dur else None
            pl.bloqueos = mve.bloqueos_metas(vehiculos, orden, excepto=i,
                                             pose0=veh.inicio)
            traj = pl.planificar(veh, reservas)
            motivos[i] = pl.motivo
            ok = _ruta_real(veh, traj)
            intentos += 1
        # DIAGNÓSTICO (consola local / logs de Render): por qué cada vehículo
        # logró o no su ruta, con nodos creados/expandidos y tiempo:
        #   sin_ruta   → encajonado: algo (otro vehículo/obstáculo) lo bloquea
        #   techo_nodos→ llegó al presupuesto CAP_NODOS (subirlo o revisar mapa)
        #   limite     → agotó expansiones
        print(f"[PLAN] {pref}veh id={veh.vid} pos={pos+1}/{len(orden)} "
              f"{'OK' if ok else 'SIN_RUTA'} motivo={pl.motivo} "
              f"nodos_creados={pl.nodos} expandidos={pl.expandidos} "
              f"ang_llegada={'no' if veh.meta_th is None else 'si'} "
              f"ang_relajado={'si' if getattr(pl, 'relajado', False) else 'no'} "
              f"t={_dt:.1f}s", flush=True)
        if ok:
            trays[i] = traj
            reservas.add(traj, veh.length, veh.width)
            coste += (len(traj) - 1) * mve.DT
        else:
            # No llegó: queda APARCADO en su salida y sigue contando como
            # obstáculo (reserva estática) para los vehículos que van detrás.
            trays[i] = None
            px, py, pth = veh.inicio
            reservas.add([(px, py, pth)], veh.length, veh.width)
            coste += mve.PENAL_FALLO
    pl.tick = None
    fallos = sum(1 for i in orden if trays[i] is None)
    return trays, motivos, fallos, coste


def _ejecutar_planificacion(env, vehiculos, indices, base, calidad, modo_opt, max_cand,
                            reubicable=False, progreso=None):
    """Planificación cooperativa de la flota (igual que el escritorio).

    Genera los ÓRDENES candidatos según el modo ("secuencial" da uno solo;
    "global" y "grupos" dan varios, hasta max_cand) y, si hay más de uno, los
    EXPLORA: cada orden se evalúa con una pasada cooperativa completa y gana el
    de MENOS fallos y, a igualdad, MENOR tiempo total de flota. El ganador se
    replanifica una última vez sin límite de tiempo (solo nodos) para producir
    la ruta definitiva. Cada vehículo se RESERVA como obstáculo (móvil si logra
    ruta, estático si no) antes del siguiente; si no llega, bloquea a los de
    detrás (en modo aleatorio se intenta reubicarlo con extremos nuevos).

    'progreso' (opcional): callback(texto) para informar del avance en vivo
    (orden en curso, vehículo actual y nodos explorados), usado por la ejecución
    en segundo plano para mover la barra de progreso del navegador."""
    pl = mve.Planificador(env)
    pl.configurar_calidad(calidad)

    ordenes = mve.generar_ordenes(vehiculos, indices, modo_opt, max_cand)

    # Un solo orden (secuencial, o modos sin nada que permutar): una sola pasada,
    # sin límite de tiempo (solo el tope de nodos), como hasta ahora.
    if len(ordenes) <= 1:
        trays, motivos, _, _ = _planificar_una_pasada(
            env, vehiculos, list(ordenes[0]), base, pl,
            reubicable, progreso)
        return trays, motivos

    # EXPLORACIÓN: cada orden candidato se evalúa con una pasada cooperativa SIN
    # reubicación (determinista y comparable). Igual que el escritorio, con varios
    # órdenes y varios núcleos se evalúan EN PARALELO por HORNADAS (parada al 95 %
    # de llegadas acumuladas); con un solo núcleo, o si el paralelo falla, se cae
    # al bucle secuencial de siempre.
    total = len(ordenes)
    mejor = None                             # (fallos, coste, orden)
    cal_int = int(round(float(calidad)))
    if mve.nucleos_disponibles() >= 2 and len(indices) >= 3:
        try:
            if progreso is not None:
                progreso(f"Explorando {total} órdenes en paralelo (hornadas)…")
            res = mve.evaluar_ordenes_hornadas(
                env, vehiculos, list(ordenes), base, cal_int,
                max_exp=None, progreso=progreso, sigue_vivo=None)
            if res is not None:
                mejor = (res[0], res[1], list(res[4]))
                print(f"[OPT] paralelo fallos={res[0]} coste={res[1]:.2f} "
                      f"orden={[vehiculos[i].vid for i in res[4]]}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[OPT] paralelo falló ({type(e).__name__}: {e}); "
                  "se usa el bucle secuencial", flush=True)
            mejor = None

    if mejor is None:                        # secuencial (1 núcleo o retroceso)
        for oi, orden in enumerate(ordenes):
            etiqueta = f"Orden {oi + 1}/{total}"
            if progreso is not None:
                progreso(f"Explorando {etiqueta}…")
            _, _, fallos, coste = _planificar_una_pasada(
                env, vehiculos, list(orden), base, pl,
                reubicable=False, progreso=progreso, etiqueta=etiqueta,
                deadline_dur=PLAZO_VEH_CAND)
            if mejor is None or (fallos, coste) < (mejor[0], mejor[1]):
                mejor = (fallos, coste, list(orden))
            print(f"[OPT] {etiqueta} fallos={fallos} coste={coste:.2f}", flush=True)

    # El orden GANADOR se replanifica SIN límite de tiempo (solo nodos) y CON
    # reubicación, para producir la ruta definitiva.
    if progreso is not None:
        progreso("Planificando el mejor orden encontrado…")
    trays, motivos, _, _ = _planificar_una_pasada(
        env, vehiculos, mejor[2], base, pl,
        reubicable, progreso, etiqueta="Mejor orden")
    return trays, motivos

@app.route("/api/simular", methods=["POST"])
def simular():
    global env_actual, vehiculos_actuales
    asegurar_warmup()

    data = request.json or {}
    calidad = int(data.get("calidad", 3))
    modo_opt = data.get("optimizacion", "secuencial")
    max_cand = int(data.get("max_cand", 12))

    if not vehiculos_actuales:
        return jsonify({"ok": False, "error": "Primero genera o inserta los vehículos."}), 400

    if not _sim_lock.acquire(blocking=False):
        return jsonify({"ok": False, "estado": "ocupado",
                        "error": "Ya hay una simulación en curso."}), 409
    # Se lanza el cálculo en segundo plano y se responde AL INSTANTE. El navegador
    # seguirá el progreso con GET /api/simular/estado. El hilo libera el cerrojo
    # al terminar (en su 'finally').
    _set_job(estado="running", progreso="Iniciando…", resultado=None, error=None)
    threading.Thread(target=_worker_simular,
                     args=(calidad, modo_opt, max_cand), daemon=True).start()
    return jsonify({"ok": True, "estado": "running"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    print(f"Servidor listo en http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=debug)

