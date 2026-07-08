import os
import math
import random
import json
import tempfile
from flask import Flask, request, jsonify, send_from_directory

import multi_v_evo as mve

app = Flask(__name__, static_folder="web")

env_actual = mve.Entorno()
env_actual.generar(0.0)
vehiculos_actuales = []
warmed_up = False

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
        largo = random.uniform(1.0, 1.8)
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
        "giro_max": round(v.delta_max, 1),
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
        rects = mve.imagen_a_rects(tmp_path)
        nombre = os.path.splitext(archivo.filename)[0]
        env_actual.desde_rects(rects, nombre)
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
            env_actual.obstaculos = data["obstaculos"]
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

#  Presupuesto de tiempo POR VEHÍCULO (segundos). El núcleo de búsqueda no
#  vectoriza el chequeo de colisión/cinemática, así que su ritmo real es de
#  unos pocos miles de expansiones por segundo (no cientos de miles); con un
#  plazo compartido entre varios vehículos, los primeros de la cola podían
#  agotarlo entero y dejar a los últimos sin tiempo para buscar siquiera un
#  primer nodo válido, aunque su ruta fuera perfectamente posible.
PRESUPUESTO_VEHICULO_S = 20.0

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


def _ejecutar_planificacion(env, vehiculos, indices, base, calidad, modo_opt, max_cand, reubicable=False, inicios=None):
    import time
    n = len(indices)
    ordenes = mve.generar_ordenes(vehiculos, indices, modo_opt, max_cand)
    pl = mve.Planificador(env)
    pl.configurar_calidad(calidad)
    cap_prev = pl.max_exp

    mejor = None
    for oi, orden in enumerate(ordenes):
        pl.max_exp = cap_prev
        trays, motivos, coste = mve.planificar_orden(
            pl, vehiculos, orden, base, inicios=inicios,
            deadline_dur=PRESUPUESTO_VEHICULO_S
        )
        # Un vehículo con ruta estática (no se mueve pese a no estar en su meta)
        # cuenta como FALLO: así no se acepta un orden que deja coches congelados
        # y esos vehículos pasan por el rescate en vez de mostrarse inmóviles.
        for i in orden:
            if trays[i] is not None and not _ruta_real(vehiculos[i], trays[i]):
                trays[i] = None
        fallos = sum(1 for i in orden if trays[i] is None)
        if mejor is None or (fallos, coste) < (mejor[0], mejor[1]):
            mejor = (fallos, coste, trays, motivos, orden)
        if fallos == 0:
            break

    fallos, _, trays, motivos, orden = mejor

    if fallos > 0:
        reservas = base.copia()
        for i in orden:
            if trays[i] is not None:
                v = vehiculos[i]
                reservas.add(trays[i], v.length, v.width)
        for i in orden:
            if trays[i] is not None:
                continue
            veh = vehiculos[i]
            pl.max_exp = max(cap_prev, 3000000)
            pl.deadline = time.perf_counter() + PRESUPUESTO_VEHICULO_S
            pl.bloqueos = mve.bloqueos_metas(vehiculos, orden, excepto=i)
            ini = inicios.get(i) if inicios else None
            traj = pl.planificar(veh, reservas, inicio=ini)
            motivos[i] = pl.motivo

            # "sin_ruta" == el punto de partida/plaza está realmente atrapado
            # (típicamente porque cae sobre la meta bloqueada de otro
            # vehículo): reubicar sirve. "limite" == la búsqueda se queda sin
            # tiempo/nodos, algo mucho más probable cuando el ÁNGULO DE
            # LLEGADA exigido obliga a una maniobra de varios tramos que el
            # conector reversible no cubre en un solo intento; ahí reubicar
            # una y otra vez (cada intento con su propio presupuesto) solo
            # desperdicia tiempo, porque el problema no es la posición.
            # Presupuesto reducido para los reintentos de reubicación: si la
            # nueva posición vuelve a estar atrapada, conviene averiguarlo
            # rápido y probar otra, en vez de agotar el presupuesto completo
            # en cada intento (antes: hasta 6 × 20 s = 2 min solo en esto).
            intentos = 0
            while (pl.motivo == "sin_ruta" and reubicable and inicios is None and intentos < 3):
                if not _reubicar(env, veh, vehiculos):
                    break
                pl.deadline = time.perf_counter() + 8.0
                pl.bloqueos = mve.bloqueos_metas(vehiculos, orden, excepto=i)
                traj = pl.planificar(veh, reservas)
                motivos[i] = pl.motivo
                intentos += 1

            if (traj is None or len(traj) < 2) and veh.meta_th is not None:
                # Último recurso: antes de dejar el vehículo sin ruta
                # ("aparcado"), se reintenta UNA vez con el ángulo de llegada
                # libre. Es preferible llegar sin la orientación exacta que
                # no llegar, y evita agotar el presupuesto reubicando cuando
                # el problema real es el ángulo, no la posición.
                veh.meta_th = None
                pl.deadline = time.perf_counter() + PRESUPUESTO_VEHICULO_S
                pl.bloqueos = mve.bloqueos_metas(vehiculos, orden, excepto=i)
                traj = pl.planificar(veh, reservas, inicio=ini)
                motivos[i] = pl.motivo

            if _ruta_real(veh, traj):
                trays[i] = traj
                motivos[i] = "ok"
                reservas.add(traj, veh.length, veh.width)
            elif traj:
                # No consiguió avanzar: se deja aparcado como obstáculo estático,
                # pero NO se marca "ok" (no ha hecho su ruta).
                trays[i] = [traj[0], traj[0]]
                reservas.add(trays[i], veh.length, veh.width)

    pl.deadline = None
    pl.max_exp = cap_prev
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

    indices = list(range(len(vehiculos_actuales)))
    base = mve.Reservas()

    trays, motivos = _ejecutar_planificacion(
        env_actual, vehiculos_actuales, indices, base, calidad, modo_opt, max_cand,
        reubicable=True, inicios=None
    )

    for idx, v in enumerate(vehiculos_actuales):
        traj_data = trays.get(idx)
        if traj_data is not None:
            v.traj = traj_data
            v.mision_ok = True
        else:
            v.traj = []
            v.mision_ok = False

    raw_frames = mve.construir_frames(vehiculos_actuales)
    frames_json = []
    for fr in raw_frames:
        fdict = {}
        for v in vehiculos_actuales:
            if v.idx in fr:
                pose = fr[v.idx]
                fdict[str(v.vid)] = [round(float(pose[0]), 3), round(float(pose[1]), 3), round(float(pose[2]), 4)]
        frames_json.append(fdict)

    salida_trayectorias = []
    for idx, v in enumerate(vehiculos_actuales):
        traj_data = v.traj
        puntos = [[float(pt[0]), float(pt[1]), float(pt[2])] for pt in traj_data] if traj_data else []
        salida_trayectorias.append({
            "id": v.vid,
            "mision_ok": v.mision_ok,
            "traj": puntos
        })

    return jsonify({
        "ok": True,
        "trayectorias": salida_trayectorias,
        "frames": frames_json,
        "motivos": {str(k): str(v) for k, v in motivos.items()}
    })

@app.route("/api/nuevas_rutas", methods=["POST"])
def nuevas_rutas():
    global env_actual, vehiculos_actuales
    asegurar_warmup()
    data = request.json or {}
    texto = data.get("texto", "")
    calidad = int(data.get("calidad", 3))
    modo_opt = data.get("optimizacion", "secuencial")
    max_cand = int(data.get("max_cand", 12))

    if not vehiculos_actuales:
        return jsonify({"ok": False, "error": "Primero genera y simula vehículos."}), 400

    try:
        especs = mve.parsear_especificaciones(texto)
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400

    por_id = {v.vid: v for v in vehiculos_actuales}
    lote = []
    try:
        for e in especs:
            if e["id"] not in por_id:
                raise ValueError(f"No existe ningún vehículo con id {e['id']!r}.")
            if "inicio" in e or "giro_inicial" in e:
                raise ValueError(f"Vehículo {e['id']!r}: en una misión nueva no se cambia 'inicio'/'giro_inicial'.")
            if "largo" in e or "ancho" in e:
                raise ValueError(f"Vehículo {e['id']!r}: el tamaño del vehículo no cambia en una misión nueva.")
            if "meta" not in e:
                raise ValueError(f"Vehículo {e['id']!r}: la misión nueva necesita 'meta'.")
            veh = por_id[e["id"]]
            mx, my = e["meta"]
            px, py, pth = veh.pose_final
            if "angulo_llegada" in e:
                thm = e["angulo_llegada"]
            else:
                thm = math.atan2(my - py, mx - px)
            th_plaza = thm if thm is not None else 0.0
            if not env_actual.libre(mx, my, th_plaza, veh.length, veh.width, margen=0.15):
                raise ValueError(f"Vehículo {e['id']!r}: la plaza nueva ({mx:g}, {my:g}) no cabe.")
            lote.append((veh, e, thm))
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400

    k_inicio = max(len(v.traj) if v.traj else 1 for v in vehiculos_actuales)
    ids_lote = {veh.vid for veh, _, _ in lote}
    base = mve.Reservas()
    for v in vehiculos_actuales:
        if v.vid not in ids_lote:
            base.add([v.pose_final], v.length, v.width)

    indices = []
    inicios = {}
    for veh, e, thm in lote:
        veh.meta = tuple(e["meta"])
        veh.meta_th = thm
        if "grupo" in e:
            veh.grupo = e["grupo"]
        if "prioridad" in e:
            veh.prioridad = e["prioridad"]
        if "v_max" in e:
            veh.v_max = e["v_max"]
        if "a_max" in e:
            veh.a_max = e["a_max"]
        if "giro_max" in e:
            veh.delta_max = e["giro_max"]
        indices.append(veh.idx)
        inicios[veh.idx] = veh.pose_final

    trays, motivos = _ejecutar_planificacion(
        env_actual, vehiculos_actuales, indices, base, calidad, modo_opt, max_cand,
        reubicable=False, inicios=inicios
    )

    for i in indices:
        veh = vehiculos_actuales[i]
        traj = trays.get(i)
        if traj is None:
            veh.mision_ok = False
        else:
            veh.mision_ok = True
            base_traj = veh.traj if veh.traj else [veh.inicio]
            ultima = base_traj[-1]
            relleno = [ultima] * max(0, k_inicio - len(base_traj))
            veh.traj = base_traj + relleno + traj[1:]
            veh.dt_plan = mve.DT

    raw_frames = mve.construir_frames(vehiculos_actuales)
    frames_json = []
    for fr in raw_frames:
        fdict = {}
        for v in vehiculos_actuales:
            if v.idx in fr:
                pose = fr[v.idx]
                fdict[str(v.vid)] = [round(float(pose[0]), 3), round(float(pose[1]), 3), round(float(pose[2]), 4)]
        frames_json.append(fdict)

    salida_trayectorias = []
    for idx, v in enumerate(vehiculos_actuales):
        traj_data = v.traj
        puntos = [[float(pt[0]), float(pt[1]), float(pt[2])] for pt in traj_data] if traj_data else []
        salida_trayectorias.append({
            "id": v.vid,
            "mision_ok": v.mision_ok,
            "traj": puntos
        })

    return jsonify({
        "ok": True,
        "trayectorias": salida_trayectorias,
        "frames": frames_json,
        "vehiculos": [
            {
                "id": v.vid,
                "inicio": [float(v.inicio[0]), float(v.inicio[1]), float(v.inicio[2])],
                "meta": [float(v.meta[0]), float(v.meta[1])],
                "meta_th": float(v.meta_th) if v.meta_th is not None else None,
                "largo": float(v.length),
                "ancho": float(v.width),
                "grupo": int(getattr(v, "grupo", 1)),
                "prioridad": int(getattr(v, "prioridad", 1)),
                "color": mve.PALETA[idx % len(mve.PALETA)]
            }
            for idx, v in enumerate(vehiculos_actuales)
        ],
        "motivos": {str(k): str(v) for k, v in motivos.items()}
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    print(f"Servidor listo en http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=debug)

