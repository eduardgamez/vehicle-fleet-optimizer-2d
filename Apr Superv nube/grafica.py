#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gráfica EN DIRECTO de cómo avanza la búsqueda.

Escribe una página HTML que se refresca sola, para tenerla abierta en el
navegador mientras la tarjeta trabaja. Sirve para decidir cuándo parar: cuando
la línea del mejor lleva horas plana, seguir buscando ya no compensa.

Muestra:
  · la NOTA DEL MEJOR hasta cada momento (la línea que importa),
  · cada prueba individual como un punto, para ver la nube y cuánto se está
    explorando fuera de la zona buena,
  · el ritmo, el porcentaje de abandonos y cuánto hace que no mejora.

No usa ninguna librería de dibujo: el gráfico es SVG escrito a mano, así que
funciona sin internet y sin instalar nada.

Uso:
    python grafica.py                 (la genera una vez y sale)
    python grafica.py --bucle         (la rehace cada 30 s hasta que la cortes)
    python grafica.py --abrir         (además la abre en el navegador)
"""

import argparse
import datetime as dt
import html
import os
import time
import webbrowser

from comun import MODELOS_DIR

ANCHO, ALTO = 1000, 420
MARGEN = {"i": 62, "d": 24, "arr": 24, "ab": 46}


# Minutos de pruebas que entran en cada punto de la media. Solo las que entrenan
# enteras tienen nota, así que la ventana tiene que dar para varias: con el
# abandono bajado a la mitad caen unas cinco en cinco minutos.
SUAVIZADO_MIN = 5.0


def media_suavizada(puntos, ventana_min=SUAVIZADO_MIN):
    """Nota MEDIA de las pruebas de los últimos 'ventana_min' minutos, punto a
    punto.

    Es el segundo indicador de que la búsqueda va bien: la línea del mejor solo
    puede subir (basta un golpe de suerte), pero la media sube únicamente si el
    muestreador propone candidatas mejores de verdad. No debe alcanzar a la del
    mejor: si lo hiciera, sería que ha dejado de explorar.

    La ventana es de TIEMPO y no de número de pruebas, y se desliza en vez de ir
    a saltos: así la línea no pega botes cada vez que entra una candidata mala,
    y sigue significando lo mismo aunque el ritmo cambie."""
    if not puntos or ventana_min <= 0:
        return []
    salida, ini, suma = [], 0, 0.0
    for i, (t, v) in enumerate(puntos):
        suma += v
        while puntos[ini][0] < t - ventana_min:   # sale lo que ya no cae dentro
            suma -= puntos[ini][1]
            ini += 1
        salida.append((t, suma / (i - ini + 1)))
    return salida


def _trazar(puntos, mejores, medias, t_max, y_min, y_max, ini=None,
            abandonos=()):
    """Devuelve el SVG del gráfico. Las tres listas son de pares (t_min, nota)."""
    x0, y0 = MARGEN["i"], MARGEN["arr"]
    w = ANCHO - MARGEN["i"] - MARGEN["d"]
    h = ALTO - MARGEN["arr"] - MARGEN["ab"]
    rango = max(y_max - y_min, 1e-6)

    def px(t):
        return x0 + (t / t_max if t_max else 0) * w

    def py(v):
        return y0 + h - (v - y_min) / rango * h

    partes = []
    # Rejilla horizontal con sus etiquetas.
    for k in range(6):
        v = y_min + rango * k / 5
        y = py(v)
        partes.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0+w}" y2="{y:.1f}" '
                      f'class="rejilla"/>')
        partes.append(f'<text x="{x0-10}" y="{y+4:.1f}" class="eje" '
                      f'text-anchor="end">{v:.2f}</text>')
    # Rejilla vertical: unas 8 marcas, con la HORA DEL RELOJ y, debajo, cuánto
    # tiempo llevaba la búsqueda en ese momento. La hora del reloj es lo que
    # permite atar un salto de la gráfica a lo que estuviera pasando entonces.
    minutos_totales = max(t_max, 1.0)
    for escala in (5, 10, 15, 30, 60, 120, 180, 360, 720, 1440):
        if minutos_totales / escala <= 8:
            break
    marca = 0.0
    while marca <= minutos_totales:
        x = px(marca)
        partes.append(f'<line x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y0+h}" '
                      f'class="rejilla"/>')
        if ini is not None:
            reloj = (ini + dt.timedelta(minutes=marca)).strftime("%H:%M")
            partes.append(f'<text x="{x:.1f}" y="{y0+h+20}" class="eje" '
                          f'text-anchor="middle">{reloj}</text>')
            transcurrido = (f"{marca/60:.0f} h" if marca >= 60
                            else f"{marca:.0f} min")
            partes.append(f'<text x="{x:.1f}" y="{y0+h+34}" class="eje tenue" '
                          f'text-anchor="middle">{transcurrido}</text>')
        marca += escala

    # Las abandonadas van como rayitas al pie: no tienen nota comparable (se
    # cortaron antes de conducir las flotas), pero enseñan que la búsqueda está
    # trabajando. Sin ellas la gráfica parece parada, porque con el abandono
    # alto solo una de cada veinte llega a ser un punto azul.
    for t in abandonos:
        x = px(t)
        partes.append(f'<line x1="{x:.1f}" y1="{y0+h-6}" x2="{x:.1f}" '
                      f'y2="{y0+h}" class="abandono"/>')
    for t, v in puntos:
        partes.append(f'<circle cx="{px(t):.1f}" cy="{py(v):.1f}" r="2.1" '
                      f'class="punto"/>')
    if len(medias) > 1:
        d = " ".join(("M" if i == 0 else "L") + f"{px(t):.1f},{py(v):.1f}"
                     for i, (t, v) in enumerate(medias))
        partes.append(f'<path d="{d}" class="media"/>')
    if mejores:
        d = []
        for i, (t, v) in enumerate(mejores):
            d.append(("M" if i == 0 else "L") + f"{px(t):.1f},{py(v):.1f}")
            if i + 1 < len(mejores):     # escalón: se mantiene hasta la siguiente
                d.append(f"L{px(mejores[i+1][0]):.1f},{py(v):.1f}")
        d.append(f"L{px(t_max):.1f},{py(mejores[-1][1]):.1f}")
        partes.append(f'<path d="{" ".join(d)}" class="mejor"/>')
        tf, vf = mejores[-1]
        partes.append(f'<circle cx="{px(t_max):.1f}" cy="{py(vf):.1f}" r="4.5" '
                      f'class="punta"/>')
    return "\n".join(partes)


def construir(estudio, salida, refresco, objetivo=4800,
              suavizado=SUAVIZADO_MIN):
    from optuna.trial import TrialState

    ts = estudio.get_trials(deepcopy=False)
    # Las de siembra se insertaron de golpe y no pasaron por el entrenamiento:
    # se reconocen porque no tienen ni anotaciones ni hitos intermedios.
    reales = [t for t in ts
              if t.datetime_start and t.datetime_complete
              and (t.user_attrs.get("segundos") is not None
                   or t.intermediate_values)]
    siembra = [t for t in ts if t not in reales
               and t.state == TrialState.COMPLETE and t.value is not None]

    ini = min((t.datetime_start for t in reales), default=None)
    ahora = dt.datetime.now()

    puntos, mejores = [], []
    mejor = max((t.value for t in siembra), default=None)
    if mejor is not None:
        mejores.append((0.0, mejor))     # de dónde se partía
    for t in sorted(reales, key=lambda x: x.datetime_complete):
        if t.state != TrialState.COMPLETE or t.value is None:
            continue
        tm = (t.datetime_complete - ini).total_seconds() / 60.0
        puntos.append((tm, t.value))
        if mejor is None or t.value > mejor:
            mejor = t.value
            mejores.append((tm, mejor))

    t_max = ((ahora - ini).total_seconds() / 60.0) if ini else 1.0
    completas = [t for t in reales if t.state == TrialState.COMPLETE]
    # Dos cosas muy distintas que Optuna mete en el mismo saco:
    #   · CORTADAS pronto — iban peor que las demás y no se les dio más tiempo.
    #     Es el funcionamiento normal, no un error.
    #   · FALLIDAS — no cabían en memoria o la red se fue a infinito.
    # Además, una cortada deja de contar en cuanto esa misma configuración se
    # vuelve a lanzar y termina: ya no es esfuerzo perdido.
    resueltas = {tuple(sorted(t.params.items())) for t in completas}
    sin_acabar = [t for t in reales if t.state == TrialState.PRUNED
                  and tuple(sorted(t.params.items())) not in resueltas]
    fallidas = [t for t in sin_acabar if t.user_attrs.get("fallo")]
    podadas = [t for t in sin_acabar if not t.user_attrs.get("fallo")]

    # Ritmo de la última hora (o de todo si lleva menos).
    corte = ahora - dt.timedelta(hours=1)
    ult = [t for t in reales if t.datetime_complete >= corte]
    minutos = min(60.0, t_max) or 1.0
    ritmo = len(ult) / (minutos / 60.0)

    # Cuánto falta para llegar al objetivo. Se usa el ritmo RECIENTE, no el
    # medio de toda la búsqueda: el coste por prueba cambia según se abandonan
    # más o menos y según el tamaño de las redes que va proponiendo, así que la
    # última hora predice mejor la siguiente que el promedio de todas.
    faltan = max(0, objetivo - len(reales))
    if faltan == 0:
        eta_txt, eta_pie = "completado", f"{len(reales):,} pruebas"
    elif ritmo > 0:
        horas = faltan / ritmo
        eta_txt = f"{horas:.1f} h" if horas < 48 else f"{horas/24:.1f} días"
        fin = ahora + dt.timedelta(hours=horas)
        mismo_dia = fin.date() == ahora.date()
        eta_pie = (f"faltan {faltan:,} · acaba "
                   + (f"a las {fin:%H:%M}" if mismo_dia else f"{fin:%d/%m %H:%M}"))
    else:
        eta_txt, eta_pie = "—", f"faltan {faltan:,}"

    sin_mejora = (t_max - mejores[-1][0]) if len(mejores) > 1 else t_max
    valores = [v for _, v in puntos] or [0.0]
    y_max = max(max(valores), mejor or 0) * 1.02
    y_min = min(min(valores), 0.0)

    abandonos = sorted((t.datetime_complete - ini).total_seconds() / 60.0
                       for t in podadas)
    medias = media_suavizada(puntos, suavizado)
    grafico = _trazar(puntos, mejores, medias, max(t_max, 1.0), y_min, y_max,
                      ini, abandonos)

    def tarjeta(titulo, valor, pie=""):
        return (f'<div class="t"><div class="tt">{html.escape(titulo)}</div>'
                f'<div class="tv">{html.escape(str(valor))}</div>'
                f'<div class="tp">{html.escape(pie)}</div></div>')

    mejor_txt = f"{mejor:.4f}" if mejor is not None else "—"
    tarjetas = "".join([
        # Al pie, el récord ANTERIOR: así se ve de un vistazo cuánto ha subido
        # el último salto, y cambia solo cada vez que se bate la marca.
        tarjeta("Mejor nota", mejor_txt,
                (f"antes {mejores[-2][1]:.4f}" if len(mejores) > 1
                 else "aún sin superar la fase 1")),
        tarjeta("Media reciente", f"{medias[-1][1]:.3f}" if medias else "—",
                (f"últimos {suavizado:.0f} min" if medias
                 else "esperando pruebas")),
        tarjeta("Sin mejorar", f"{sin_mejora/60:.1f} h", "desde el último récord"),
        tarjeta("Falta", eta_txt, eta_pie),
        # Para el objetivo cuentan TODAS las candidatas exploradas, enteras o
        # cortadas: lo que se mide es cuánto espacio se ha barrido, y una
        # cortada también informa (esa zona no valía). Es además lo que usa la
        # cuenta atrás de "Falta", que si no diría cosas distintas.
        tarjeta("Pruebas", f"{len(reales):,} / {objetivo:,}",
                f"{len(completas):,} enteras · {len(podadas):,} cortadas"
                + (f" · {len(fallidas):,} fallidas" if fallidas else "")),
        tarjeta("Ritmo", f"{ritmo/60:.1f}/min", f"lleva {t_max/60:.1f} h"),
    ])

    doc = f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<meta http-equiv="refresh" content="{refresco}">
<title>Búsqueda de la red · en directo</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font: 15px/1.5 system-ui, sans-serif; margin: 0; padding: 28px;
         background: #0f1115; color: #e8eaed; }}
  h1 {{ font-size: 19px; margin: 0 0 4px; font-weight: 600; }}
  .sub {{ color: #9aa0a6; font-size: 13px; margin-bottom: 22px; }}
  .tarjetas {{ display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 22px; }}
  .t {{ background: #181b21; border: 1px solid #262a33; border-radius: 10px;
        padding: 12px 16px; min-width: 140px; }}
  .tt {{ color: #9aa0a6; font-size: 12px; }}
  .tv {{ font-size: 24px; font-weight: 600; margin: 2px 0; }}
  .tp {{ color: #6b7280; font-size: 11px; }}
  .caja {{ background: #181b21; border: 1px solid #262a33; border-radius: 10px;
           padding: 14px; overflow-x: auto; }}
  svg {{ display: block; max-width: 100%; height: auto; }}
  .rejilla {{ stroke: #262a33; stroke-width: 1; }}
  .eje {{ fill: #6b7280; font: 11px system-ui, sans-serif; }}
  .tenue {{ fill: #4b5563; font-size: 10px; }}
  .punto {{ fill: #6ea8fe; opacity: .45; }}
  .abandono {{ stroke: #6b7280; stroke-width: 1.4; opacity: .7; }}
  .mejor {{ fill: none; stroke: #4ade80; stroke-width: 2.2; }}
  .media {{ fill: none; stroke: #f59e0b; stroke-width: 1.8;
            stroke-dasharray: 5 4; }}
  .punta {{ fill: #4ade80; }}
  .leyenda {{ color: #9aa0a6; font-size: 12px; margin-top: 10px; }}
  .leyenda b {{ color: #4ade80; }}
  .leyenda i {{ color: #f59e0b; font-style: normal; }}
  /* Los puntos van con opacidad para que la nube no tape las líneas; en texto
     eso se leería gris, así que la palabra lleva el azul a plena intensidad. */
  .leyenda u {{ color: #6ea8fe; text-decoration: none; }}
  .leyenda s {{ color: #6b7280; text-decoration: none; }}
</style></head><body>
<h1>Búsqueda de la red · en directo</h1>
<div class="sub">Se actualiza sola cada {refresco} s · última {ahora:%H:%M:%S}</div>
<div class="tarjetas">{tarjetas}</div>
<div class="caja">
<svg viewBox="0 0 {ANCHO} {ALTO}" width="{ANCHO}" height="{ALTO}">{grafico}</svg>
<div class="leyenda"><b>verde</b> mejor nota · <i>naranja</i> media de los
últimos {suavizado:.0f} min · <u>azul</u> cada candidata entera ·
<s>rayitas grises</s> cortadas pronto por ir peor (sin nota).<br>
Se para entre 3.000 y 5.000 pruebas, cuando ambas se hayan aplanado.</div>
</div>
</body></html>"""

    tmp = salida + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(doc)
    os.replace(tmp, salida)         # así el navegador nunca lee un fichero a medias
    return mejor, len(reales)


def main():
    ap = argparse.ArgumentParser(description="Gráfica en directo de la búsqueda.")
    ap.add_argument("--bd", default=None)
    ap.add_argument("--estudio", default="tdr_flota_v2")
    ap.add_argument("--salida", default=None,
                    help="def. modelos/avance.html")
    ap.add_argument("--refresco", type=int, default=30,
                    help="segundos entre actualizaciones (def. 30)")
    ap.add_argument("--suavizado", type=float, default=SUAVIZADO_MIN,
                    help="minutos de pruebas que entran en cada punto de la "
                         "línea de la media (def. 3)")
    ap.add_argument("--objetivo", type=int, default=4800,
                    help="pruebas a las que se quiere llegar, para estimar "
                         "cuánto falta (def. 4800)")
    ap.add_argument("--bucle", action="store_true",
                    help="rehacerla continuamente hasta que la cortes")
    ap.add_argument("--abrir", action="store_true",
                    help="abrirla en el navegador al empezar")
    args = ap.parse_args()

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    bd = args.bd or os.path.join(MODELOS_DIR, "optuna.db")
    salida = args.salida or os.path.join(MODELOS_DIR, "avance.html")
    estudio = optuna.load_study(study_name=args.estudio,
                                storage=f"sqlite:///{bd}")

    mejor, n = construir(estudio, salida, args.refresco, args.objetivo,
                         args.suavizado)
    print(f"[grafica] {n} pruebas · mejor {mejor} → {salida}", flush=True)
    if args.abrir:
        webbrowser.open(f"file:///{salida.replace(os.sep, '/')}")
    if not args.bucle:
        return
    print(f"[grafica] actualizando cada {args.refresco} s. Ctrl+C para parar.",
          flush=True)
    while True:
        time.sleep(args.refresco)
        try:
            estudio = optuna.load_study(study_name=args.estudio,
                                        storage=f"sqlite:///{bd}")
            construir(estudio, salida, args.refresco, args.objetivo,
                      args.suavizado)
        except Exception as ex:      # la base de datos puede estar ocupada
            print(f"[grafica] reintento tras {type(ex).__name__}", flush=True)


if __name__ == "__main__":
    main()

