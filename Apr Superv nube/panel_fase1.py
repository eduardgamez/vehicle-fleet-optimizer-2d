#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PANEL de la FASE 1 (el barrido al azar), en http://localhost:8770

Muestra cómo va y trae los botones de arrancar y parar. Es el equivalente de
`panel.py`, que es el de la fase 2; se separan porque las dos fases guardan sus
resultados de forma distinta: la fase 1 escribe un CSV por proceso y la fase 2
una base de datos compartida.

Lo que enseña, además del avance: el porcentaje de vehículos que CHOCAN y el que
LLEGA, por separado. La nota los mezcla en una cifra, y ahora mismo la pregunta
que hay que responder —si el problema de los choques es de tamaño de red o de
otra cosa— necesita verlos aparte.

Uso:
    python panel_fase1.py
    python panel_fase1.py --puerto 8771 --objetivo 240
"""

import argparse
import csv
import glob
import html
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import psutil

import comun
from comun import MODELOS_DIR

RAIZ = os.path.dirname(os.path.abspath(__file__))
GUION = os.path.join(RAIZ, "entrenar_nube.py")


def trabajadores_vivos():
    """Procesos del barrido en marcha, los haya lanzado quien los haya lanzado.

    Se descarta el hijo cuando su padre también es un trabajador: el python.exe
    del entorno virtual es un lanzador y cada proceso aparece por duplicado."""
    enc = {}
    for p in psutil.process_iter(["pid", "ppid", "name", "cmdline"]):
        try:
            # Tiene que ser un python: si no, cuela cualquier consola que
            # MENCIONE el fichero en su línea de comandos (la que lo lanzó, sin
            # ir más lejos), y el panel cree que hay más trabajadores que reales.
            if not (p.info.get("name") or "").lower().startswith("python"):
                continue
            cmd = p.info.get("cmdline") or []
            if any("entrenar_nube.py" in str(a) for a in cmd):
                enc[p.pid] = p
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return [p for pid, p in enc.items() if p.info.get("ppid") not in enc]


class Supervisor:
    def __init__(self, tareas, n_configs, frac_vram, escenarios):
        self.tareas = tareas
        self.n_configs = n_configs
        self.frac_vram = frac_vram or round(0.90 / tareas, 2)
        self.escenarios = escenarios
        self.activo = False
        self.aviso = ""
        threading.Thread(target=self._vigilar, daemon=True).start()

    def _lanzar(self, idt):
        cmd = [sys.executable, GUION, "--n-configs", str(self.n_configs),
               "--semilla", "0", "--tarea", str(idt), "--tareas",
               str(self.tareas), "--criba=", "--n-escenarios",
               str(self.escenarios), "--frac-vram", str(self.frac_vram)]
        sin_ventana = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        suelto = getattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0)
        for banderas in (sin_ventana | suelto, sin_ventana):
            try:
                return subprocess.Popen(cmd, cwd=RAIZ,
                                        stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL,
                                        creationflags=banderas)
            except OSError:
                continue
        return subprocess.Popen(cmd, cwd=RAIZ, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)

    def _ocupadas(self):
        """Números de tarea que ya están corriendo, para no duplicar ninguna."""
        fuera = set()
        for p in trabajadores_vivos():
            cmd = p.info.get("cmdline") or []
            if "--tarea" in cmd:
                try:
                    fuera.add(int(cmd[cmd.index("--tarea") + 1]))
                except (ValueError, IndexError):
                    pass
        return fuera

    def arrancar(self):
        self.activo = True
        self._completar()
        return f"En marcha: {self.tareas} procesos."

    def parar(self):
        self.activo = False
        vivos = trabajadores_vivos()
        for p in vivos:
            try:
                p.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        psutil.wait_procs(vivos, timeout=8)
        for p in trabajadores_vivos():
            try:
                p.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return f"Parados {len(vivos)}. Lo evaluado queda en el registro."

    def _completar(self):
        ocup = self._ocupadas()
        for i in range(self.tareas):
            if i not in ocup:
                self._lanzar(i)

    def _vigilar(self):
        while True:
            time.sleep(20)
            if self.activo and len(trabajadores_vivos()) < self.tareas:
                antes = len(trabajadores_vivos())
                self._completar()
                self.aviso = (f"Relanzados {self.tareas - antes} procesos "
                              f"caídos · {time.strftime('%H:%M')}")


def leer_registros(carpeta):
    """Filas de todos los barrido_t*.csv, con los números ya convertidos."""
    filas = []
    for f in sorted(glob.glob(os.path.join(carpeta, "barrido_t*.csv"))):
        try:
            with open(f, encoding="utf-8", newline="") as fh:
                for r in csv.DictReader(fh):
                    try:
                        r["_nota"] = float(r["nota_seleccion"])
                        r["_seg"] = float(r["segundos"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    for k, dest in (("pct_choque", "_choque"),
                                    ("pct_llegada", "_llegada")):
                        try:
                            r[dest] = float(r[k])
                        except (KeyError, TypeError, ValueError):
                            r[dest] = None
                    filas.append(r)
        except OSError:
            continue
    return filas


def pagina(sup, args):
    filas = leer_registros(args.registros)
    n = len(filas)
    vivos = len(trabajadores_vivos())

    mejor = max(filas, key=lambda r: r["_nota"]) if filas else None
    seg_total = sum(r["_seg"] for r in filas)
    # Ritmo por reloj: el tiempo de las filas es de cada proceso por separado,
    # así que se divide entre los que había trabajando.
    procesos = max(1, vivos)
    ritmo = (n / (seg_total / procesos / 60.0)) if seg_total else 0.0
    faltan = max(0, args.objetivo - n)
    horas = (faltan / ritmo / 60.0) if ritmo else 0.0

    con_choque = [r for r in filas if r["_choque"] is not None]
    med_choque = (sum(r["_choque"] for r in con_choque) / len(con_choque)
                  if con_choque else None)
    min_choque = min((r["_choque"] for r in con_choque), default=None)

    def caja(t, v, pie=""):
        return (f'<div class="c"><div class="t">{html.escape(t)}</div>'
                f'<div class="v">{html.escape(str(v))}</div>'
                f'<div class="p">{html.escape(pie)}</div></div>')

    cajas = caja("Hechas", f"{n} / {args.objetivo}",
                 f"{vivos} procesos trabajando")
    if mejor:
        cajas += caja("Mejor nota", f"{mejor['_nota']:.4f}",
                      f"{mejor['n_capas']} capas x {mejor['oculto']} · "
                      f"lote {mejor['lote']}")
        if mejor["_choque"] is not None:
            cajas += caja("La mejor choca", f"{mejor['_choque']:.0f} %",
                          f"llega el {mejor['_llegada']:.0f} %")
    if med_choque is not None:
        cajas += caja("Choques (media)", f"{med_choque:.0f} %",
                      f"la que menos: {min_choque:.0f} %")
    cajas += caja("Ritmo", f"{ritmo:.1f}/min",
                  (f"faltan ~{horas:.1f} h" if faltan else "completado"))

    # Tabla: por tamaño de red, para ver si con más parámetros se choca menos.
    porte = {}
    for r in filas:
        if r["_choque"] is None:
            continue
        try:
            clave = f"{r['n_capas']} x {int(float(r['oculto']))}"
        except (KeyError, ValueError):
            continue
        d = porte.setdefault(clave, {"n": 0, "ch": 0.0, "nota": 0.0})
        d["n"] += 1
        d["ch"] += r["_choque"]
        d["nota"] = max(d["nota"], r["_nota"])
    orden = sorted(porte.items(),
                   key=lambda kv: (int(kv[0].split(" x ")[1]),
                                   int(kv[0].split(" x ")[0])))
    tabla = "".join(
        f"<tr><td>{html.escape(k)}</td><td>{v['n']}</td>"
        f"<td>{v['ch']/v['n']:.0f} %</td><td>{v['nota']:.3f}</td></tr>"
        for k, v in orden)

    filas_top = sorted(filas, key=lambda r: -r["_nota"])[:8]
    trozos = []
    for r in filas_top:
        ch = "—" if r["_choque"] is None else f"{r['_choque']:.0f} %"
        trozos.append(f"<tr><td>{r['_nota']:.4f}</td>"
                      f"<td>{r['n_capas']} x {r['oculto']}</td>"
                      f"<td>{r['lote']}</td><td>{r['lr']}</td>"
                      f"<td>{r['mezcla']}</td><td>{ch}</td></tr>")
    top = "".join(trozos)

    va = vivos > 0
    return f"""<!doctype html><html lang="es"><head><meta charset="utf-8">
<meta http-equiv="refresh" content="{args.refresco}">
<title>Fase 1 · barrido al azar</title><style>
 body{{font:15px/1.5 system-ui,sans-serif;margin:0;padding:26px;
      background:#0f1115;color:#e8eaed}}
 h1{{font-size:19px;margin:0 0 3px}} .sub{{color:#9aa0a6;font-size:13px;
      margin-bottom:20px}}
 .barra{{display:flex;align-items:center;gap:12px;margin-bottom:20px}}
 .barra form{{margin:0}}
 .barra button{{font:600 14px system-ui;padding:9px 20px;border:0;
      border-radius:8px;cursor:pointer;color:#0f1115}}
 .on{{background:#4ade80}} .off{{background:#f87171}}
 button:disabled{{opacity:.35;cursor:default}}
 .est{{font:600 14px system-ui;padding:8px 14px;border-radius:8px}}
 .va{{background:#14331f;color:#4ade80}} .no{{background:#331414;color:#f87171}}
 .cajas{{display:flex;flex-wrap:wrap;gap:12px;margin-bottom:22px}}
 .c{{background:#181b21;border:1px solid #262a33;border-radius:10px;
      padding:12px 16px;min-width:140px}}
 .t{{color:#9aa0a6;font-size:12px}} .v{{font-size:24px;font-weight:600}}
 .p{{color:#6b7280;font-size:11px}}
 table{{border-collapse:collapse;margin:0 24px 22px 0;font-size:13px}}
 th,td{{text-align:left;padding:5px 14px 5px 0;
      border-bottom:1px solid #262a33}}
 th{{color:#9aa0a6;font-weight:600}}
 .zona{{display:flex;flex-wrap:wrap}}
 h2{{font-size:14px;color:#9aa0a6;margin:0 0 8px;font-weight:600}}
 .nota{{color:#6b7280;font-size:12px}}
</style></head><body>
<h1>Fase 1 · barrido al azar</h1>
<div class="sub">Se actualiza sola cada {args.refresco} s ·
 {time.strftime('%H:%M:%S')}</div>
<div class="barra">
 <span class="est {'va' if va else 'no'}">
   {'Trabajando · ' + str(vivos) + ' procesos' if va else 'Parado'}</span>
 <form method="post" action="/arrancar"><button class="on"
   {'disabled' if va else ''}>Arrancar</button></form>
 <form method="post" action="/parar"><button class="off"
   {'' if va else 'disabled'}>Parar</button></form>
 <span class="nota">{html.escape(sup.aviso)}</span>
</div>
<div class="cajas">{cajas}</div>
<div class="zona">
<div><h2>Choques por tamaño de red</h2><table>
<tr><th>capas x ancho</th><th>probadas</th><th>choque medio</th>
<th>mejor nota</th></tr>{tabla or '<tr><td>—</td></tr>'}</table></div>
<div><h2>Las mejores</h2><table>
<tr><th>nota</th><th>red</th><th>lote</th><th>lr</th><th>mezcla</th>
<th>choque</th></tr>{top or '<tr><td>—</td></tr>'}</table></div>
</div>
<div class="nota">Chocar hunde la nota por debajo de 0,1, así que una nota alta
ya implica conducir limpio.</div>
</body></html>"""


def crear_manejador(sup, args):
    class Manejador(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            cuerpo = pagina(sup, args).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(cuerpo)))
            self.end_headers()
            self.wfile.write(cuerpo)

        def do_POST(self):
            if self.path == "/arrancar":
                sup.aviso = sup.arrancar()
            elif self.path == "/parar":
                sup.aviso = sup.parar()
            self.send_response(303)
            self.send_header("Location", "/")
            self.end_headers()

    return Manejador


def main():
    ap = argparse.ArgumentParser(description="Panel de la fase 1.")
    ap.add_argument("--puerto", type=int, default=8770)
    ap.add_argument("--registros", default=MODELOS_DIR)
    ap.add_argument("--tareas", type=int, default=3)
    ap.add_argument("--n-configs", dest="n_configs", type=int, default=240)
    ap.add_argument("--objetivo", type=int, default=240)
    ap.add_argument("--frac-vram", dest="frac_vram", type=float, default=0.27)
    ap.add_argument("--n-escenarios", dest="n_escenarios", type=int, default=400)
    ap.add_argument("--refresco", type=int, default=30)
    ap.add_argument("--sin-abrir", dest="sin_abrir", action="store_true")
    args = ap.parse_args()

    sup = Supervisor(args.tareas, args.n_configs, args.frac_vram,
                     args.n_escenarios)
    sup.activo = len(trabajadores_vivos()) > 0
    servidor = ThreadingHTTPServer(("127.0.0.1", args.puerto),
                                   crear_manejador(sup, args))
    url = f"http://localhost:{args.puerto}/"
    print(f"[panel1] {url}  ({len(trabajadores_vivos())} procesos en marcha)",
          flush=True)
    if not args.sin_abrir:
        import webbrowser
        webbrowser.open(url)
    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        print("\n[panel1] cerrado. El barrido sigue por su cuenta.")


if __name__ == "__main__":
    main()
