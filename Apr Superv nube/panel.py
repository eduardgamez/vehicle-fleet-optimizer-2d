#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PANEL de control de la búsqueda: la gráfica en directo con botones para
arrancarla y pararla.

Una página HTML suelta no puede lanzar ni matar procesos, así que esto levanta
un servidor diminuto en el propio ordenador (nada sale a internet) que sirve la
gráfica y atiende los botones.

Sobrevive a apagar el ordenador: los trabajadores se reconocen por su línea de
comandos, no por haberlos lanzado este panel. Así que si arrancas el panel
después de un reinicio, verás "parado" y con darle a Arrancar sigue por donde
iba —lo evaluado está en la base de datos del estudio, no en la memoria de
ningún proceso.

Uso:
    python panel.py                    → http://localhost:8770
    python panel.py --puerto 9000 --procesos 4
"""

import argparse
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import psutil

import comun
from comun import MODELOS_DIR
import grafica

RAIZ = os.path.dirname(os.path.abspath(__file__))
GUION = os.path.join(RAIZ, "buscar_optuna.py")


# --------------------------------------------------------------------------- #
# Los trabajadores
# --------------------------------------------------------------------------- #
def trabajadores_vivos():
    """Procesos de búsqueda en marcha AHORA MISMO, los haya lanzado quien los
    haya lanzado. Se miran todos los procesos del sistema en vez de guardar una
    lista propia: así el panel dice la verdad aunque se haya reiniciado el
    ordenador, o aunque los trabajadores se hayan lanzado a mano.

    Cada trabajador aparece DOS veces en la lista de procesos: el python.exe del
    entorno virtual es un lanzador que arranca otro proceso con la misma línea
    de comandos. Se descarta el hijo (el que tiene un padre que también es un
    trabajador) para no contar cada uno por dos, que haría creer al panel que
    hay el doble de los que hay."""
    encontrados = {}
    for p in psutil.process_iter(["pid", "ppid", "name", "cmdline"]):
        try:
            # Tiene que ser un python: si no, cuela cualquier consola que
            # MENCIONE el fichero en su línea de comandos (la que lo lanzó, sin
            # ir más lejos), y el panel cree que hay más trabajadores que reales.
            if not (p.info.get("name") or "").lower().startswith("python"):
                continue
            cmd = p.info.get("cmdline") or []
            if any("buscar_optuna.py" in str(a) for a in cmd):
                encontrados[p.pid] = p
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return [p for pid, p in encontrados.items()
            if p.info.get("ppid") not in encontrados]


class Supervisor:
    """Mantiene en marcha el número de trabajadores pedido.

    Vigila en segundo plano porque un trabajador puede morirse solo: si el
    driver de la tarjeta se reinicia, se lleva por delante el proceso. Sin esta
    vigilancia habría que estar pendiente de relanzarlos a mano."""

    def __init__(self, procesos, frac_vram, n_pruebas, escenarios):
        self.procesos = procesos
        self.frac_vram = frac_vram or round(0.90 / procesos, 2)
        self.n_pruebas = n_pruebas
        self.escenarios = escenarios
        self.activo = False
        self.mios = []
        self.ultimo_aviso = ""
        threading.Thread(target=self._vigilar, daemon=True).start()

    def _lanzar_uno(self, idt):
        cmd = [sys.executable, GUION, "--idt", str(idt),
               "--n-pruebas", str(self.n_pruebas),
               "--frac-vram", str(self.frac_vram),
               "--n-escenarios", str(self.escenarios)]
        # Sin ventana propia y con la salida al vacío: el panel no la lee, y lo
        # que importa queda en la base de datos del estudio.
        #
        # DESENGANCHADO del panel a propósito. Si el trabajador fuese un hijo
        # normal, cerrar el panel de malas maneras (matar la ventana, cerrar la
        # sesión) se llevaría por delante la búsqueda entera. Así sobrevive, que
        # es lo que promete el mensaje al cerrar: el panel solo mira y manda.
        # Sin ventana (CREATE_NO_WINDOW) y suelto del panel
        # (CREATE_BREAKAWAY_FROM_JOB): lo primero evita que aparezcan ventanas
        # negras por el escritorio, lo segundo que cerrar el panel de malas
        # maneras se lleve la búsqueda por delante. Si el sistema no permite
        # soltarlo —hay entornos que lo prohíben— se lanza igualmente sin
        # ventana, que es lo que más molesta.
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

    def arrancar(self):
        self.activo = True
        self._completar()
        return f"En marcha con {self.procesos} procesos."

    def parar(self):
        self.activo = False
        n = 0
        for p in trabajadores_vivos():
            try:
                p.terminate()
                n += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        psutil.wait_procs(trabajadores_vivos(), timeout=8)
        for p in trabajadores_vivos():          # los que se resistan
            try:
                p.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        self.mios = []
        return f"Parados {n} procesos. Lo evaluado queda guardado."

    def _completar(self):
        self.mios = [p for p in self.mios if p.poll() is None]
        faltan = self.procesos - len(trabajadores_vivos())
        for i in range(max(0, faltan)):
            self.mios.append(self._lanzar_uno(len(self.mios) + i))

    def _vigilar(self):
        while True:
            time.sleep(20)
            if self.activo:
                antes = len(trabajadores_vivos())
                if antes < self.procesos:
                    self._completar()
                    self.ultimo_aviso = (
                        f"Relanzados {self.procesos - antes} procesos que se "
                        f"habían caído · {time.strftime('%H:%M')}")


# --------------------------------------------------------------------------- #
# La página
# --------------------------------------------------------------------------- #
BARRA = """
<div class="barra">
  <span class="estado {clase}">{estado}</span>
  <form method="post" action="/arrancar"><button class="on" {dis_on}>
    Arrancar</button></form>
  <form method="post" action="/parar"><button class="off" {dis_off}>
    Parar</button></form>
  <span class="nota">{aviso}</span>
</div>
<style>
  .barra {{ display:flex; align-items:center; gap:12px; margin:0 0 18px; }}
  .barra form {{ margin:0; }}
  .barra button {{ font:600 14px system-ui,sans-serif; padding:9px 20px;
      border:0; border-radius:8px; cursor:pointer; color:#0f1115; }}
  .barra button.on {{ background:#4ade80; }}
  .barra button.off {{ background:#f87171; }}
  .barra button:disabled {{ opacity:.35; cursor:default; }}
  .estado {{ font:600 14px system-ui,sans-serif; padding:8px 14px;
      border-radius:8px; }}
  .estado.va {{ background:#14331f; color:#4ade80; }}
  .estado.no {{ background:#331414; color:#f87171; }}
  .nota {{ color:#9aa0a6; font-size:12px; }}
</style>
"""


def crear_manejador(sup, args):
    class Manejador(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass                                  # sin ruido en la consola

        def _pagina(self):
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            bd = os.path.join(MODELOS_DIR, "optuna.db")
            salida = os.path.join(MODELOS_DIR, "avance.html")
            try:
                estudio = optuna.load_study(study_name=args.estudio,
                                            storage=f"sqlite:///{bd}")
                grafica.construir(estudio, salida, args.refresco,
                                  args.objetivo, args.suavizado)
                with open(salida, encoding="utf-8") as f:
                    doc = f.read()
            except Exception as ex:
                doc = (f"<!doctype html><meta charset=utf-8>"
                       f"<meta http-equiv=refresh content={args.refresco}>"
                       f"<body style='font:15px system-ui;background:#0f1115;"
                       f"color:#e8eaed;padding:28px'>"
                       f"<p>Aún no hay gráfica: {type(ex).__name__}</p>")

            n = len(trabajadores_vivos())
            va = n > 0
            barra = BARRA.format(
                clase="va" if va else "no",
                estado=(f"Buscando · {n} procesos" if va else "Parado"),
                dis_on="disabled" if va else "",
                dis_off="" if va else "disabled",
                aviso=sup.ultimo_aviso)
            # La barra se mete justo detrás del título de la gráfica.
            marca = '<div class="tarjetas">'
            return (doc.replace(marca, barra + marca, 1) if marca in doc
                    else doc + barra)

        def do_GET(self):
            cuerpo = self._pagina().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(cuerpo)))
            self.end_headers()
            self.wfile.write(cuerpo)

        def do_POST(self):
            if self.path == "/arrancar":
                sup.ultimo_aviso = sup.arrancar()
            elif self.path == "/parar":
                sup.ultimo_aviso = sup.parar()
            self.send_response(303)               # de vuelta a la página
            self.send_header("Location", "/")
            self.end_headers()

    return Manejador


def main():
    ap = argparse.ArgumentParser(description="Panel de control de la búsqueda.")
    ap.add_argument("--puerto", type=int, default=8770)
    ap.add_argument("--procesos", type=int, default=4)
    ap.add_argument("--frac-vram", dest="frac_vram", type=float, default=0.0,
                    help="memoria de tarjeta por proceso; 0 = repartir sola")
    ap.add_argument("--n-pruebas", dest="n_pruebas", type=int, default=5000)
    ap.add_argument("--n-escenarios", dest="n_escenarios", type=int, default=400)
    ap.add_argument("--estudio", default="tdr_flota_v2")
    ap.add_argument("--refresco", type=int, default=30)
    ap.add_argument("--suavizado", type=float, default=grafica.SUAVIZADO_MIN)
    ap.add_argument("--objetivo", type=int, default=4800)
    ap.add_argument("--sin-abrir", dest="sin_abrir", action="store_true")
    args = ap.parse_args()

    sup = Supervisor(args.procesos, args.frac_vram, args.n_pruebas,
                     args.n_escenarios)
    servidor = ThreadingHTTPServer(("127.0.0.1", args.puerto),
                                   crear_manejador(sup, args))
    url = f"http://localhost:{args.puerto}/"
    n = len(trabajadores_vivos())
    sup.activo = n > 0                    # adopta lo que ya estuviera corriendo
    print(f"[panel] {url}  ({n} procesos de búsqueda en marcha)", flush=True)
    print("[panel] Ctrl+C cierra el panel; la búsqueda sigue por su cuenta.",
          flush=True)
    if not args.sin_abrir:
        import webbrowser
        webbrowser.open(url)
    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        print("\n[panel] cerrado. Los procesos de búsqueda siguen en marcha.")


if __name__ == "__main__":
    main()
