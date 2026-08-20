#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VISOR de las rutas del dataset de entrenamiento. Solo mira: no planifica nada.

`multi_v_evo.py` calcula rutas; esto reproduce las que YA están calculadas, las
de datos/rutas, para poder juzgar a ojo si el planificador las hizo bien. No
importa el planificador ni la red: lee el CSV, reconstruye las poses y las pinta
sobre el mismo mapa con el que se generaron.

Además de verlas, las AUDITA con el mismo criterio que usa la nota del barrido
(`vectorizado.choques_entre_vehiculos` y `choques_con_mapa`, SAT sobre
rectángulos orientados). De cada run dice si algún vehículo toca a otro o al
mapa, cuándo, si todos llegaron a su plaza y con cuánto error de ángulo. En la
animación, un vehículo que está tocando algo se pinta en ROJO.

Eso es lo que contesta la pregunta "¿estas rutas están bien hechas?": no hay que
fiarse de la vista, porque el planificador conduce con holguras de un palmo y a
simple vista un roce y un casi-roce son iguales.

Ejecutar:   python visor_rutas.py
            python visor_rutas.py --rutas datos/rutas
"""

import argparse
import glob
import math
import os

import numpy as np

import comun                       # añade «Apr Superv local» al path
import escenarios as esc
import preparar_datos as prep
import vectorizado as vec
from nucleo import DT, W, H, SCALE, obb_corners, ang_norm

PALETA = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#46f0f0",
          "#f032e6", "#bcf60c", "#fabebe", "#008080", "#9a6324", "#800000",
          "#808000", "#000075", "#808080", "#e6beff"]
COL_OBST = "#2b2b2b"
COL_FONDO = "#f5f5f5"
COL_CHOQUE = "#ff0000"


# --------------------------------------------------------------------------- #
# Carga de un run a poses
# --------------------------------------------------------------------------- #
class Run:
    """Un escenario ya planificado, listo para pintar y auditar.

    'poses' es (T, M, 3) con (x, y, θ) de cada vehículo en cada instante. Los
    vehículos que acaban antes se quedan APARCADOS en su pose final, que es lo
    que de verdad ven los que siguen circulando —y por tanto lo que hay que
    mirar para saber si alguien choca contra uno ya aparcado—."""

    def __init__(self, run_id, conds, opt, datos):
        self.run_id = run_id
        self.opt = opt
        self.vids = [v for v in datos if v in conds]
        self.conds = conds
        M = len(self.vids)
        largos = [len(datos[v]) for v in self.vids]
        T = max(largos) if largos else 0
        self.T = T
        self.dur = [n * DT for n in largos]

        poses = np.zeros((T, M, 3))
        for k, vid in enumerate(self.vids):
            arr = datos[vid]
            n = len(arr)
            poses[:n, k, :] = arr[:, :3]
            if n < T:
                poses[n:, k, :] = arr[-1, :3]
        self.poses = poses

        self.largo = np.array([conds[v]["largo"] for v in self.vids])
        self.ancho = np.array([conds[v]["ancho"] for v in self.vids])
        self.meta = np.array([conds[v]["meta"] for v in self.vids])
        self.meta_th = [conds[v]["meta_th"] for v in self.vids]
        self.inicio = poses[0].copy() if T else np.zeros((M, 3))

    @property
    def n_veh(self):
        return len(self.vids)

    def auditar(self, obst, mundo):
        """(tocando, informe): 'tocando' es (T, M) booleana con quién está
        tocando algo en cada instante; 'informe' un texto con el veredicto."""
        M, T = self.n_veh, self.T
        if M == 0 or T == 0:
            return np.zeros((0, 0), bool), "run vacío"

        par = {"largo": self.largo, "ancho": self.ancho}
        if M > 1:
            idx_otros = np.array([[j for j in range(M) if j != k]
                                  for k in range(M)])
        else:
            idx_otros = np.zeros((1, 0), dtype=int)
        valido = np.ones(idx_otros.shape, dtype=bool)

        tocando = np.zeros((T, M), dtype=bool)
        entre = np.zeros((T, M), dtype=bool)
        mapa = np.zeros((T, M), dtype=bool)
        est = np.zeros((M, 4))
        for t in range(T):
            est[:, :3] = self.poses[t]
            if M > 1:
                entre[t] = vec.choques_entre_vehiculos(est, par, idx_otros,
                                                       valido)
            mapa[t] = vec.choques_con_mapa(est, par, obst, mundo)
            tocando[t] = entre[t] | mapa[t]

        lin = []
        n_entre = int(entre.any(axis=0).sum())
        n_mapa = int(mapa.any(axis=0).sum())
        if n_entre or n_mapa:
            lin.append("CHOQUES: %d vehiculo(s) tocan a otro, %d tocan el mapa"
                       % (n_entre, n_mapa))
            for k in range(M):
                if tocando[:, k].any():
                    t0 = int(np.argmax(tocando[:, k]))
                    seg = float(tocando[:, k].sum()) * DT
                    que = []
                    if entre[:, k].any():
                        que.append("otro vehiculo")
                    if mapa[:, k].any():
                        que.append("el mapa")
                    lin.append("   veh %s: desde t=%.1f s, %.1f s tocando %s"
                               % (self.vids[k], t0 * DT, seg, " y ".join(que)))
        else:
            lin.append("LIMPIO: nadie toca a nadie ni al mapa en todo el run")

        lin.append("")
        lin.append("Llegada a la plaza:")
        for k in range(M):
            fx, fy, fth = self.poses[-1, k]
            d = math.hypot(fx - self.meta[k, 0], fy - self.meta[k, 1])
            th_ex = self.meta_th[k]
            if th_ex is None:
                ang = "libre"
            else:
                ang = "%.1f grados" % abs(math.degrees(ang_norm(fth - th_ex)))
            lin.append("   veh %s: a %.3f de la meta, error de angulo %s, "
                       "tarda %.1f s" % (self.vids[k], d, ang, self.dur[k]))
        return tocando, "\n".join(lin)


def cargar_csv(ruta):
    """[Run] de un fichero de rutas, usando el MISMO lector que el preparador de
    datos: si el visor leyera el CSV por su cuenta y se desviara un milímetro,
    estaría enseñando algo distinto de lo que entrena la red."""
    return [Run(rid, conds, opt, datos)
            for rid, conds, opt, datos in prep.leer_archivo(ruta)]


def listar_csv(carpeta):
    """Rutas de todos los CSV, incluidas las subcarpetas por tamaño de flota."""
    fich = glob.glob(os.path.join(carpeta, "**", "*.csv"), recursive=True)
    return sorted(fich)


# --------------------------------------------------------------------------- #
# Ventana
# --------------------------------------------------------------------------- #
class Visor:
    def __init__(self, root, carpeta):
        import tkinter as tk
        from tkinter import ttk
        self.tk, self.ttk = tk, ttk
        self.root = root
        root.title("Visor de rutas del dataset · solo reproduce lo calculado")

        self.carpeta = carpeta
        self.ficheros = listar_csv(carpeta)
        self.runs = []
        self.run = None
        self.tocando = None
        self.frame = 0
        self.animando = False
        self.anim_id = None
        self.obst, self.mundo = esc.obstaculos()

        izq = tk.Frame(root)
        izq.pack(side="left", fill="y", padx=6, pady=6)

        tk.Label(izq, text="Fichero de rutas").pack(anchor="w")
        self.cb_fich = ttk.Combobox(izq, width=38, state="readonly",
                                    values=[os.path.relpath(f, carpeta)
                                            for f in self.ficheros])
        self.cb_fich.pack()
        self.cb_fich.bind("<<ComboboxSelected>>", self._elegir_fichero)

        tk.Label(izq, text="Escenario (run)").pack(anchor="w", pady=(8, 0))
        marco = tk.Frame(izq)
        marco.pack()
        self.lista = tk.Listbox(marco, width=38, height=18,
                                font=("Consolas", 9), exportselection=False)
        sb = tk.Scrollbar(marco, command=self.lista.yview)
        self.lista.configure(yscrollcommand=sb.set)
        self.lista.pack(side="left")
        sb.pack(side="left", fill="y")
        self.lista.bind("<<ListboxSelect>>", self._elegir_run)

        bot = tk.Frame(izq)
        bot.pack(pady=6, fill="x")
        self.b_play = tk.Button(bot, text="Reproducir", width=11,
                                command=self._play_pausa)
        self.b_play.pack(side="left")
        tk.Button(bot, text="Inicio", width=7,
                  command=self._reiniciar).pack(side="left", padx=3)
        tk.Label(bot, text="x").pack(side="left", padx=(8, 0))
        self.velocidad = ttk.Combobox(bot, width=4, state="readonly",
                                      values=["0.25", "0.5", "1", "2", "4"])
        self.velocidad.set("1")
        self.velocidad.pack(side="left")

        self.barra = ttk.Scale(izq, from_=0, to=1, orient="horizontal",
                               command=self._mover_barra)
        self.barra.pack(fill="x")

        self.chk_rutas = tk.BooleanVar(value=True)
        tk.Checkbutton(izq, text="dibujar las trayectorias completas",
                       variable=self.chk_rutas,
                       command=self._pintar).pack(anchor="w")

        tk.Label(izq, text="Auditoria del escenario").pack(anchor="w",
                                                          pady=(8, 0))
        self.texto = tk.Text(izq, width=46, height=16, font=("Consolas", 8),
                             wrap="none")
        self.texto.pack()

        self.canvas = tk.Canvas(root, width=int(W * SCALE), height=int(H * SCALE),
                                bg=COL_FONDO, highlightthickness=0)
        self.canvas.pack(side="left", padx=6, pady=6)

        if self.ficheros:
            self.cb_fich.current(0)
            self._elegir_fichero()

    # ------------------------------- carga -------------------------------- #
    def _elegir_fichero(self, *_):
        i = self.cb_fich.current()
        if i < 0:
            return
        self._parar()
        self.lista.delete(0, "end")
        self.lista.insert("end", " cargando…")
        self.root.update_idletasks()
        self.runs = cargar_csv(self.ficheros[i])
        self.lista.delete(0, "end")
        for r in self.runs:
            self.lista.insert("end", " run %-5s  %d veh  %-12s %5.1f s"
                              % (r.run_id, r.n_veh, r.opt, r.T * DT))
        if self.runs:
            self.lista.selection_set(0)
            self._elegir_run()

    def _elegir_run(self, *_):
        sel = self.lista.curselection()
        if not sel:
            return
        self._parar()
        self.run = self.runs[sel[0]]
        self.tocando, informe = self.run.auditar(self.obst, self.mundo)
        self.texto.delete("1.0", "end")
        self.texto.insert("end", "run %s · %d vehiculos · modo %s · %.1f s\n\n"
                          % (self.run.run_id, self.run.n_veh, self.run.opt,
                             self.run.T * DT))
        self.texto.insert("end", informe)
        self.frame = 0
        self.barra.configure(to=max(self.run.T - 1, 1))
        self.barra.set(0)
        self._pintar()

    # ----------------------------- animación ------------------------------ #
    def _play_pausa(self):
        if self.animando:
            self._parar()
        elif self.run is not None:
            self.animando = True
            self.b_play.configure(text="Pausa")
            self._paso()

    def _parar(self):
        self.animando = False
        self.b_play.configure(text="Reproducir")
        if self.anim_id is not None:
            self.root.after_cancel(self.anim_id)
            self.anim_id = None

    def _reiniciar(self):
        self._parar()
        self.frame = 0
        self.barra.set(0)
        self._pintar()

    def _paso(self):
        if not self.animando or self.run is None:
            return
        if self.frame >= self.run.T - 1:
            self._parar()
            return
        self.frame += 1
        self.barra.set(self.frame)
        self._pintar()
        try:
            v = float(self.velocidad.get())
        except ValueError:
            v = 1.0
        self.anim_id = self.root.after(max(int(DT * 1000 / max(v, 0.01)), 1),
                                       self._paso)

    def _mover_barra(self, valor):
        if self.run is None:
            return
        f = int(float(valor))
        if f != self.frame:
            self.frame = min(max(f, 0), self.run.T - 1)
            self._pintar()

    # ------------------------------- dibujo ------------------------------- #
    def _px(self, poly):
        out = []
        for x, y in poly:
            out.extend((x * SCALE, y * SCALE))
        return out

    def _pintar(self):
        c = self.canvas
        c.delete("all")
        for poly in self._polis_mapa():
            c.create_polygon(self._px(poly), fill=COL_OBST, outline=COL_OBST)
        r = self.run
        if r is None:
            return

        for k in range(r.n_veh):
            col = PALETA[k % len(PALETA)]
            mx, my = r.meta[k]
            c.create_oval(mx * SCALE - 7, my * SCALE - 7,
                          mx * SCALE + 7, my * SCALE + 7, outline=col, width=2)
            if r.meta_th[k] is not None:
                ax = mx + 0.7 * math.cos(r.meta_th[k])
                ay = my + 0.7 * math.sin(r.meta_th[k])
                c.create_line(mx * SCALE, my * SCALE, ax * SCALE, ay * SCALE,
                              fill=col, width=1, arrow="last",
                              arrowshape=(6, 7, 2))
            # Pose de salida, a trazos.
            c.create_polygon(self._px(obb_corners(r.inicio[k, 0], r.inicio[k, 1],
                                                  r.inicio[k, 2], r.largo[k],
                                                  r.ancho[k])),
                             outline=col, fill="", width=1, dash=(3, 3))

        if self.chk_rutas.get():
            for k in range(r.n_veh):
                pts = []
                for t in range(r.T):
                    pts.extend((r.poses[t, k, 0] * SCALE,
                                r.poses[t, k, 1] * SCALE))
                if len(pts) >= 4:
                    c.create_line(pts, fill=PALETA[k % len(PALETA)], width=1,
                                  smooth=True)

        t = min(self.frame, r.T - 1)
        for k in range(r.n_veh):
            x, y, th = r.poses[t, k]
            choca = bool(self.tocando[t, k]) if self.tocando is not None else False
            col = COL_CHOQUE if choca else PALETA[k % len(PALETA)]
            c.create_polygon(self._px(obb_corners(x, y, th, r.largo[k],
                                                  r.ancho[k])),
                             fill=col, outline="#101010", width=1)
            nx = x + math.cos(th) * r.largo[k] * 0.35
            ny = y + math.sin(th) * r.largo[k] * 0.35
            c.create_line(x * SCALE, y * SCALE, nx * SCALE, ny * SCALE,
                          fill="white", width=2)
            c.create_text(x * SCALE, y * SCALE, text=str(r.vids[k]),
                          fill="white", font=("", 9, "bold"))

        n_toc = int(self.tocando[t].sum()) if self.tocando is not None else 0
        aviso = "  ·  TOCANDO: %d" % n_toc if n_toc else ""
        c.create_text(10, 12, anchor="w",
                      text="t = %.1f s   (%d/%d)%s"
                           % (t * DT, t, r.T - 1, aviso),
                      fill=COL_CHOQUE if n_toc else "#202020",
                      font=("", 11, "bold"))

    def _polis_mapa(self):
        """Polígonos de los obstáculos fijos, cacheados: el mapa no cambia."""
        if not hasattr(self, "_cache_polis"):
            from nucleo import Entorno, cargar_mapa_en
            env = Entorno()
            cargar_mapa_en(env, esc.MAPA_ENTRENAMIENTO)
            self._cache_polis = list(env.obstaculos)
        return self._cache_polis


def main():
    ap = argparse.ArgumentParser(description="Visor de las rutas del dataset.")
    ap.add_argument("--rutas", default=comun.RUTAS_DIR,
                    help="carpeta con los CSV de rutas")
    args = ap.parse_args()
    if not listar_csv(args.rutas):
        raise SystemExit("No hay CSV de rutas en %s" % args.rutas)

    import tkinter as tk
    root = tk.Tk()
    root.resizable(False, False)
    Visor(root, args.rutas)
    root.mainloop()


if __name__ == "__main__":
    main()
