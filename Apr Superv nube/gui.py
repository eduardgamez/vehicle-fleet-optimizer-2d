#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Interfaz visual para probar los modelos entrenados en «Apr Superv nube».

Reutiliza la interfaz de «Apr Superv local», pero ejecuta la red con el mismo
rollout vectorizado, mapa y detección de choques que se usan para puntuarla en
las fases de selección. Los vehículos se pintan en rojo mientras están tocando
otro vehículo o el mapa.

Uso:
    python gui.py
    python gui.py --modelo datos/modelos/fase5/mejor_c5-d20-e640.pt
    python gui.py --escenario 25
"""

import argparse
import glob
import importlib.util
import os

import numpy as np

from comun import MAPA_ENTRENAMIENTO, MODELOS_DIR, RAIZ_LOCAL
import escenarios as esc
import vectorizado as vec
from entrenar import _puntuar_vehiculo
from nucleo import (COL_BORDE, DT, H, PALETA, SCALE, W, Planificador,
                    cargar_mapa_en, construir_frames, especs_a_texto,
                    obb_corners)


def _cargar_gui_local():
    ruta = os.path.join(RAIZ_LOCAL, "gui.py")
    spec = importlib.util.spec_from_file_location("tdr_gui_local", ruta)
    if spec is None or spec.loader is None:
        raise ImportError("No se puede cargar %s" % ruta)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


GUI_LOCAL = _cargar_gui_local()


def modelos_disponibles(carpeta=MODELOS_DIR):
    return sorted(glob.glob(os.path.join(carpeta, "**", "*.pt"),
                            recursive=True))


def modelo_por_defecto(carpeta=MODELOS_DIR):
    """Modelo definitivo si existe; mientras tanto, el ganador de fase 6."""
    preferidos = [
        os.path.join(carpeta, "politica.pt"),
        os.path.join(carpeta, "fase6", "mejor_c5-d20-e640_s2.pt"),
        os.path.join(carpeta, "fase5", "mejor_c5-d20-e640.pt"),
    ]
    for ruta in preferidos:
        if os.path.isfile(ruta):
            return ruta
    modelos = modelos_disponibles(carpeta)
    return max(modelos, key=os.path.getmtime) if modelos else None


class App(GUI_LOCAL.App):
    def __init__(self, root, modelo=None, escenario=0):
        self.modelo_pt = os.path.abspath(modelo) if modelo else modelo_por_defecto()
        self._indice_escenario = -1
        self._escenarios = []
        self._tocando = None
        super().__init__(root)
        self._adaptar_interfaz_ia()

        # La red se entrenó sobre este mapa concreto. Se carga desde el inicio
        # para que también los vehículos aleatorios se generen en él.
        cargar_mapa_en(self.env, MAPA_ENTRENAMIENTO)
        self.planificador = Planificador(self.env)
        self._actualizar_densidad()
        self.motor.set("ia")
        self.refresco = self.tk.IntVar(value=10)
        self._crear_menu_nube()
        self._actualizar_titulo()

        try:
            self._escenarios = esc.cargar("seleccion")
        except (OSError, ValueError, SystemExit):
            self._escenarios = []
        if self._escenarios:
            self.cargar_escenario(escenario % len(self._escenarios))
        else:
            self._dibujar_estatico()
            self.estado.set("Mapa de entrenamiento cargado. Genera vehículos.")

    def _adaptar_interfaz_ia(self):
        """Retira controles heredados que no intervienen en la inferencia."""
        panel = self.e_num.master
        hijos = panel.winfo_children()

        def valor(widget, clave):
            try:
                return str(widget.cget(clave))
            except self.tk.TclError:
                return ""

        def fila(widget):
            info = widget.grid_info()
            return int(info["row"]) if info else None

        def ocultar_filas(*filas):
            for widget in hijos:
                if fila(widget) in filas:
                    widget.grid_remove()

        # Calidad: presupuesto de búsqueda del Hybrid A*, ajeno a la red.
        for widget in hijos:
            if (valor(widget, "variable") == str(self.calidad)
                    or valor(widget, "textvariable") == str(self.calidad_txt)):
                widget.grid_remove()

        # Solo controlaban la creación de un mapa aleatorio. Los modelos y los
        # escenarios de evaluación usan el mapa fijo de entrenamiento.
        for widget in hijos:
            if valor(widget, "text") == "Densidad obstáculos (0-1):":
                ocultar_filas(fila(widget))
            elif valor(widget, "text") == "Nuevo mapa aleatorio":
                widget.grid_remove()

        # El número de órdenes solo limita la búsqueda del planificador clásico.
        for widget in hijos:
            if valor(widget, "text").startswith("Órdenes a explorar"):
                ocultar_filas(fila(widget))
                break

        # Esta ventana ejecuta siempre la IA; el selector de motor sobra.
        cab_motor = next((w for w in hijos
                          if valor(w, "text") == "Motor de control"), None)
        radios_motor = [w for w in hijos
                        if valor(w, "variable") == str(self.motor)]
        if cab_motor is not None and radios_motor:
            inicio = fila(cab_motor) - 1
            fin = max(fila(w) for w in radios_motor) + 1
            ocultar_filas(*range(inicio, fin + 1))

        # El visor de CSV reproduce rutas del planificador, no salidas de la red.
        cab_rutas = next((w for w in hijos
                          if valor(w, "text") == "Rutas guardadas"), None)
        boton_rutas = next((w for w in hijos
                            if valor(w, "text").startswith("Ver rutas guardadas")),
                           None)
        if cab_rutas is not None and boton_rutas is not None:
            ocultar_filas(*range(fila(cab_rutas) - 1,
                                 fila(boton_rutas) + 1))

        # Sí llega a la red como one-hot; se aclara para no confundirlo con una
        # búsqueda de órdenes realizada por la propia interfaz.
        for widget in hijos:
            if valor(widget, "text") == "Optimización de flota":
                widget.configure(text="Contexto de flota (entrada de la IA)")
            elif valor(widget, "text").startswith("Global (explora órdenes"):
                widget.configure(text="Global")
            elif valor(widget, "text").startswith("Secuencial (grupo+prioridad"):
                widget.configure(text="Secuencial")

    # --------------------------- menú de nube --------------------------- #
    def _crear_menu_nube(self):
        tk = self.tk
        menu = tk.Menu(self.root)

        m_modelo = tk.Menu(menu, tearoff=False)
        m_modelo.add_command(label="Cambiar modelo…",
                             command=self._seguro(self.elegir_modelo))
        m_modelo.add_command(label="Información del modelo",
                             command=self._seguro(self.info_modelo))
        menu.add_cascade(label="Modelo", menu=m_modelo)

        m_esc = tk.Menu(menu, tearoff=False)
        m_esc.add_command(label="Escenario anterior",
                          command=self._seguro(self.escenario_anterior))
        m_esc.add_command(label="Escenario siguiente",
                          command=self._seguro(self.escenario_siguiente))
        m_esc.add_command(label="Elegir número…",
                          command=self._seguro(self.elegir_escenario))
        menu.add_cascade(label="Escenarios de selección", menu=m_esc)

        m_control = tk.Menu(menu, tearoff=False)
        for n in (10, 5, 2, 1):
            m_control.add_radiobutton(
                label="Recalcular cada %d paso%s" % (n, "" if n == 1 else "s"),
                variable=self.refresco, value=n)
        menu.add_cascade(label="Frecuencia de control", menu=m_control)
        self.root.configure(menu=menu)

    def _actualizar_titulo(self):
        nombre = os.path.basename(self.modelo_pt) if self.modelo_pt else "sin modelo"
        self.root.title("Modelo supervisado · %s" % nombre)

    def elegir_modelo(self):
        from tkinter import filedialog
        inicial = (os.path.dirname(self.modelo_pt) if self.modelo_pt
                   else MODELOS_DIR)
        ruta = filedialog.askopenfilename(
            title="Modelo PyTorch entrenado", initialdir=inicial,
            filetypes=(("Modelo PyTorch", "*.pt"), ("Todos", "*.*")))
        if ruta:
            self.modelo_pt = os.path.abspath(ruta)
            self._actualizar_titulo()
            self.estado.set("Modelo seleccionado: %s" % self.modelo_pt)

    def info_modelo(self):
        from tkinter import messagebox
        if not self.modelo_pt or not os.path.isfile(self.modelo_pt):
            messagebox.showinfo("Modelo", "No hay ningún modelo seleccionado.")
            return
        import torch
        punto = torch.load(self.modelo_pt, map_location="cpu", weights_only=False)
        cfg = punto.get("config", {})
        claves = ("n_capas", "oculto", "dropout", "n_vecinos", "h_pasado",
                  "horizonte", "n_fourier", "n_rayos", "nota_rollout",
                  "nota_test")
        datos = ["%s: %s" % (k, cfg[k]) for k in claves if k in cfg]
        messagebox.showinfo("Modelo", "%s\n\n%s" %
                            (self.modelo_pt, "\n".join(datos)))

    # ------------------------ escenarios guardados ---------------------- #
    def cargar_escenario(self, indice):
        from tkinter import messagebox
        if not self._escenarios:
            messagebox.showinfo("Escenarios", "No hay escenarios de selección.")
            return
        indice %= len(self._escenarios)
        flotas, opts = esc.flotas([self._escenarios[indice]])
        if not flotas:
            messagebox.showerror("Escenario", "El escenario no es válido.")
            return
        self._detener()
        self._indice_escenario = indice
        self.vehiculos = flotas[0]
        self.frames = []
        self.frame = 0
        self._tocando = None
        self.modo.set("manual")
        self.opt.set(opts[0])
        self.texto.delete("1.0", "end")
        self.texto.insert("1.0",
                          especs_a_texto(self._escenarios[indice]["especs"]))
        self._actualizar_combos()
        self._dibujar_estatico()
        self.estado.set("Selección %d/%d · %d vehículos · modo %s. "
                        "Pulsa «Calcular y simular»."
                        % (indice + 1, len(self._escenarios),
                           len(self.vehiculos), opts[0]))

    def escenario_anterior(self):
        self.cargar_escenario(self._indice_escenario - 1)

    def escenario_siguiente(self):
        self.cargar_escenario(self._indice_escenario + 1)

    def elegir_escenario(self):
        from tkinter import simpledialog
        if not self._escenarios:
            return
        n = simpledialog.askinteger(
            "Escenario de selección", "Número:",
            initialvalue=self._indice_escenario + 1,
            minvalue=1, maxvalue=len(self._escenarios))
        if n is not None:
            self.cargar_escenario(n - 1)

    # --------------------- rollout exacto de selección ------------------ #
    def _calcular_ia(self):
        from tkinter import messagebox
        if not self.modelo_pt or not os.path.isfile(self.modelo_pt):
            messagebox.showinfo("Sin modelo", "Selecciona primero un fichero .pt")
            return
        self._detener()
        self._ocupado = True
        try:
            from politica import Politica
            self.estado.set("Cargando el modelo y simulando…")
            self.root.update()
            politica = Politica.cargar(self.modelo_pt)
            obstaculos = (self.env.np_c, self.env.np_bb)
            mundo = (W, H)
            llegados = vec.rollout(
                [self.vehiculos], politica, opts=[self.opt.get()],
                guardar_traj=True, obstaculos=obstaculos, mundo=mundo,
                refresco=self.refresco.get())
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Error en el modo IA",
                                 "%s: %s" % (type(e).__name__, e))
            return
        finally:
            self._ocupado = False

        self.frames = construir_frames(self.vehiculos)
        self._tocando = self._auditar_frames(obstaculos, mundo)
        self.ruta_progresiva = True
        self._dibujar_estatico()
        n_veh = len(self.vehiculos)
        choques = sum(bool(getattr(v, "choque", False)) for v in self.vehiculos)
        nota = sum(_puntuar_vehiculo(v) for v in self.vehiculos) / max(1, n_veh)
        self.estado.set(
            "IA · refresco %d: llegan %d/%d · chocan %d/%d · nota %.4f. "
            "Reproduciendo…" % (self.refresco.get(), llegados[0], n_veh,
                                 choques, n_veh, nota))
        self.frame = 0
        self.reproduciendo = True
        self._anim()

    def _auditar_frames(self, obstaculos, mundo):
        """Matriz (fotograma, vehículo) para colorear cada contacto en rojo."""
        M, T = len(self.vehiculos), len(self.frames)
        tocando = np.zeros((T, M), dtype=bool)
        if not M or not T:
            return tocando
        par = {
            "largo": np.asarray([v.length for v in self.vehiculos]),
            "ancho": np.asarray([v.width for v in self.vehiculos]),
        }
        if M > 1:
            otros = np.asarray([[j for j in range(M) if j != i]
                                for i in range(M)], dtype=np.int64)
        else:
            otros = np.zeros((1, 0), dtype=np.int64)
        valido = np.ones(otros.shape, dtype=bool)
        est = np.zeros((M, 4), dtype=np.float64)
        for t, frame in enumerate(self.frames):
            for i, veh in enumerate(self.vehiculos):
                est[i, :3] = frame[veh.idx]
            tocando[t] = (
                vec.choques_entre_vehiculos(est, par, otros, valido)
                | vec.choques_con_mapa(est, par, obstaculos, mundo))
        return tocando

    def _dibujar_frame(self):
        self._fondo()
        self._rutas()
        fr = self.frames[self.frame]
        n_toc = 0
        for veh in self.vehiculos:
            x, y, th = fr[veh.idx]
            choca = bool(self._tocando[self.frame, veh.idx]) \
                if self._tocando is not None else False
            n_toc += int(choca)
            color = PALETA[veh.idx % len(PALETA)]
            self._dibujar_coche(x, y, th, veh.length, veh.width,
                                color, veh.vid)
            if choca:
                # El relleno conserva el color asociado a su ruta. El choque se
                # superpone para no hacer parecer que el vehículo cambió de ruta.
                puntos = self._poly_px(obb_corners(
                    x, y, th, veh.length, veh.width))
                self.canvas.create_polygon(
                    puntos, fill="", outline="#ff1744", width=4)
                px, py = x * SCALE, y * SCALE
                self.canvas.create_text(
                    px, py - veh.width * SCALE,
                    text="✕", fill="#ff1744", font=("", 16, "bold"))
        aviso = " · TOCANDO: %d" % n_toc if n_toc else ""
        self.canvas.create_text(
            10, 12, anchor="w",
            text="t = %.1f s (%d/%d)%s"
                 % (self.frame * DT, self.frame, len(self.frames) - 1, aviso),
            fill="#ff0000" if n_toc else COL_BORDE,
            font=("", 11, "bold"))


def construir_parser():
    ap = argparse.ArgumentParser(
        description="Interfaz visual para los modelos de Apr Superv nube.")
    ap.add_argument("--modelo", default=None,
                    help="fichero .pt (por defecto: final o ganador de fase 6)")
    ap.add_argument("--escenario", type=int, default=1,
                    help="escenario inicial de selección, empezando en 1")
    return ap


def main():
    args = construir_parser().parse_args()
    import tkinter as tk
    root = tk.Tk()
    App(root, args.modelo, max(0, args.escenario - 1))
    root.mainloop()


if __name__ == "__main__":
    main()
