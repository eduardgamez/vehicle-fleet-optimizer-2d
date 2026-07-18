#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interfaz gráfica (tkinter) del optimizador de flotas.

Toda la lógica de planificación vive en `nucleo.py`; aquí solo están la
ventana, el dibujo y la reproducción. Además del planificador clásico
(Hybrid A* cooperativo), permite ejecutar la flota con la RED NEURONAL
entrenada (`entrenar.py` -> modelos/politica.pt): la red predice, por
vehículo, los próximos N_PRED pares (aceleración, giro) y se replanifica
en bucle cerrado sin volver a tocar el planificador.

Ejecutar:   python gui.py
"""

import math
import os
import random

from nucleo import *      # noqa: F401,F403 — constantes, Entorno, Planificador, …


class App:
    def __init__(self, root):
        import tkinter as tk
        self.tk = tk
        self.root = root
        root.title("Optimizador de flotas · espacio continuo + SAT  "
                   "[núcleo nativo Numba]")
        root.resizable(False, False)

        self.env = Entorno()
        self.planificador = Planificador(self.env)
        self.vehiculos = []
        self.frames = []
        self.frame = 0
        self.anim_id = None
        self.reproduciendo = False
        self._warmed = False

        self._ocupado = False
        self._plan_msg = ""

        self._construir_ui()
        self.env.generar(0.0)
        self._actualizar_densidad()
        self._dibujar_estatico()

    # ------------------------------ robustez ----------------------------- #
    def _seguro(self, fn):
        """Envuelve un callback para que ninguna combinación de pulsaciones pueda
        colgar la aplicación: ignora acciones reentrantes, absorbe los errores de
        Tk al cerrar y muestra cualquier excepción en un diálogo."""
        tk = self.tk
        from tkinter import messagebox
        def envuelto(*args, **kwargs):
            if self._ocupado:
                return None
            try:
                return fn(*args, **kwargs)
            except tk.TclError:
                return None
            except Exception as e:  # noqa: BLE001
                try:
                    messagebox.showerror("Error inesperado",
                                         f"{type(e).__name__}: {e}")
                except Exception:
                    pass
                return None
        return envuelto

    # -------------------------------- UI --------------------------------- #
    def _construir_ui(self):
        tk = self.tk
        from tkinter import ttk
        # Estilo para destacar ligeramente el botón principal (Calcular y simular).
        estilo = ttk.Style()
        estilo.configure("Destacado.TButton", font=("", 12, "bold"))
        cont = ttk.Frame(self.root, padding=8)
        cont.grid(row=0, column=0)

        panel = ttk.Frame(cont, padding=(0, 0, 12, 0))
        panel.grid(row=0, column=0, rowspan=2, sticky="n")

        fila = [0]

        def sep():
            ttk.Separator(panel, orient="horizontal").grid(
                row=fila[0], column=0, columnspan=2, sticky="ew", pady=6)
            fila[0] += 1

        ttk.Label(panel, text="Parámetros", font=("", 13, "bold")).grid(
            row=fila[0], column=0, columnspan=2, sticky="w", pady=(0, 4))
        fila[0] += 1
        self.e_num = self._campo(panel, fila, "Nº de vehículos:", "4")
        self.e_dens = self._campo(panel, fila, "Densidad obstáculos (0-1):", "0.0")

        self.calidad = tk.IntVar(value=3)
        self.calidad_txt = tk.StringVar()
        ttk.Label(panel, textvariable=self.calidad_txt, width=38,
                  anchor="w").grid(
            row=fila[0], column=0, columnspan=2, sticky="w", pady=(4, 0))
        fila[0] += 1
        tk.Scale(panel, from_=1, to=5, resolution=1, tickinterval=1,
                 orient="horizontal", variable=self.calidad, showvalue=False,
                 command=self._calidad_cambia).grid(
            row=fila[0], column=0, columnspan=2, sticky="ew")
        fila[0] += 1
        self._calidad_cambia()

        sep()
        ttk.Label(panel, text="Vehículos", font=("", 11, "bold")).grid(
            row=fila[0], column=0, columnspan=2, sticky="w")
        fila[0] += 1
        self.modo = tk.StringVar(value="aleatorio")
        for txt, val in (("Aleatorios (todo al azar)", "aleatorio"),
                         ("Manuales (caja de texto inferior)", "manual")):
            ttk.Radiobutton(panel, text=txt, variable=self.modo,
                            value=val).grid(row=fila[0], column=0,
                                            columnspan=2, sticky="w")
            fila[0] += 1
        ttk.Button(panel, text="Generar / cargar vehículos",
                   command=self._seguro(self.generar_posiciones)).grid(
            row=fila[0], column=0, columnspan=2, sticky="ew", pady=(4, 0))
        fila[0] += 1

        sep()
        ttk.Label(panel, text="Optimización de flota", font=("", 11, "bold")).grid(
            row=fila[0], column=0, columnspan=2, sticky="w")
        fila[0] += 1
        self.opt = tk.StringVar(value="secuencial")
        for txt, val in (("Global (explora órdenes, mejor total)", "global"),
                         ("Prioridades personalizadas", "prioridades"),
                         ("Secuencial (grupo+prioridad, sin explorar)", "secuencial")):
            ttk.Radiobutton(panel, text=txt, variable=self.opt,
                            value=val).grid(row=fila[0], column=0,
                                            columnspan=2, sticky="w")
            fila[0] += 1
        # "Órdenes a explorar (máx)" + total de combinaciones posibles ("de X").
        # La casilla se desplaza a la izquierda y a su derecha, FUERA de ella, se
        # muestra el número teórico de órdenes que el modo podría llegar a probar.
        ttk.Label(panel, text="Órdenes a explorar (máx):").grid(
            row=fila[0], column=0, sticky="w", pady=1)
        sub = ttk.Frame(panel)
        sub.grid(row=fila[0], column=1, sticky="e", pady=1)
        self.e_cand = ttk.Entry(sub, width=5)
        self.e_cand.insert(0, "12")
        self.e_cand.pack(side="left")
        self.lbl_combos = ttk.Label(sub, text="de —")
        self.lbl_combos.pack(side="left", padx=(4, 0))
        fila[0] += 1

        sep()
        ttk.Label(panel, text="Motor de control", font=("", 11, "bold")).grid(
            row=fila[0], column=0, columnspan=2, sticky="w")
        fila[0] += 1
        self.motor = tk.StringVar(value="clasico")
        for txt, val in (("Clásico (Hybrid A* cooperativo)", "clasico"),
                         ("IA (red neuronal entrenada)", "ia")):
            ttk.Radiobutton(panel, text=txt, variable=self.motor,
                            value=val).grid(row=fila[0], column=0,
                                            columnspan=2, sticky="w")
            fila[0] += 1

        sep()
        # Botón principal destacado.
        ttk.Button(panel, text="▶  Calcular y simular", style="Destacado.TButton",
                   command=self._seguro(self.calcular_y_simular)).grid(
            row=fila[0], column=0, columnspan=2, sticky="ew", pady=1)
        fila[0] += 1
        for txt, cmd in (("↺  Reproducir de nuevo", self.reproducir),
                         ("⏸  Pausar / reanudar", self.pausar),
                         ("⟲  Reiniciar", self.reiniciar)):
            ttk.Button(panel, text=txt, command=self._seguro(cmd)).grid(
                row=fila[0], column=0, columnspan=2, sticky="ew", pady=1)
            fila[0] += 1

        sep()
        ttk.Label(panel, text="Mapa", font=("", 11, "bold")).grid(
            row=fila[0], column=0, columnspan=2, sticky="w")
        fila[0] += 1
        for txt, cmd in (("Nuevo mapa aleatorio", self.nuevo_mapa),
                         ("Importar imagen como mapa…", self.importar_imagen),
                         ("Guardar mapa actual…", self.guardar_mapa_actual),
                         ("Cargar mapa guardado…", self.cargar_mapa_guardado)):
            ttk.Button(panel, text=txt, command=self._seguro(cmd)).grid(
                row=fila[0], column=0, columnspan=2, sticky="ew", pady=1)
            fila[0] += 1

        sep()
        ttk.Label(panel, text="Rutas guardadas", font=("", 11, "bold")).grid(
            row=fila[0], column=0, columnspan=2, sticky="w")
        fila[0] += 1
        ttk.Button(panel, text="Ver rutas guardadas…",
                   command=self._seguro(self.ver_rutas_guardadas)).grid(
            row=fila[0], column=0, columnspan=2, sticky="ew", pady=1)
        fila[0] += 1

        sep()
        ttk.Button(panel, text="✕  Salir", command=self.root.destroy).grid(
            row=fila[0], column=0, columnspan=2, sticky="ew", pady=(1, 0))
        fila[0] += 1

        self.canvas = tk.Canvas(cont, width=int(W * SCALE), height=int(H * SCALE),
                                bg=COL_FONDO, highlightthickness=2,
                                highlightbackground=COL_BORDE)
        self.canvas.grid(row=0, column=1)
        self.canvas.bind("<Motion>", self._on_canvas_motion)
        self.canvas.bind("<Leave>", self._on_canvas_leave)
        self._cursor_info = None

        caja = ttk.Frame(cont)
        caja.grid(row=1, column=1, sticky="nsew", pady=(6, 0))
        ttk.Label(caja, text="Vehículos manuales "
                             "(lista de diccionarios; ángulos en grados):").grid(
            row=0, column=0, sticky="w")
        self.texto = tk.Text(caja, height=8, width=int(W * SCALE / 8),
                             font=("Menlo", 10), undo=True)
        self.texto.grid(row=1, column=0, sticky="ew")
        self.texto.insert("1.0", TEXTO_EJEMPLO)

        # El total de combinaciones («de X») se recalcula al cambiar de modo de
        # optimización, al editar los vehículos y al (re)generarlos. Cambiar el
        # modo a ALEATORIO o el nº de vehículos regenera las instrucciones para
        # que las preferencias queden definidas y el total se pueda calcular.
        self._ultimo_num = self.e_num.get()
        self.opt.trace_add("write", self._actualizar_combos)
        self.modo.trace_add("write", self._modo_cambia)
        self.e_num.bind("<Return>", self._num_cambia)
        self.e_num.bind("<FocusOut>", self._num_cambia)
        self.texto.bind("<KeyRelease>", self._actualizar_combos)
        self._actualizar_combos()

        self.estado = tk.StringVar(value="Listo. Ajusta parámetros y genera vehículos.")
        ttk.Label(cont, textvariable=self.estado, relief="sunken",
                  anchor="w", padding=4, width=1).grid(
            row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))

    def _campo(self, panel, fila, etiqueta, valor):
        from tkinter import ttk
        ttk.Label(panel, text=etiqueta).grid(row=fila[0], column=0, sticky="w", pady=1)
        e = ttk.Entry(panel, width=7)
        e.insert(0, valor)
        e.grid(row=fila[0], column=1, sticky="e", pady=1)
        fila[0] += 1
        return e

    def _calidad_cambia(self, *_):
        nivel = int(round(float(self.calidad.get())))
        self.calidad_txt.set(f"Calidad de ruta (1 rápida ⟷ 5 máxima):  {nivel}")
        self.planificador.configurar_calidad(nivel)

    def _on_canvas_motion(self, ev):
        x_m = ev.x / SCALE
        y_m = ev.y / SCALE
        texto = f"x: {x_m:.1f}, y: {y_m:.1f}"
        
        tx = ev.x + 12
        ty = ev.y + 12
        if ev.x > (W * SCALE) - 90:
            tx = ev.x - 80
        if ev.y > (H * SCALE) - 30:
            ty = ev.y - 20
        
        if self._cursor_info is None:
            rect = self.canvas.create_rectangle(0, 0, 1, 1, fill="white", outline="black", tags="tooltip")
            txt = self.canvas.create_text(tx, ty, text=texto, fill="black", anchor="nw", font=("", 8), tags="tooltip")
            self._cursor_info = (rect, txt)
        else:
            rect, txt = self._cursor_info
            self.canvas.itemconfigure(txt, text=texto)
            self.canvas.coords(txt, tx, ty)
            self.canvas.tag_raise(rect)
            self.canvas.tag_raise(txt)
            
        rect, txt = self._cursor_info
        bbox = self.canvas.bbox(txt)
        if bbox:
            self.canvas.coords(rect, bbox[0]-2, bbox[1]-1, bbox[2]+2, bbox[3]+1)

    def _on_canvas_leave(self, ev):
        if self._cursor_info is not None:
            self.canvas.delete("tooltip")
            self._cursor_info = None

    def _params(self):
        from tkinter import messagebox
        try:
            n = int(self.e_num.get())
            dens = float(self.e_dens.get())
        except ValueError:
            messagebox.showerror("Error", "Los parámetros deben ser numéricos.")
            return None
        if not (1 <= n <= len(PALETA)):
            messagebox.showerror("Error", f"Nº de vehículos entre 1 y {len(PALETA)}.")
            return None
        return n, max(0.0, min(1.0, dens))

    def _max_cand(self):
        try:
            return max(1, min(120, int(self.e_cand.get())))
        except ValueError:
            return 12

    def _actualizar_combos(self, *_):
        """Recalcula el total de órdenes posibles («de X») a partir de las
        instrucciones actuales de la caja de texto y del modo de optimización
        elegido. Si el texto aún no es válido, muestra «de —»."""
        if not hasattr(self, "lbl_combos"):
            return
        try:
            especs = parsear_especificaciones(self.texto.get("1.0", "end"))
            total = total_combinaciones(especs, self.opt.get())
            self.lbl_combos.config(text=f"de {total:,}")
        except Exception:
            self.lbl_combos.config(text="de —")

    def _modo_cambia(self, *_):
        """Al pasar a ALEATORIO se regeneran las instrucciones (para que las
        preferencias queden definidas y el total sea calculable); en manual solo
        se refresca el total con lo que ya hay en el texto."""
        if self.modo.get() == "aleatorio":
            self._seguro(self.generar_posiciones)()
        else:
            self._actualizar_combos()

    def _num_cambia(self, *_):
        """Cambiar el nº de vehículos (en modo aleatorio) regenera las
        instrucciones; solo si el valor realmente cambió, para no rehacer la
        flota al mover el foco sin editar."""
        actual = self.e_num.get()
        if actual == getattr(self, "_ultimo_num", None):
            return
        self._ultimo_num = actual
        if self.modo.get() == "aleatorio":
            self._seguro(self.generar_posiciones)()
        else:
            self._actualizar_combos()

    # ------------------------------- mapa -------------------------------- #
    def _actualizar_densidad(self):
        """La densidad de obstáculos solo tiene sentido en mapas aleatorios; con un
        mapa importado de imagen se deshabilita el campo."""
        importado = self.env.rects is not None
        try:
            self.e_dens.configure(state=("disabled" if importado else "normal"))
        except self.tk.TclError:
            pass

    def _mapa_cambiado(self, msg):
        self._detener()
        self.vehiculos, self.frames = [], []
        self.frame = 0
        self._actualizar_densidad()
        self.estado.set(msg)
        self._dibujar_estatico()

    def nuevo_mapa(self):
        p = self._params()
        dens = p[1] if p else 0.0
        self.env.generar(dens)
        self._mapa_cambiado("Nuevo mapa aleatorio. Genera vehículos.")

    def importar_imagen(self):
        from tkinter import filedialog, messagebox, simpledialog
        ruta = filedialog.askopenfilename(
            title="Imagen a convertir en mapa (negro = obstáculo)",
            filetypes=[("Imágenes", "*.png *.gif *.ppm *.pgm *.jpg *.jpeg *.bmp *.tif *.tiff"),
                       ("Todos", "*.*")])
        if not ruta:
            return
        self._ocupado = True
        try:
            self.estado.set("Convirtiendo imagen en mapa…")
            self.root.update()
            polys = imagen_a_poligonos(ruta)
        except Exception as e:  # noqa: BLE001
            self._ocupado = False
            messagebox.showerror("Imagen no válida",
                                 f"No se pudo leer/convertir la imagen:\n{e}\n\n"
                                 "Sin Pillow instalado solo se admiten PNG/GIF/PPM "
                                 "(pip install pillow para el resto).")
            return
        self._ocupado = False
        if len(polys) == 0:
            messagebox.showwarning("Mapa vacío",
                                   "La imagen no contiene píxeles oscuros: el mapa "
                                   "queda completamente libre.")
        nombre = os.path.splitext(os.path.basename(ruta))[0]
        self.env.desde_poligonos(polys, nombre)
        self._mapa_cambiado(f"Mapa importado de «{nombre}» "
                            f"({len(polys)} bloques). Genera vehículos.")
        if messagebox.askyesno("Guardar mapa",
                               "¿Guardar este mapa en la carpeta «mapas/» para "
                               "poder reutilizarlo en otras ejecuciones?"):
            nom = simpledialog.askstring("Nombre del mapa",
                                         "Nombre con el que guardarlo:",
                                         initialvalue=nombre)
            if nom:
                ruta_j = guardar_mapa(nom, polys=polys)
                self.estado.set(f"Mapa guardado en {ruta_j}")

    def guardar_mapa_actual(self):
        from tkinter import messagebox, simpledialog
        if self.env.origen == "aleatorio" or (self.env.rects is None
                                              and not self.env.obstaculos):
            messagebox.showinfo("Solo mapas importados",
                                "Se guardan los mapas creados desde una imagen. "
                                "Importa una imagen primero.")
            return
        nom = simpledialog.askstring("Nombre del mapa",
                                     "Nombre con el que guardarlo:",
                                     initialvalue=self.env.origen)
        if nom:
            if self.env.rects is not None:       # mapa antiguo alineado a ejes
                ruta_j = guardar_mapa(nom, self.env.rects)
            else:                                 # cajas orientadas
                ruta_j = guardar_mapa(nom, polys=self.env.obstaculos)
            self.estado.set(f"Mapa guardado en {ruta_j}")

    def cargar_mapa_guardado(self):
        from tkinter import filedialog, messagebox
        os.makedirs(MAPAS_DIR, exist_ok=True)
        if not listar_mapas():
            messagebox.showinfo("Sin mapas guardados",
                                "Aún no hay mapas en la carpeta «mapas/». "
                                "Importa una imagen y guárdala primero.")
            return
        ruta = filedialog.askopenfilename(title="Mapa guardado",
                                          initialdir=MAPAS_DIR,
                                          filetypes=[("Mapa JSON", "*.json")])
        if not ruta:
            return
        try:
            n = cargar_mapa_en(self.env, ruta)
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Mapa corrupto", f"No se pudo cargar:\n{e}")
            return
        nombre = os.path.splitext(os.path.basename(ruta))[0]
        self._mapa_cambiado(f"Mapa «{nombre}» cargado "
                            f"({n} bloques). Genera vehículos.")

    # --------------------------- crear vehículos -------------------------- #
    def generar_posiciones(self):
        from tkinter import messagebox
        p = self._params()
        if not p:
            return
        n, _ = p
        self._detener()
        self.frames = []
        self.frame = 0
        modo = self.modo.get()

        if modo in ("manual", "personalizado"):
            try:
                especs = parsear_especificaciones(self.texto.get("1.0", "end"))
                self.vehiculos = [espec_a_vehiculo(e, i, self.env)
                                  for i, e in enumerate(especs)]
            except ValueError as e:
                messagebox.showerror("Entrada de texto no válida", str(e))
                return
            if len(self.vehiculos) > len(PALETA):
                messagebox.showerror("Demasiados vehículos",
                                     f"Máximo {len(PALETA)} vehículos.")
                self.vehiculos = []
                return
            self.estado.set(f"{len(self.vehiculos)} vehículos manuales "
                            "cargados. Pulsa «Calcular y simular».")
            self._actualizar_combos()
            self._dibujar_estatico()
            return

        # ALEATORIO: se genera una lista de diccionarios de longitud n (con TODO
        # al azar), se vuelca al cuadro de texto y se crean los vehículos a partir
        # de ese texto — el mismo camino que el modo manual. Así el usuario
        # ve y puede editar los vehículos generados.
        especs = self._aleatorios_spec(n)
        if especs is None:
            messagebox.showwarning("Sin espacio",
                "No se hallaron posiciones libres. Reduce vehículos/obstáculos.")
            return
        self.texto.delete("1.0", "end")
        self.texto.insert("1.0", especs_a_texto(especs))
        try:
            normal = parsear_especificaciones(self.texto.get("1.0", "end"))
            self.vehiculos = [espec_a_vehiculo(e, i, self.env)
                              for i, e in enumerate(normal)]
        except ValueError as e:
            messagebox.showerror("Entrada de texto no válida", str(e))
            return
        self.estado.set(f"{n} vehículos aleatorios generados en el texto "
                        "(edítalos si quieres). Pulsa «Calcular y simular».")
        self._ultimo_num = self.e_num.get()
        self._actualizar_combos()
        self._dibujar_estatico()

    def _muestrear(self, length, width, otros, sep):
        for _ in range(4000):
            x = random.uniform(length, W - length)
            y = random.uniform(length, H - length)
            th = random.uniform(0, 2 * math.pi)
            if not self.env.libre(x, y, th, length, width, margen=0.3):
                continue
            if all(math.hypot(x - ox, y - oy) >= sep for ox, oy in otros):
                return (x, y, th)
        return None

    def _angulo_llegada(self, ix, iy, mx, my, length, width):
        """Ángulo de llegada CONCRETO y totalmente al azar para la plaza (mx, my).
        None si ninguna de las orientaciones al azar cabe."""
        candidatos = [random.uniform(0, 2 * math.pi) for _ in range(25)]
        for c in candidatos:
            if self.env.libre(mx, my, c, length, width, margen=0.25):
                return c
        return None

    def _aleatorios_spec(self, n):
        """Genera n especificaciones (diccionarios en GRADOS, como los del texto)
        con TODO al azar: tamaño, dinámica, capacidad de giro, pose inicial,
        destino, ángulo de llegada, grupo y prioridad. Devuelve la lista o None si
        no encuentra sitio libre para todos."""
        especs = []
        inicios, metas = [], []
        for i in range(n):
            largo = random.uniform(0.7, 1.2)
            ancho = largo * random.uniform(0.50, 0.62)
            sep = largo * 1.6
            ini = self._muestrear(largo, ancho, inicios, sep)
            met = self._muestrear(largo, ancho, metas, sep)
            if ini is None or met is None:
                return None
            ix, iy, ith = ini
            mx, my, _ = met
            thm = self._angulo_llegada(ix, iy, mx, my, largo, ancho)
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
                "v_max": round(random.uniform(1.8, 3), 2),
                "a_max": round(random.uniform(0.6, 1), 2),
                "giro_max": round(random.uniform(28.0, 40.0), 1),
                "grupo": random.randint(1, 2),
                "prioridad": random.randint(1, 3),
            })
            inicios.append((ix, iy))
            metas.append((mx, my))          # solo (x, y): _muestrear espera pares
        return especs

    def _reubicar(self, veh):
        """Modo aleatorio: si no existe ruta, prueba con extremos nuevos."""
        sep = veh.length * 1.6
        otros_ini = [v.inicio[:2] for v in self.vehiculos if v.idx != veh.idx]
        otros_meta = [v.meta for v in self.vehiculos if v.idx != veh.idx]
        ini = self._muestrear(veh.length, veh.width, otros_ini, sep)
        met = self._muestrear(veh.length, veh.width, otros_meta, sep)
        if ini is None or met is None:
            return False
        ix, iy, ith = ini
        thm = self._angulo_llegada(ix, iy, met[0], met[1], veh.length, veh.width)
        th0 = ith
        veh.inicio = (ix, iy, th0)
        veh.meta = (met[0], met[1])
        veh.meta_th = thm
        return True

    # ----------------------- ver rutas guardadas ------------------------- #
    def ver_rutas_guardadas(self):
        """Abre un selector con los escenarios guardados en rutas/ y reproduce el
        elegido, cargando su mapa fijo para que el fondo coincida."""
        tk = self.tk
        from tkinter import ttk, messagebox
        runs = listar_runs()
        if not runs:
            messagebox.showinfo("Sin rutas guardadas",
                "No hay escenarios en la carpeta rutas/. Genera algunos con "
                "generador.py (o «Calcular y simular» en modo clásico).")
            return

        win = tk.Toplevel(self.root)
        win.title("Rutas guardadas")
        win.transient(self.root)
        ttk.Label(win, text="Selecciona un escenario y pulsa «Reproducir»:",
                  padding=8).grid(row=0, column=0, columnspan=2, sticky="w")
        marco = ttk.Frame(win, padding=(8, 0, 8, 8))
        marco.grid(row=1, column=0, columnspan=2, sticky="nsew")
        scroll = ttk.Scrollbar(marco, orient="vertical")
        lista = tk.Listbox(marco, width=64, height=16, font=("Menlo", 9),
                           yscrollcommand=scroll.set)
        scroll.config(command=lista.yview)
        lista.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        for arch, rid, nveh in runs:
            try:
                nombre = os.path.relpath(arch, RUTAS_DIR)
            except ValueError:
                nombre = os.path.basename(arch)
            lista.insert("end", f"{nveh} veh  ·  {nombre}  ·  run {rid}")
        lista.selection_set(0)

        def reproducir_sel(*_):
            sel = lista.curselection()
            if not sel:
                return
            arch, rid, _ = runs[sel[0]]
            win.destroy()
            self._reproducir_run(arch, rid)

        lista.bind("<Double-Button-1>", self._seguro(reproducir_sel))
        botones = ttk.Frame(win, padding=(8, 0, 8, 8))
        botones.grid(row=2, column=0, columnspan=2, sticky="ew")
        ttk.Button(botones, text="▶  Reproducir",
                   command=self._seguro(reproducir_sel)).pack(side="left")
        ttk.Button(botones, text="Cerrar", command=win.destroy).pack(side="right")

    def _reproducir_run(self, archivo, run_id):
        """Carga un escenario del CSV, pone su mapa de fondo y lo reproduce."""
        from tkinter import messagebox
        self._detener()
        # El mapa fijo con el que se generaron las rutas está en la RAÍZ del
        # proyecto (junto a multi_v_evo.py); se carga para que el fondo coincida.
        raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ruta_mapa = os.path.join(raiz, "mapas", "mapa_entrenamiento.json")
        if os.path.exists(ruta_mapa):
            try:
                cargar_mapa_en(self.env, ruta_mapa)
                self._actualizar_densidad()
            except Exception:
                pass                         # sin fondo correcto, pero se ve la ruta
        try:
            self.vehiculos = cargar_run(archivo, run_id)
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("No se pudo cargar la ruta",
                                 f"{type(e).__name__}: {e}")
            return
        self.frames = construir_frames(self.vehiculos)
        self._dibujar_estatico()
        ok = sum(1 for v in self.vehiculos if v.mision_ok)
        self.estado.set(f"Run {run_id}: {ok}/{len(self.vehiculos)} vehículos con "
                        f"ruta ({len(self.frames)} fotogramas). Reproduciendo…")
        self.frame = 0
        self.reproduciendo = True
        self._anim()

    # --------------------------- planificación --------------------------- #
    def calcular_y_simular(self):
        from tkinter import messagebox
        if not self.vehiculos:
            messagebox.showinfo("Faltan vehículos",
                                "Genera o carga primero todos los vehículos.")
            return
        if self.motor.get() == "ia":
            self._calcular_ia()
            return
        self._detener()
        for v in self.vehiculos:
            v.traj = []
            v.mision_ok = False
        self._ocupado = True
        try:
            self._warm()
            indices = list(range(len(self.vehiculos)))
            base = Reservas()
            trays, motivos = self._planificar_flota(
                indices, base,
                reubicable=(self.modo.get() == "aleatorio"))
            self._aplicar_resultado(indices, trays, motivos)
        finally:
            self._ocupado = False

    def _calcular_ia(self):
        """Ejecuta la flota con la red neuronal en bucle cerrado: cada vehículo
        consulta la red cada N_PRED pasos y aplica los controles predichos con
        el mismo modelo de bicicleta del planificador."""
        from tkinter import messagebox
        try:
            from politica import Politica, rollout_flota, MODELO_PT
        except ImportError as e:
            messagebox.showerror("Falta PyTorch",
                                 f"El modo IA necesita torch instalado:\n{e}")
            return
        if not os.path.exists(MODELO_PT):
            messagebox.showinfo("Sin modelo entrenado",
                                f"No existe {MODELO_PT}.\n"
                                "Entrena primero con:  python entrenar.py")
            return
        self._detener()
        self._ocupado = True
        try:
            self.estado.set("Cargando modelo y simulando con la red…")
            self.root.update()
            politica = Politica.cargar()
            llegados = rollout_flota(self.vehiculos, politica,
                                     opt=self.opt.get())
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Error en el modo IA",
                                 f"{type(e).__name__}: {e}")
            return
        finally:
            self._ocupado = False
        self.frames = construir_frames(self.vehiculos)
        self._dibujar_estatico()
        self.estado.set(f"IA: {llegados}/{len(self.vehiculos)} vehículos llegan "
                        f"a su meta ({len(self.frames)} fotogramas). "
                        "Reproduciendo…")
        self.frame = 0
        self.reproduciendo = True
        self._anim()

    def _warm(self):
        if not self._warmed:
            self.estado.set("Compilando núcleos nativos (solo la 1ª vez)…")
            self.root.update()
            warmup()
            self._warmed = True

    def _tick_plan(self, expand):
        self.estado.set(f"{self._plan_msg}  ({expand:,} nodos explorados)")
        self.root.update()

    def _planificar_flota(self, indices, reservas_base, reubicable):
        """Optimiza la flota: genera los órdenes candidatos según el modo elegido,
        evalúa cada uno con planificación cooperativa completa y se queda con el
        mejor (menos fallos; a igualdad, menor tiempo total). Los vehículos que
        sigan sin ruta en el mejor orden se rescatan al final agotando la búsqueda
        (y, si 'reubicable', probando extremos nuevos)."""
        modo_opt = self.opt.get()
        ordenes = generar_ordenes(self.vehiculos, indices, modo_opt,
                                  self._max_cand())
        pl = self.planificador
        pl.tick = self._tick_plan
        cap_prev = pl.max_exp
        n = len(indices)
        mejor = None            # (fallos, coste, trays, motivos, orden)
        try:
            explorar = len(ordenes) > 1
            # Con varios órdenes y varios núcleos, se evalúan EN PARALELO (un
            # proceso por núcleo). Con uno solo, o si el paralelo falla por
            # cualquier motivo, se usa el bucle secuencial de siempre.
            hecho_paralelo = False
            if explorar and nucleos_disponibles() >= 2 and n >= 3:
                try:
                    mejor = self._evaluar_ordenes_paralelo(
                        ordenes, reservas_base, cap_prev)
                    hecho_paralelo = True
                    if mejor is None:            # ventana cerrada durante el cálculo
                        return {}, {}
                except Exception:
                    hecho_paralelo = False       # se recae al modo secuencial

            if not hecho_paralelo:
                mejor = None
                for oi, orden in enumerate(ordenes):
                    if not self.root.winfo_exists():
                        return {}, {}
                    etiqueta = (f"Orden {oi + 1}/{len(ordenes)} "
                                f"[{'·'.join(str(self.vehiculos[i].vid) for i in orden)}]"
                                if explorar else "Planificación cooperativa")

                    def tick_veh(pos, idx, _et=etiqueta):
                        self._plan_msg = (f"{_et} · vehículo {pos + 1}/{n} "
                                          f"(id {self.vehiculos[idx].vid})")
                        self.estado.set(self._plan_msg + "…")
                        self.root.update()

                    if explorar:
                        # Presupuesto por candidato: cap de calidad + plazo POR
                        # VEHÍCULO, para que explorar muchos órdenes siga siendo
                        # interactivo y ningún vehículo se coma el plazo ajeno.
                        pl.max_exp = cap_prev
                        dur_veh = PLAZO_VEH_CAND
                    else:
                        # Orden DEFINITIVO: sin plazo de tiempo, solo nodos
                        # (cap_completo). El "cuándo se rinde" es max_exp/
                        # max_nodos y la relajación del ángulo por nodos-en-
                        # meta; nada de segundos aquí.
                        pl.max_exp = CAP_COMPLETO
                        pl.deadline = None
                        dur_veh = None
                    trays, motivos, coste = planificar_orden(
                        pl, self.vehiculos, orden, reservas_base,
                        tick_veh=tick_veh, deadline_dur=dur_veh)
                    fallos = sum(1 for i in orden if trays[i] is None)
                    if mejor is None or (fallos, coste) < (mejor[0], mejor[1]):
                        mejor = (fallos, coste, trays, motivos, orden)
                    if fallos == 0 and not explorar:
                        break

            fallos, _, trays, motivos, orden = mejor
            # ---- rescate de los vehículos sin ruta en el mejor orden ---- #
            if fallos:
                pl.max_exp = CAP_COMPLETO
                pl.deadline = None            # solo nodos, nada de segundos
                reservas = reservas_base.copia()
                for i in orden:
                    if trays[i] is not None:
                        v = self.vehiculos[i]
                        reservas.add(trays[i], v.length, v.width)
                for i in orden:
                    if trays[i] is not None:
                        continue
                    veh = self.vehiculos[i]
                    self._plan_msg = (f"Rescate del vehículo id {veh.vid} "
                                      "(búsqueda exhaustiva)")
                    self.estado.set(self._plan_msg + "…")
                    self.root.update()
                    pl.bloqueos = bloqueos_metas(
                        self.vehiculos, orden, excepto=i, pose0=veh.inicio)
                    pl.deadline = None            # solo nodos
                    traj = pl.planificar(veh, reservas)
                    motivos[i] = pl.motivo
                    intentos = 0
                    while (pl.motivo == "sin_ruta" and reubicable
                           and intentos < 12):
                        if not self._reubicar(veh):
                            break
                        pl.bloqueos = bloqueos_metas(self.vehiculos, orden,
                                                     excepto=i, pose0=veh.inicio)
                        pl.deadline = None            # solo nodos
                        traj = pl.planificar(veh, reservas)
                        motivos[i] = pl.motivo
                        intentos += 1
                    if traj is not None and len(traj) >= 2:
                        trays[i] = traj
                        motivos[i] = "ok"
                        reservas.add(traj, veh.length, veh.width)
                    elif traj:
                        trays[i] = [traj[0], traj[0]]
                        motivos[i] = "ok"
            return trays, motivos
        finally:
            pl.tick = None
            pl.deadline = None
            pl.max_exp = cap_prev

    def _evaluar_ordenes_paralelo(self, ordenes, reservas_base, max_exp):
        """Envoltorio de la evaluación POR HORNADAS (evaluar_ordenes_hornadas)
        con los 'callbacks' de la interfaz de escritorio: refresca la barra de
        estado y aborta si se cierra la ventana. Devuelve el mejor (fallos,
        coste, trays, motivos, orden) o None."""
        calidad = int(round(float(self.calidad.get())))

        def progreso(texto):
            self._plan_msg = texto
            self.estado.set(texto)
            self.root.update()

        return evaluar_ordenes_hornadas(
            self.env, self.vehiculos, ordenes, reservas_base,
            calidad, max_exp=max_exp, progreso=progreso,
            sigue_vivo=self.root.winfo_exists)

    def _aplicar_resultado(self, indices, trays, motivos):
        """Vuelca las trayectorias en los vehículos, reconstruye los fotogramas e
        informa del resultado."""
        from tkinter import messagebox
        if not self.root.winfo_exists():
            return
        ok = 0; sin_ruta = 0; inconcluso = 0
        for i in indices:
            veh = self.vehiculos[i]
            traj = trays.get(i)
            if traj is None:
                veh.mision_ok = False
                veh.traj = []
                if motivos.get(i) == "sin_ruta":
                    sin_ruta += 1
                else:
                    inconcluso += 1
                continue
            veh.mision_ok = True
            ok += 1
            veh.traj = traj
            veh.dt_plan = DT

        # Rama DEFINITIVA: vuelca el dataset de aprendizaje supervisado (estado de
        # 6 parámetros + control de 2) al CSV acumulado, descargable de /rutas.
        try:
            n_filas = guardar_dataset_supervisado(
                [self.vehiculos[i] for i in indices], opt=self.opt.get())
            if n_filas:
                print(f"[dataset] +{n_filas} muestras → {DATASET_CSV}")
        except Exception as e:      # nunca debe tumbar la ejecución de la ruta
            print(f"[dataset] no se pudo guardar: {e}")

        self._dibujar_estatico()
        if ok == 0:
            if inconcluso:
                messagebox.showwarning("Búsqueda inconclusa",
                    "No se agotó la búsqueda dentro del techo de nodos: el "
                    "resultado es INCONCLUSO (podría existir ruta, pero el "
                    "espacio de búsqueda era demasiado grande).\nSube la "
                    "«Calidad de ruta», usa menos vehículos/obstáculos o cambia "
                    "las posiciones.")
            else:
                messagebox.showwarning("Sin ruta",
                    "No existe ruta para ningún vehículo: la búsqueda agotó todo "
                    "el espacio alcanzable sin llegar a la meta.\nLas posiciones "
                    "elegidas no tienen solución en este mapa.")
            self.estado.set("Sin rutas. Ajusta parámetros y reintenta.")
            return

        self.frames = construir_frames(self.vehiculos)
        avisos = []
        if sin_ruta:
            avisos.append(f"{sin_ruta} sin ruta (no existe)")
        if inconcluso:
            avisos.append(f"{inconcluso} inconcluso (techo de búsqueda)")
        aviso = f"  ⚠ {'; '.join(avisos)}" if avisos else ""
        total = sum(v.duracion for v in self.vehiculos if v.mision_ok)
        self.estado.set(f"{ok} rutas sin colisiones · tiempo total de flota "
                        f"{total:.1f} s ({len(self.frames)} fotogramas).{aviso}  "
                        "Reproduciendo…")
        self.frame = 0
        self.reproduciendo = True
        self._anim()

    # ----------------------------- reproducción --------------------------- #
    def reproducir(self):
        if not self.frames:
            from tkinter import messagebox
            messagebox.showinfo("Nada que reproducir", "Primero calcula y simula.")
            return
        self._detener()
        self.frame = 0
        self.reproduciendo = True
        self._anim()

    def _anim(self):
        if not self.reproduciendo or not self.frames:
            return
        try:
            if not self.root.winfo_exists():
                return
            self.frame = max(0, min(self.frame, len(self.frames) - 1))
            self._dibujar_frame()
            if self.frame >= len(self.frames) - 1:
                self.reproduciendo = False
                self.estado.set("Reproducción finalizada.")
                return
            self.frame += 1
            self.anim_id = self.root.after(40, self._anim)
        except self.tk.TclError:
            self.reproduciendo = False

    def pausar(self):
        if not self.frames:
            return
        if self.reproduciendo:
            self.reproduciendo = False
            if self.anim_id:
                self.root.after_cancel(self.anim_id)
            self.estado.set("Pausado.")
        else:
            self.reproduciendo = True
            self.estado.set("Reanudando…")
            self._anim()

    def reiniciar(self):
        self._detener()
        self.frame = 0
        self.estado.set("Reiniciado al primer fotograma.")
        if self.frames:
            self._dibujar_frame()
        else:
            self._dibujar_estatico()

    def _detener(self):
        self.reproduciendo = False
        if self.anim_id:
            self.root.after_cancel(self.anim_id)
            self.anim_id = None

    # -------------------------------- dibujo ------------------------------ #
    def _poly_px(self, poly):
        out = []
        for x, y in poly:
            out.extend((x * SCALE, y * SCALE))
        return out

    def _fondo(self):
        c = self.canvas
        c.delete("all")
        for poly in self.env.obstaculos:
            c.create_polygon(self._poly_px(poly), fill=COL_OBST, outline=COL_OBST)
        for veh in self.vehiculos:
            col = PALETA[veh.idx % len(PALETA)]
            mx, my = veh.meta
            c.create_oval(mx * SCALE - 7, my * SCALE - 7,
                          mx * SCALE + 7, my * SCALE + 7, outline=col, width=2)
            c.create_text(mx * SCALE, my * SCALE, text="◎", fill=col,
                          font=("", 13, "bold"))
            if veh.meta_th is not None:
                # Flecha del ÁNGULO DE LLEGADA exigido en el destino.
                ax = mx + 0.7 * math.cos(veh.meta_th)
                ay = my + 0.7 * math.sin(veh.meta_th)
                c.create_line(mx * SCALE, my * SCALE, ax * SCALE, ay * SCALE,
                              fill=col, width=1, arrow="last",
                              arrowshape=(6, 7, 2))
            poly = obb_corners(veh.inicio[0], veh.inicio[1], veh.inicio[2],
                               veh.length, veh.width)
            c.create_polygon(self._poly_px(poly), outline=col, fill="",
                             width=1, dash=(3, 3))

    def _rutas(self):
        for veh in self.vehiculos:
            if veh.traj:
                col = PALETA[veh.idx % len(PALETA)]
                pts = []
                for x, y, _ in veh.traj:
                    pts.extend((x * SCALE, y * SCALE))
                if len(pts) >= 4:
                    self.canvas.create_line(pts, fill=col, width=1, smooth=True)

    def _dibujar_estatico(self):
        self._fondo()
        for veh in self.vehiculos:
            self._dibujar_coche(veh.inicio[0], veh.inicio[1], veh.inicio[2],
                                veh.length, veh.width,
                                PALETA[veh.idx % len(PALETA)], veh.vid)

    def _dibujar_frame(self):
        self._fondo()
        self._rutas()
        fr = self.frames[self.frame]
        for veh in self.vehiculos:
            x, y, th = fr[veh.idx]
            self._dibujar_coche(x, y, th, veh.length, veh.width,
                                PALETA[veh.idx % len(PALETA)], veh.vid)
        self.canvas.create_text(10, 12, anchor="w",
                                text=f"t = {self.frame * DT:.1f} s   "
                                     f"({self.frame}/{len(self.frames) - 1})",
                                fill=COL_BORDE, font=("", 11, "bold"))

    def _dibujar_coche(self, x, y, th, length, width, col, num):
        c = self.canvas
        c.create_polygon(self._poly_px(obb_corners(x, y, th, length, width)),
                         fill=col, outline="#101010", width=1)
        nx = x + math.cos(th) * length * 0.35
        ny = y + math.sin(th) * length * 0.35
        c.create_line(x * SCALE, y * SCALE, nx * SCALE, ny * SCALE,
                      fill="white", width=2)
        c.create_text(x * SCALE, y * SCALE, text=str(num), fill="white",
                      font=("", 10, "bold"))


def main():
    import tkinter as tk
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
