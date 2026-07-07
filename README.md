# Optimizador de flotas 2D [WIP]

Simulador de una flota de vehículos que deben ir cada uno de un punto a otro de
un mapa con obstáculos, **sin chocar entre ellos ni con obstáculos**, con física
realista. El programa calcula las rutas, las coordina en el tiempo y
las puede reproducir.

- **Prioridades avanzadas:** se pueden asignar prioridades por vehículo y por grupos.
- **Personalización total:** es posible definir todas las características de cada vehículo, incluyendo incluso el **ángulo de llegada** a la meta.
- **Misiones encadenadas:** permite reutilizar vehículos y asignarles nuevos objetivos una vez han alcanzado su destino anterior.
- **Mapas personalizados:** admite importar imágenes como mapas de obstáculos, así como guardar y cargar entornos propios.
- **Control calidad-tiempo:** se puede ajustar el equilibrio entre la calidad de las rutas y el tiempo de cálculo.

Ejecutar:

```bash
python3 multi_v_evo.py
```

El archivo `multi_vehiculo.py` incluido es una versión reducida. No tiene algunas características avanzadas como importar mapas desde imágenes, modos de optimización por prioridades, ángulos de llegada específicos y reasignación de destinos.

## Cómo funciona por dentro (resumen)

- Cada vehículo se mueve con el **modelo de bicicleta**: no puede girar sobre
  sí mismo, tiene un radio de giro mínimo, y su velocidad y aceleración están
  acotadas.
- Las rutas se buscan con **Hybrid A\***: una búsqueda que explora maniobras
  físicamente posibles (acelerar/frenar + girar más o menos) hasta llegar al
  destino, guiada por un campo de distancias que ya conoce los obstáculos.
- La coordinación es **espacio-tiempo**: cuando un vehículo ya tiene ruta, los
  demás la ven como un obstáculo móvil y la esquivan (o esperan).
- Las colisiones se comprueban con rectángulos orientados reales (SAT), no con
  círculos aproximados.
- El núcleo numérico se compila a **código nativo con Numba** al arrancar, así
  que el cálculo es rápido (la primera ejecución tarda un poco más por la
  compilación; luego queda en caché).

## Los tres modos de priorización

El orden en que se planifican los vehículos importa: el primero elige ruta a
sus anchas y los demás se van adaptando. El panel «Optimización de flota»
permite elegir cómo se decide ese orden:

- **Global** — el programa prueba muchos órdenes distintos (todas las
  permutaciones si hay pocos vehículos; órdenes heurísticos y barajados si hay
  muchos) y se queda con la solución con **menos fallos y menor tiempo total de
  la flota**. El campo «Órdenes a explorar» limita cuántos candidatos prueba.
- **Prioridades personalizadas** — cada vehículo lleva dos números en la
  entrada de texto: su `grupo` de prioridad y su `prioridad` dentro del grupo.
  - Los **grupos** se resuelven en orden ascendente (grupo 1 antes que grupo 2,
    etc.): un grupo entero se planifica antes de pasar al siguiente.
  - Dentro de un grupo, la **prioridad** ordena los vehículos (menor primero).
    Los que comparten el **mismo** número de prioridad se optimizan
    **globalmente** entre sí (sin orden impuesto). El número es local al grupo,
    así que puede repetirse en grupos distintos.

  Así se cubren los dos casos típicos: «este grupo de coches tiene más
  prioridad pero es igual entre ellos» (mismo grupo, misma prioridad) y «estos
  coches van con prioridad máxima y en este orden» (mismo grupo, prioridades
  1, 2, 3…).
- **Secuencial** — un único orden determinista, sin explorar combinaciones: se
  planifica primero el grupo prioritario (menor número de `grupo`) y, dentro de
  cada grupo y prioridad, según la posición en la lista. Es el modo más rápido y
  predecible.

En cualquier modo, si algún vehículo se queda sin ruta, al final se le hace un
«rescate» con búsqueda exhaustiva.

## Los dos modos de vehículos

- **Aleatorios** — se genera una **lista de N diccionarios** con todo al azar
  (tamaño, velocidad, aceleración, capacidad de giro, punto inicial, destino,
  ángulo de llegada, grupo y prioridad), se **vuelca en la caja de texto** y a
  partir de ahí se crean los vehículos. Como queda escrita, puedes revisarla y
  editar cualquier valor antes de simular. N es el «Nº de vehículos».
- **Manuales** — se definen en la caja de texto inferior como una
  **lista de diccionarios**, uno por vehículo (ángulos en grados):

  ```python
  [
   {"id": 1, "inicio": (3, 3),  "giro_inicial": 0,  "meta": (36, 20),
    "angulo_llegada": 90, "largo": 1.6, "ancho": 0.8, "v_max": 3.0,
    "a_max": 1.2, "giro_max": 33, "grupo": 1, "prioridad": 1},
   {"id": 2, "inicio": (36, 3), "meta": (4, 20), "grupo": 2, "prioridad": 1},
  ]
  ```
  
  Todas las claves salvo `inicio` y `meta` son opcionales y tienen valores por
  defecto razonables.

## Ángulo de llegada

No basta con llegar al destino: el vehículo debe llegar **orientado con un
ángulo concreto** (como aparcar mirando hacia una dirección). En el mapa se
dibuja una flecha en cada destino con el ángulo exigido. En modo aleatorio el
ángulo se sortea; en modo manual lo fija `angulo_llegada` (y si se
omite, se usa la dirección natural inicio→meta). Con `"angulo_llegada":
"libre"` la orientación final queda libre.

## Reutilizar vehículos (misiones nuevas)

Cuando la simulación termina y los coches están aparcados, se les puede dar
otra misión **sin reiniciar nada**:

1. Borra el contenido de la caja de texto.
2. Escribe una lista de diccionarios **solo con los vehículos que quieras
   mover**, identificados por su `id`, con su nueva `meta` (y, si quieres,
   `angulo_llegada`, `prioridad`, `v_max`, `a_max` o `giro_max` nuevos):

   ```python
   [{"id": 2, "meta": (35, 4), "angulo_llegada": -90}]
   ```

3. Pulsa **«⟳ Nuevas rutas para ids del texto»**.

Cada vehículo mencionado arranca desde la plaza donde quedó aparcado; los no
mencionados no se mueven y se respetan como obstáculos. Se puede repetir tantas
veces como se quiera.

## Mapas desde una imagen

Además del mapa aleatorio tipo ciudad, cualquier imagen puede convertirse en
mapa con **«Importar imagen como mapa…»**: los píxeles **cercanos al negro se
vuelven obstáculo** y los **cercanos al blanco, espacio libre** (umbral en el
punto medio de brillo). La imagen se ajusta al mundo de 40 × 24 m.

Al importar, el programa ofrece **guardar el mapa** en la carpeta `mapas/`
(ficheros JSON pequeños); esa carpeta se crea sola. Con **«Cargar mapa
guardado…»** el mismo mapa puede reutilizarse en cualquier otra ejecución del
programa. **«Guardar mapa actual…»** guarda en cualquier momento el mapa
importado que esté en pantalla.

## Controles restantes

- **Calidad de ruta (1–5)** — equilibrio entre velocidad de cálculo y calidad
  de las rutas (más calidad = rutas más cortas y directas, más tiempo de
  cálculo).
- **Densidad de obstáculos** — obstáculos extra en el mapa aleatorio. Se
  deshabilita automáticamente cuando el mapa proviene de una imagen (ahí no
  tiene sentido).
- **Velocidad de reproducción**, **Reproducir**, **Pausar/Reanudar**,
  **Reiniciar** — controlan la animación.

## Instalación

Librerías necesarias: `numpy`, `numba` (y `pillow` opcional, para importar
imágenes JPG/BMP como mapas; sin él se admiten PNG/GIF/PPM).

```bash
pip install numpy numba pillow
```

En Linux además hace falta tkinter:

```bash
sudo apt install python3-tk      # Debian/Ubuntu
sudo dnf install python3-tkinter # Fedora
sudo pacman -S tk                # Arch
```
