# Optimizador de flotas 2D [WIP]

Simulador de una flota de vehículos que deben ir cada uno de un punto a otro de
un mapa con obstáculos, **sin chocar entre ellos ni con obstáculos**, con física
realista. El programa calcula las rutas, las coordina en el tiempo y
las puede reproducir.

- **Prioridades:** se pueden asignar prioridades por vehículo y por grupos.
- **Personalización total:** es posible definir todas las características de cada vehículo, incluyendo el **ángulo de llegada** a la meta, y la velocidad inicial para posible encadenamiento de rutas. 
- **Mapas personalizados:** admite importar imágenes que se transforman en mapas de obstáculos, así como guardar y cargar estos entornos.
- **Control calidad-tiempo:** se puede ajustar el equilibrio entre la calidad de las rutas y el tiempo de cálculo del 1 al 5, modificando, entre otros factores, la cantidad de movimientos probados.

## Modos de ejecución

El proyecto ofrece dos formas principales de ejecutarse:

### 1. Aplicación de escritorio (Tkinter)

Ejecuta el simulador completo directamente en una ventana nativa de escritorio:

```bash
python3 multi_v_evo.py
```

### 2. Servidor e interfaz web

Para ejecutar el panel web en el navegador, inicia el servidor Flask:

```bash
python3 server.py
```

Abre tu navegador en `http://localhost:5000` para acceder a la interfaz web.

---

## Estructura del proyecto y descripción de archivos

El repositorio se compone de los siguientes archivos y carpetas:

### Núcleo de simulación
- **`multi_v_evo.py`** — Simulador principal e interfaz de Tkinter. Contiene todo el motor: física de bicicleta, planificación **Hybrid A\*** cooperativa con coordinación espacio-tiempo, colisiones exactas (OBB/SAT), optimización de flota en paralelo, compilación JIT con **Numba**...
- **`multi_vehiculo.py`** — Versión reducida y ligera del simulador de escritorio. Contiene el motor de física y planificación Hybrid A*, pero omite funciones avanzadas como importar mapas desde imágenes, modos de optimización por prioridades o ángulos de llegada personalizados.

### Servidor e interfaz web (`web/`)
- **`server.py`** — Servidor web basado en Flask y Gunicorn. Expone el motor de `multi_v_evo.py` mediante una API REST asíncrona (`/api/simular`, `/api/*/estado`, `/api/misiones`, etc.). Ejecuta las planificaciones en hilos de fondo con control de concurrencia y permite al navegador consultar el progreso en tiempo real.
- **`web/index.html`** — Incluye la estructura de la interfaz diseñada con más detalle, con el panel lateral de parámetros, controles segmentados para selección de modos, deslizador de calidad de ruta, controles de ejecución y el mapa (`<canvas>`) para visualizar la animación.
- **`web/script.js`** — Lógica del cliente web. Gestiona el renderizado y animación de la flota en el canvas HTML5 a 60 FPS, la comunicación asíncrona con el servidor, el seguimiento de la barra de progreso y la edición de vehículos en formato JSON.
- **`web/styles.css`** — Estilos visuales del panel web. Proporciona una interfaz limpia y responsiva con variables CSS, controles adaptados tanto para pantallas de escritorio como para dispositivos móviles.

### Mapas y datos
- **`mapas/`** — Directorio donde se almacenan y cargan los mapas de obstáculos exportados o importados desde imágenes en formato JSON.

### Configuración y despliegue
- **`requirements.txt`** — Lista de librerías de Python requeridas (`flask`, `numpy`, `numba`, `pillow`, `gunicorn`).
- **`render.yaml`** — Configuración de despliegue como servicio web en Render, configurando workers de Gunicorn, hilos de trabajo y la caché de compilación de Numba.
- **`runtime.txt`** — Especifica la versión del intérprete de Python (`python-3.12.4`) para entornos de despliegue en la nube.
- **`LICENSE`** — Archivo con la licencia de distribución del proyecto.

---

## Cómo funciona por dentro (resumen)

- Cada vehículo se mueve con el **modelo de bicicleta**: no puede girar sobre
  sí mismo, tiene un radio de giro mínimo, y su velocidad y aceleración están
  acotadas.
- Las rutas se buscan con **Hybrid A\***: una búsqueda que explora maniobras
  físicamente posibles (acelerar/frenar + girar más o menos) hasta llegar al
  destino, guiada por un campo de distancias que conoce los obstáculos.
  Hace crecer caminos parciales por turnos: en cada turno
  amplía un paso el **más prometedor** —el de menor **coste recorrido + peso ×
  estimación hasta la meta**— y sus ramas nuevas quedan en espera junto a las
  anteriores; el siguiente turno vuelve a elegir el mejor de todas (puede ser
  una aparcada hace rato). Así no se compromete con ningún camino hasta que uno
  llega a la meta.
- La coordinación es **espacio-tiempo**: cuando un vehículo ya tiene ruta, los
  demás la ven como un obstáculo móvil y la esquivan (o esperan).
- La búsqueda del mejor orden se **reparte entre todos los núcleos**: en los modos correspondientes, los órdenes
  candidatos se evalúan a la vez en **hornadas** de tantos como núcleos. Cada
  hornada se corta en cuanto ha llegado el **~95 %** de los vehículos (contando
  los de hornadas previas), descartando a los rezagados en lugar de esperar a los
  casos difíciles. 
- Las colisiones se comprueban con rectángulos orientados reales (SAT).
- El núcleo numérico se compila a **código nativo con Numba** al arrancar, así
  que el cálculo es rápido (la primera ejecución tarda un poco más por la
  compilación; luego queda en caché).

## Movimientos probados por calidad

En cada nodo la búsqueda prueba **3 aceleraciones × N ángulos de volante**:

- **Aceleraciones (siempre 3):** acelerar a tope, ni acelerar ni frenar, y frenar
  a tope.
- **Ángulos de volante (N según calidad del 1 al 5):** repartidos entre el giro máximo a
  izquierda y a derecha del vehículo, más densos cerca del recto (que siempre se
  incluye).

El **ángulo de poda** descarta caminos redundantes: la búsqueda agrupa los
estados en casillas (posición, orientación, velocidad e instante) y, de los que
caen en la misma, conserva solo el mejor. Ese ángulo es el ancho de la casilla en
orientación; más fino conserva más variantes. 

| Calidad | Ángulos de volante (N) | Movimientos totales (×3) | Ángulo de poda |
|:---:|:---:|:---:|:---:|
| 1 | 17 | 51 | 12° |
| 2 | 23 | 69 | 11° |
| 3 | 31 | 93 | 10° |
| 4 | 41 | 123 | 8° |
| 5 | 55 | 165 | 6° |

## Los tres modos de priorización

Los vehículos deben evitar a los que han planificado ruta antes para evitar colisión, por lo que el primero elige ruta a
sus anchas y los demás se van adaptando. El panel «Optimización de flota»
permite elegir cómo se decide ese orden:

- **Global:** el programa prueba muchos órdenes distintos (tantos como se indique en la entrada de texto), los evalúa en paralelo por hornadas como se ha explicado arriba, y se queda con la
  solución con más vehículos habiendo llegado al objetivo. Luego, a igualdad, escoje el que haya resultado en menor tiempo total para hacer llegar a los vehículos. 
- **Prioridades personalizadas:** cada vehículo lleva dos números en la
  entrada de texto: su `grupo` de prioridad y su `prioridad` dentro del grupo.
  - Los **grupos** se priorizan en orden ascendente (grupo 1 antes que grupo 2,
    etc.): un grupo entero se planifica antes de pasar al siguiente.
  - Dentro de un grupo, la **prioridad** ordena los vehículos (menor primero).
    Los que comparten el mismo número de prioridad se optimizan
    globalmente entre sí. 

- **Secuencial:** un único orden probado, sin explorar combinaciones: se
  planifica primero grupos por prioridad y, dentro de
  cada grupo, según la prioridad individual. Es el modo más rápido y
  predecible.

En los modos **global** y **prioridades**, el orden ganador puede tener vehículos sin ruta por falta de tiempo. Si es el caso, se planifica la ruta hasta el final sin
límite de tiempo (solo con el tope de nodos) para calcular su ruta completa.

## Los dos modos de vehículos

- **Aleatorios:** se genera una **lista de diccionarios** con características al azar
  (tamaño, velocidad, aceleración, capacidad de giro, punto inicial, orientación
  inicial, destino, ángulo de llegada, grupo y prioridad), se **vuelca en la caja de texto** y a
  partir de ahí se crean los vehículos. Como queda escrita, puedes revisarla y
  editar cualquier valor antes de simular. 
- **Manuales:** se definen en la caja de texto inferior como una
  **lista de diccionarios**, uno por vehículo (ángulos en grados). Cada vehículo
  admite identificador, punto inicial, orientación inicial, destino, ángulo de
  llegada, tamaño, velocidad máxima, velocidad inicial, aceleración, capacidad de
  giro, grupo y prioridad. Todas las claves salvo `inicio` y `meta` son
  opcionales y tienen valores por defecto razonables.

## Mapas desde una imagen

Además del mapa aleatorio tipo ciudad, cualquier imagen puede convertirse en
mapa: los píxeles cercanos al negro se
vuelven obstáculo y los cercanos al blanco, espacio libre(umbral en el
punto medio de brillo). La proporción de la imagen se ajusta al mundo de 5:3.

La imagen de entrada puede tener cualquier forma, curvas incluidas, pero el
programa reconstruye cada región de obstáculo como un polígono, siempre de lados
rectos: muestrea la imagen en una rejilla fina, traza su contorno, lo simplifica
(Douglas-Peucker) hasta quedarse solo con los vértices que definen la forma —una
pared diagonal pasa de escalera a línea limpia— y rellena el interior con
triángulos (ear clipping) para que quede macizo. Entonces una curva se aproxima por una sucesión de tramos rectos que son sus líneas de
colisión.

Al importar, el programa ofrece guardar el mapa en la carpeta `mapas/`
(ficheros JSON pequeños). El mismo mapa puede reutilizarse en cualquier otra ejecución del
programa. Se puede guardar en cualquier momento el mapa
importado que esté en pantalla.

## Controles restantes

- **Calidad de ruta (1–5):** equilibrio entre velocidad de cálculo y calidad
  de las rutas (más calidad = rutas más cortas y directas, más tiempo de
  cálculo).
- **Densidad de obstáculos:** obstáculos extra en el mapa aleatorio. Se
  deshabilita automáticamente cuando el mapa proviene de una imagen (ahí no
  tiene sentido).

## Relajación del ángulo de llegada

La tolerancia del ángulo de llegada del vehículo se ensancha de forma lineal
según los nodos explorados en la zona de meta. Parte de **±3°**; la etapa 1 la
abre hasta el tope del nivel y, si no basta, la etapa 2 sigue —al doble de
ritmo— hasta el ángulo libre (**±180°**), salvo en la calidad 5, que mantiene ese mismo
ritmo pero se detiene en **±90°**.

| Calidad | Llegada exacta (±3°) | Etapa 1: ±3° → tope | Etapa 2: tope → ángulo libre |
|:---:|:---:|:---:|:---:|
| 1 | 0 – 84 000 | 84 000 – 216 000 (hasta ±26°) | 216 000 – 658 000 |
| 2 | 0 – 98 000 | 98 000 – 252 000 (hasta ±21°) | 252 000 – 932 000 |
| 3 | 0 – 112 000 | 112 000 – 288 000 (hasta ±17°) | 288 000 – 1 313 000 |
| 4 | 0 – 140 000 | 140 000 – 361 000 (hasta ±13°) | 361 000 – 2 206 000 |
| 5 | 0 – 350 000 | 350 000 – 700 000 (hasta ±7°) | 700 000 – 4 454 000 (hasta ±90°) |

## Peso del heurístico por calidad

El **peso** de la fórmula de la búsqueda (**coste recorrido + peso × estimación
hasta la meta**) es ajustable. Con peso **1** también se dan turnos a caminos
laterales y se encuentra el más corto; con peso **> 1** casi todos los turnos van
al que apunta directo a la meta: se llega antes y con menos nodos, pero quizá no
por el más corto. Cada calidad fija ese peso:

| Calidad | 1 | 2 | 3 | 4 | 5 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| Peso | 1,9 | 1,6 | 1,3 | 1,1 | 1,0 |

A más calidad, peso más cercano a 1: rutas más óptimas y más cálculo. A menos
calidad, peso mayor: más rápido y menos óptimo.

## Instalación

Para instalar todas las librerías necesarias tanto para la versión de escritorio como para la interfaz web y el servidor (`numpy`, `numba`, `pillow`, `flask`, `gunicorn`):

```bash
pip install -r requirements.txt
```

Si únicamente deseas ejecutar el simulador local en escritorio, puedes instalar solo las librerías base:

```bash
pip install numpy numba pillow
```

En Linux además hace falta tkinter para la versión de escritorio:

```bash
sudo apt install python3-tk      # Debian/Ubuntu
sudo dnf install python3-tkinter # Fedora
sudo pacman -S tk                # Arch
```
