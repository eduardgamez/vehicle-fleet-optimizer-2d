# Optimizador de flotas 2D [WIP]

Este proyecto estudia distintas maneras de resolver un mismo problema: coordinar
vehículos autónomos para que todos alcancen su destino con rapidez, respetando
su física y sin chocar. No parte de una única técnica. La intención es construir,
medir y comparar tres enfoques: planificación clásica, aprendizaje supervisado y
aprendizaje por refuerzo.

Cada etapa aprovecha la anterior. El planificador clásico sirve como solución
directa y también como profesor de la red supervisada. Después, el aprendizaje
por refuerzo buscará superar las limitaciones de imitar rutas ya calculadas.

# Parte I — Planificación clásica

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

# Parte II — Aprendizaje supervisado

El segundo enfoque sustituye la búsqueda de una ruta por una red neuronal. El
planificador clásico resuelve muchos escenarios y deja ejemplos de cómo actuar;
la red aprende a reproducir esos controles y después conduce por sí sola, en
bucle cerrado. La meta es obtener decisiones mucho más rápidas que una búsqueda
completa, aunque aprender a imitar una ruta no garantiza por sí solo que el
vehículo se recupere bien de sus propios errores.

## Primer planteamiento: generación y entrenamiento local

El primer pipeline está en `Apr Superv local/`. Usa un mapa fijo de entrenamiento
y sigue este proceso:

1. `generador.py` crea escenarios aleatorios con distintos vehículos, destinos,
   tamaños, velocidades, prioridades y modos de coordinación.
2. El planificador clásico resuelve cada flota. No se descarta un escenario por
   tardar: se busca hasta alcanzar los límites propios del algoritmo.
3. Cada trayectoria se guarda en CSV. En cada instante quedan registrados el
   estado del vehículo, su entorno y el control que aplicó el planificador.
4. `entrenar.py` convierte las trayectorias en pares de entrada y salida. La
   validación se separa por escenario completo para que instantes casi idénticos
   de una misma ruta no aparezcan a ambos lados.
5. La red predice los diez próximos pares de aceleración y giro. Para evaluarla
   se vuelve a introducir en el simulador y se observa el recorrido completo.

Este primer modelo era una red de cuatro capas ocultas de 512 neuronas, con
activación SiLU y dos vecinos como contexto. Su entrada tenía 46 valores y no
incluía rayos de distancia al mapa. Sirvió para comprobar que todo el recorrido
—generar, entrenar, cargar el modelo y conducir— funcionaba de extremo a extremo.
Se conserva como `Apr Superv local/modelos/politica_local_original.pt`; el
archivo `politica.pt` de esa carpeta es actualmente una copia del modelo final,
para que la interfaz antigua también pueda abrirlo.

## Ampliación de los datos en Google Cloud

La generación local era correcta, pero el planificador clásico necesita mucho
tiempo para producir miles de rutas. Por eso se preparó `Apr Superv nube/`: el
trabajo se dividió en tareas independientes, cada una con semillas distintas,
archivos de progreso y sincronización periódica. Si una máquina se detenía, la
tarea podía continuar desde lo ya guardado.

La nube se utilizó para **generar las rutas con el planificador clásico**. El
gasto real anotado al cerrar esa ejecución fue de **aproximadamente 364 € en
Google Cloud**. La cifra es aproximada porque el repositorio no contiene una
factura desglosada con la que darle más precisión. Los entrenamientos de redes
con GPU se hicieron después **en el ordenador local**, dentro de `Apr Superv
nube/`, con una NVIDIA RTX 5060 Ti de 16 GB.

Los CSV conservados contienen 1.101 runs y 4.312 trayectorias de vehículos. La
preparación seleccionó 924 runs útiles y produjo 612.770 muestras, repartidas en
64 lotes binarios. El vector más amplio posible tiene 277 valores. Los CSV
ocupan 60,73 MiB y los lotes preparados, unos 0,70 GiB.

También se generaron dos grupos independientes de 480 escenarios:

- **Selección:** se usó para decidir qué configuraciones continuaban.
- **Test:** se mantuvo fuera de todas las decisiones y se abrió una sola vez al
  final.

Esta separación importa porque elegir el modelo que mejor funciona en test
convertiría test en otro conjunto de selección y su resultado dejaría de medir
generalización.

## Qué información recibe la red

Cada muestra representa a un vehículo en un instante. La entrada contiene:

- su posición, orientación, velocidad, tamaño y límites físicos;
- la meta expresada desde el sistema de referencia del vehículo;
- los controles aplicados recientemente;
- los vecinos que podrían alcanzarlo dentro de un horizonte corto, con su
  posición, orientación, velocidad y prioridad;
- el modo de coordinación y el tamaño de la flota;
- distancias libres hasta los obstáculos, medidas mediante rayos.

La salida contiene diez pares de aceleración y giro. Durante el despliegue se
aplican esos controles al mismo modelo de bicicleta que usa el planificador.

## Los hiperparámetros en los dos planteamientos

Los **hiperparámetros** son decisiones que se fijan antes del entrenamiento. No
son los pesos que aprende la red, sino la forma de la red, la información que
recibe y la manera de actualizar sus pesos. El planteamiento local usó una
configuración pequeña para validar el método. El ampliado trató esos valores
como ejes de búsqueda y acabó con esta comparación:

| Hiperparámetro | Qué controla | Primer modelo local | Modelo ampliado final |
|---|---|---:|---:|
| Capas ocultas | Cuántas transformaciones sucesivas hace la red | 4 | 5 |
| Anchura | Neuronas por capa; aumenta capacidad y coste | 512 | 2.048 |
| Activación | Función no lineal entre capas | SiLU | GELU |
| Normalización | Estabiliza la escala interna de cada capa | LayerNorm | LayerNorm |
| Dropout | Fracción de activaciones apagadas al entrenar para regularizar | 0 | 0,2 |
| Residual | Atajos entre capas para facilitar redes profundas | No | No |
| Vecinos | Máximo de otros vehículos descritos en la entrada | 2 | 3 |
| Horizonte vecinal | Hasta cuántos pasos se considera relevante un vecino | 15 | 20 |
| Controles pasados | Memoria de aceleraciones y giros anteriores | 3 | 1 |
| Fourier | Pares seno/coseno para codificar la posición en varias escalas | 0 | 0 |
| Rayos | Distancias al mapa medidas en distintas direcciones | 0 | 18 |
| Lote | Muestras procesadas antes de cada actualización | No consta en el modelo guardado | 2.048 |
| Tasa de aprendizaje | Tamaño de cada actualización de pesos | No consta | 0,003 |
| Optimizador | Regla que convierte el gradiente en una actualización | No consta | AdamW |
| Decaimiento de pesos | Penalización de pesos grandes | No consta | 0,01 |
| Mezcla | Peso relativo de los modos poco frecuentes | No consta | Raros ×4 |
| Épocas | Pasadas completas sobre los datos | No consta | 640 |

`No consta` significa que ese valor no quedó guardado dentro del primer `.pt`;
inventarlo a partir de la configuración actual sería mezclar versiones.

## Cómo se eligieron los hiperparámetros

El error de imitación o MSE no bastó para elegir. En las primeras mediciones su
correlación con las colisiones fue aproximadamente `-0,03`: una red podía copiar
bien los controles medios y conducir mal al acumular pequeños errores. Desde
entonces cada candidata se evaluó conduciendo flotas completas, con detección de
colisiones y hasta 300 segundos simulados por escenario.

La nota por vehículo se definió así:

- llegada limpia: entre 1 y 2 puntos, según el ángulo final;
- recorrido limpio sin llegada: hasta 0,5, según distancia y ángulo restantes;
- colisión: como máximo 0,1, reducida cuanto más tiempo permanece en contacto.

Por tanto, una ruta que choca nunca puede superar a una ruta limpia. El tiempo
de contacto se mantiene como señal gradual: los datos mostraron mucho ruido y
no habría sido útil convertir todos los choques en el mismo cero.

### Resultado de cada eje en la exploración grande

La primera exploración cruzó muchos ejes a la vez. La tabla muestra la **nota
media marginal** de cada valor: agrupa redes que también diferían en los demás
hiperparámetros. Por eso sirve para orientar la siguiente fase, pero no demuestra
que un eje aislado causara toda la diferencia. La columna de ruido es la
dispersión medida entre configuraciones comparables; cuando la diferencia queda
en ese orden, el resultado se considera ruidoso.

| Eje | Mejor media observada | Peor media observada | Ruido | Lectura usada |
|---|---:|---:|---:|---|
| Capas | 3: `0,0206` | 6: `0,0132` | `0,0025` | Más profundidad no ayudó; se revisó aparte en fase 4. |
| Anchura | 512: `0,0195` | 4.096: `0,0130` | `0,0028` | Una red grande no resolvía por sí sola el problema. |
| Dropout | 0,2: `0,0201` | 0,35: `0,0095` | `0,0018` | Se mantuvo 0,2 y se volvió a medir en fase 5. |
| Activación | GELU: `0,0179` | ReLU: `0,0161` | `0,0020` | Eje plano en esta fase. |
| Tasa de aprendizaje | 0,003: `0,0185` | 0,02: `0,0088` | `0,0031` | 0,02 era demasiado alta; se conservó 0,003. |
| Lote | 2.048: `0,0204` | 8.192: `0,0146` | `0,0022` | 2.048 dio el mejor equilibrio medido. |
| Vecinos | 4: `0,0180` | 5: `0,0156` | `0,0022` | Diferencia pequeña y ruidosa; el final acabó usando 3. |
| Controles pasados | 10: `0,0187` | 1: `0,0073` | `0,0042` | El mejor estaba en el límite superior y se amplió el eje; fases posteriores terminaron favoreciendo 1. |
| Horizonte vecinal | 20: `0,0186` | 30: `0,0076` | `0,0031` | Se fijó 20. |
| Decaimiento de pesos | 0: `0,0193` | 0,01: `0,0150` | `0,0020` | La primera media favoreció 0; la búsqueda posterior eligió 0,01. |
| Normalización | LayerNorm: `0,0169` | Sin normalizar: `0,0167` | `0,0016` | Eje plano; LayerNorm permaneció entre las mejores. |
| Optimizador | AdamW: `0,0225` | SGD: `0,0122` | `0,0015` | Ventaja clara de AdamW. |
| Mezcla de modos | Raros ×4: `0,0200` | Recuperados limitados: `0,0144` | `0,0022` | Se eligió Raros ×4. |
| Fourier | 0 ondas: `0,0179` | 12 ondas: `0,0161` | `0,0019` | Eje plano; se eliminó del modelo final. |

Las 241 configuraciones de este bloque chocaban entre el 98,5 % y el 99,1 % de
media al agruparlas por tamaño, y usar 0, 6 o 12 ondas dejaba el choque entre el
98,7 % y el 99,0 %. Esto descartó que faltaran únicamente capacidad o una
codificación periódica de la posición.

Los rayos se añadieron después en 69 combinaciones. Sus medias agrupadas fueron
`0,02195` con 8 rayos, `0,02125` con 12 y `0,01801` con 18. También son datos
ruidosos y mezclan otras configuraciones: 18 no ganó en esa media inicial, pero
sí formó parte del modelo que sobrevivió a las repeticiones posteriores. Las
épocas no constituyeron un eje de esta exploración porque todas esas redes se
entrenaron durante 80; se compararon de forma controlada en la fase 5.

La búsqueda quedó registrada por fases:

| Fase | Qué se probó | Decisión obtenida |
|---|---|---|
| 1 | Exploración amplia aleatoria y barrido de rayos | El tamaño de red, Fourier y los rayos no resolvían por sí solos los choques. Se registraron 241 configuraciones iniciales y 69 pruebas adicionales de rayos. |
| 2 | Optuna sobre 15 ejes de búsqueda | El estudio actual conserva 895 pruebas: 448 completas, 437 podadas y 10 interrumpidas o pendientes. |
| 3 | Las 12 mejores configuraciones, repetidas con tres semillas y 320 épocas | Ganó una red de 4 capas y 2.048 neuronas, con media `0,11304`. Las repeticiones confirmaron que una sola nota era ruidosa. |
| 4 | Más anchura y redes de 7, 9 y 11 capas con conexiones residuales | Las redes profundas empeoraron. La referencia de 4 × 2.048 obtuvo `0,12495`; 7 × 2.048, `0,1031`; y 11 × 2.048, `0,0862`. |
| 5 | Rejilla de 3–5 capas, dropout `0,2–0,4` y 480/640 épocas | 640 épocas ganó en 7 de 9 comparaciones. La mejor medida aislada fue `0,1312` con 5 capas y dropout `0,2`. |
| 6 | Cuatro finalistas repetidas con tres semillas nuevas | Ganó `c5-d20-e640`: media `0,12509`, desviación `0,00687` y notas `0,12589`, `0,13307` y `0,11631`. |
| 8 | Publicación de la mejor instancia de la configuración ganadora y apertura única de test | Se eligió la semilla 2 y ya no se ajustó ningún parámetro con test. |

Las fases 5 y 6 sumaron respectivamente 29 h 19 min y 25 h 26 min de tiempo de
proceso medido. Como se ejecutaron hasta tres entrenamientos en paralelo y hubo
pausas, estas cifras no equivalen directamente al tiempo de reloj transcurrido.

## Modelo supervisado definitivo

La configuración elegida tiene 5 capas ocultas de 2.048 neuronas, activación
GELU, LayerNorm, dropout `0,2` y 640 épocas. Se entrenó con AdamW, lote 2.048,
`lr=0,003`, `weight_decay=0,01` y una ponderación ×4 de los modos menos
frecuentes. Observa tres vecinos, un control pasado, un horizonte vecinal de 20
pasos y 18 rayos. No usa Fourier ni conexiones residuales.

En los **480 escenarios de test**, con 2.150 vehículos, obtuvo:

| Medida | Resultado |
|---|---:|
| Nota media | 0,12851 |
| Vehículos que llegan | 2.048 / 2.150 — 95,3 % |
| Vehículos que chocan | 2.025 / 2.150 — 94,2 % |

La nota de test es coherente con las tres semillas de selección, por lo que no
aparece una caída clara al cambiar de escenarios. El resultado también muestra
el límite actual con claridad: casi todos los vehículos llegan, pero casi todos
tocan otro vehículo o el mapa. Es un modelo útil para estudiar la imitación y la
velocidad de inferencia, pero todavía no conduce de forma robusta.

## Comparación entre el primer modelo y el modelo ampliado

La comparación usa exactamente los mismos escenarios, la misma física y el
mismo criterio de colisión. El tiempo se mide después de calentar la GPU y se
promedia sobre muchos escenarios; así no se confunde la carga inicial de CUDA
con el coste normal de conducir.

Se midieron 400 escenarios de selección —1.934 vehículos— tres veces en una
RTX 5060 Ti, tras un calentamiento de 8 escenarios:

| Modelo | Nota | Llegan | Chocan | Tiempo para 400 escenarios | Media por escenario |
|---|---:|---:|---:|---:|---:|
| Primer modelo local | 0,01236 | 869/1.934 — 44,9 % | 1.924/1.934 — 99,5 % | 29,23 ± 0,12 s | 73,06 ms |
| Modelo ampliado final | 0,12297 | 1.835/1.934 — 94,9 % | 1.831/1.934 — 94,7 % | 11,68 ± 0,03 s | 29,21 ms |

El modelo ampliado multiplica la nota por 9,95, duplica aproximadamente la tasa
de llegada y tarda 2,50 veces menos en completar este lote. Que una red mayor
termine antes no significa que cada pasada neuronal sea más barata: el rollout
deja de consultar a los vehículos cuando llegan, y el primer modelo mantiene
muchos más activos hasta el límite de 300 segundos simulados. El tiempo mide el
sistema completo en este conjunto, que es la comparación práctica buscada.

El modelo definitivo está en `Apr Superv nube/datos/modelos/politica.pt`. La
interfaz visual se abre con `Apr Superv nube/gui.bat` y lo selecciona por
defecto.

# Parte III — Aprendizaje por refuerzo

La tercera parte está planificada, pero aún no se ha ejecutado. El agente
aprenderá al interactuar con el simulador y recibirá recompensa por llegar,
mantener márgenes de seguridad, evitar bloqueos y completar la flota con poco
tiempo.

Aquí están las mayores expectativas de relación entre calidad y tiempo: una vez
entrenada, la política debería conservar la rapidez de una red neuronal y, al
aprender de las consecuencias de sus propios actos, corregir parte de la
fragilidad del aprendizaje puramente supervisado. Esa hipótesis tendrá que
validarse con los mismos conjuntos fijos, métricas de colisión y mediciones de
tiempo usadas en la segunda parte.
