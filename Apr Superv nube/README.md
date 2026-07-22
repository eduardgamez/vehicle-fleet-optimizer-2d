# Apr Superv nube

Pipeline de aprendizaje supervisado a escala: genera masivamente rutas de flota
con CPU y entrena miles de redes con GPU para quedarse con la mejor.

La red aprende a conducir una flota imitando a un planificador clásico. Se parte
de un escenario —vehículos con su tamaño, dinámica, pose de partida, destino,
ángulo de llegada, grupo y prioridad— sobre un mapa fijo; el planificador lo
resuelve, y de esas trayectorias salen los ejemplos: en cada instante y para cada
vehículo, qué aceleración y qué giro se aplicaron.

---

## Índice

1. [Configuración y coste](#1-configuración-y-coste)
2. [Los archivos](#2-los-archivos)
3. [Plataforma](#3-plataforma)
4. [Cómo se lanza](#4-cómo-se-lanza)
5. [Generación de datos](#5-generación-de-datos)
6. [Evaluación con casos nuevos](#6-evaluación-con-casos-nuevos)
7. [Decisiones de rendimiento](#7-decisiones-de-rendimiento)
8. [Prueba en local](#8-prueba-en-local)
9. [Lo que no está comprobado](#9-lo-que-no-está-comprobado)

---

## 1. Configuración y coste

| | |
|---|---|
| Mapa | `mapas/mapa_entrenamiento.json`, siempre el mismo |
| Calidad de ruta | 5, sin ningún plazo de reloj |
| Tamaños de flota | 1-6 y 8 vehículos (`--veh "1-6,8"`) |
| Modos de optimización | un tercio cada uno |
| Grupos y prioridades | estructura sorteada por escenario |
| Órdenes explorados | `(1 + ln(posibles)/4)^4`, tope 400 |
| Eje de vecinos | `[3, 4, 5, 7]` |
| Escenarios | 10.000 |

Órdenes de planificación que se comparan en cada escenario: **100 %** de los
posibles con 1-2 vehículos, **83 %** con 3, **46 %** con 4, **20 %** con 5,
**6,8 %** con 6 y **0,4 %** con 8.

### Coste

Medido en un i7 de 14ª generación: planificar un orden completo a calidad 5
cuesta **115 s + 35 s por vehículo**. Con eso, y sorteando 400 escenarios reales:

| modo | coste medio por escenario |
|---|---|
| secuencial | 0,15 h (un único orden) |
| prioridades | 0,44 h |
| global | 3,42 h |
| **media** | **1,33 h** |

`prioridades` sale barato porque solo permuta dentro de cada grupo, y con los
grupos variados la flota casi siempre queda repartida en varios.

| | |
|---|---|
| CPU · 10.000 escenarios | **~146 €** · 10 h de reloj con 1.280 núcleos |
| Preparar datos | ~5 € |
| GPU · barrido | ~70-100 € *(estimado, ver sección 9)* |
| Almacenamiento | ~5 € |
| **Total** | **~230-260 €** |

Salen ~7 millones de ejemplos, con 3,99 vehículos por escenario de media.

El tamaño del trabajo se regula con tres parámetros: `--escenarios`,
`--curvatura-ordenes` (cuánto se explora dentro de cada escenario) y `--veh`
(qué flotas se sortean). Con curvatura 5 el coste de CPU sube a ~200 €; con
15.000 escenarios y curvatura 4, a ~220 € para ~10,5 millones de ejemplos.

---

## 2. Los archivos

| archivo | qué es |
|---|---|
| `comun.py` | rutas, reparto entre máquinas y sincronización con el bucket. Lo importan todos |
| `vectorizado.py` | biblioteca: entradas de la red y simulación en bucle cerrado, vectorizadas |
| `generador_nube.py` | **fase 1** — genera las rutas (CPU, N máquinas) |
| `escenarios.py` | genera los casos de evaluación (segundos) |
| `preparar_datos.py` | **fase 1.5** — CSV → binario (CPU, 1 máquina) |
| `entrenar_nube.py` | **fase 2** — barrido de hiperparámetros (GPU, N máquinas) |
| `finalizar.py` | **fase 3** — reentrena el ganador con todo y lo examina |
| `verificar.py` | comprueba la coherencia de los cálculos vectorizados |
| `Dockerfile`, `requirements_nube.txt` | imagen única para las dos fases |
| `gcp/` | scripts y definiciones de trabajo de Google Cloud |

Los datos van a `datos/` (ignorada por git): `rutas/`, `muestras/`,
`escenarios/` y `modelos/`. La ubicación se reapunta con la variable `TDR_DATOS`.

---

## 3. Plataforma

**Google Cloud, con Batch y máquinas Spot.**

Las fases son trabajos por lotes, no servicios, y GCP Batch es exactamente eso:
recibe un contenedor y un número de tareas, levanta N máquinas, reparte,
reintenta la que se caiga y las apaga al acabar.

Las máquinas **Spot** cuestan un 60-80 % menos, y este trabajo las aprovecha
porque todo es reanudable: cada tarea sube su avance al bucket cada pocos minutos
y, si la cortan, Batch la reintenta y continúa donde iba.

La fase 1 escala con vCPU baratos (`c4-highcpu`) y la fase 2 con una GPU modesta
(`g2-standard-8`, una NVIDIA L4): la red es un MLP pequeño, así que lo que rinde
son varias GPU pequeñas trabajando a la vez.

El código no está atado a GCP. `TDR_BUCKET` acepta `gs://` y `s3://`,
`comun.indice_tarea` reconoce las variables de entorno de los orquestadores
habituales, y sin `TDR_BUCKET` todo funciona con carpetas locales sin tocar la
red.

---

## 4. Cómo se lanza

```
generador_nube.py   fase 1    CPU, N máquinas   escenarios → CSV de rutas
preparar_datos.py   fase 1.5  CPU, 1 máquina    CSV → superset de muestras (.npy)
escenarios.py       (segundos)                  escenarios de selección y de test
entrenar_nube.py    fase 2    GPU, N máquinas   barrido de hiperparámetros
finalizar.py        fase 3    1 máquina         reentreno del ganador + examen
```

```bash
export TDR_PROYECTO=mi-proyecto
bash "Apr Superv nube/gcp/lanzar.sh" preparar   # imagen, bucket, escenarios
bash "Apr Superv nube/gcp/lanzar.sh" generar    # fase 1
bash "Apr Superv nube/gcp/lanzar.sh" muestras   # fase 1.5
bash "Apr Superv nube/gcp/lanzar.sh" amplia     # fase 2a · 35 % de los datos
bash "Apr Superv nube/gcp/lanzar.sh" fina       # fase 2b · 65 % de los datos
bash "Apr Superv nube/gcp/lanzar.sh" final      # fase 3 · 100 %
```

El tamaño se ajusta con `TDR_ESCENARIOS`, `TDR_TAREAS_CPU`, `TDR_TAREAS_GPU` y
`TDR_CONFIGS`. Al terminar, `finalizar.py` deja el modelo en `politica.pt`.

---

## 5. Generación de datos

Cada escenario se sortea a partir de una semilla, de forma completamente
determinista: tamaños, dinámicas, poses de partida, destinos, ángulos de llegada
y separaciones. El mapa es siempre el mismo, y eso es deliberado: el mapa no se
codifica en las entradas de la red, de modo que la red lo aprende implícitamente
a partir de las coordenadas.

**Sin plazos de reloj.** El planificador solo está limitado por su tope de nodos,
que depende de la calidad. Al no intervenir el reloj, el dataset es reproducible:
la misma semilla da siempre exactamente las mismas rutas, corra donde corra y con
la máquina más o menos cargada.

**Grupos y prioridades variados** (`sortear_preferencias`). Se sortea primero la
estructura —cuántos grupos distintos hay, desde uno solo hasta uno por vehículo,
y cuántos niveles de prioridad— y luego el reparto, con pesos desiguales. Así la
red ve desde flotas sin ninguna preferencia hasta flotas totalmente
jerarquizadas, y todo lo de en medio, que es lo que da sentido a los modos
`prioridades` y `global`.

**Cuántos órdenes se exploran** (`ordenes_a_explorar`). Los modos `global` y
`prioridades` comparan varios órdenes de planificación y se quedan con el mejor.
El número sale de la curva `(1 + ln(posibles)/k)^k`, que recorta poco al
principio y se va aplanando: el porcentaje explorado acaba decayendo casi al
mismo ritmo al que el problema se vuelve inabarcable. Los posibles son n! en
`global` y el producto de los factoriales de cada grupo en `prioridades`.

**El orden ganador se replanifica entero** con el presupuesto de nodos completo.
Las rutas de la fase de comparación se descartan: solo servían para decidir cuál
gana.

**Los tres modos a partes iguales.** La red recibe el modo de optimización como
entrada, de modo que necesita ver los tres.

**Eje de vecinos `[3, 4, 5, 7]`.** El tope sale del propio problema: con flotas
de hasta 8 vehículos ninguno puede ver más de 7 vecinos, y los valores más altos
serían bloques de entrada siempre a cero.

---

## 6. Evaluación con casos nuevos

Hay tres conjuntos de escenarios **disjuntos**, en rangos de semilla separados
(0…, 900 000 000…, 950 000 000…), así que no pueden solaparse por mucho que
crezca el dataset:

- **entrenamiento** — los CSV con las rutas del planificador. De ellos sale
  también un 10 % de runs apartados para la validación por MSE, que solo sirve
  para elegir la mejor época de cada entrenamiento.
- **selección** — escenarios nuevos con los que el barrido decide qué
  configuración gana. **Todas** las configuraciones se puntúan aquí, nunca sobre
  los escenarios con los que han entrenado.
- **test** — escenarios nuevos que no intervienen en ninguna decisión. Se usan
  una sola vez, al final, sobre el ganador ya reentrenado. Esa es la cifra
  honesta: elegir el mejor de entre miles por su nota en un conjunto sesga esa
  nota al alza, porque en parte se está eligiendo a quien tuvo suerte con esos
  casos.

Los escenarios de evaluación **no necesitan que el planificador los resuelva**:
para puntuar a la red bastan las condiciones iniciales, porque lo que se mide es
si lleva su flota a las metas, no si copia una ruta concreta. Por eso
`escenarios.py` genera 400 casos en segundos en vez de horas de hybrid A\*.

---

## 7. Decisiones de rendimiento

**a) Superset de muestras con recorte de columnas** (`vectorizado.py`).
El barrido explora tres ejes que cambian la propia entrada de la red: nº de
vecinos, historia de controles y horizonte del filtro de vecinos. Por cómo está
definida la entrada, la representación pequeña es un **recorte exacto** de la
grande: los vecinos vienen ordenados por urgencia y rellenados por orden, la
historia va alineada a la derecha, y bajar el horizonte solo descarta los últimos
bloques ocupados. Las muestras se construyen una sola vez con los valores
máximos, y cada configuración del barrido es un recorte de columnas más una
máscara, sobre datos que ya residen en la GPU.

**b) Entradas y simulación vectorizadas** (`vectorizado.py`).
Tanto la construcción de las muestras como el rollout en bucle cerrado operan
sobre arrays completos en lugar de recorrer vehículos e instantes uno a uno. En
el rollout importa especialmente: la nota se calcula una vez por configuración y
son 1.800 pasos por todos los vehículos de todos los escenarios de evaluación.

> `verificar.py` comprueba que la vía vectorizada y la directa dan exactamente
> los mismos números sobre CSV reales: **diferencia máxima 0,000e+00**, tanto en
> las muestras (para varias representaciones) como en las poses finales del
> rollout.

**c) La unidad de trabajo de la fase 1 es (escenario, orden)**
(`generador_nube.py`).
Los órdenes candidatos de un escenario son independientes entre sí, así que se
reparten entre todos los núcleos en lugar de resolverse en serie dentro de un
único proceso. Las tareas viajan como `(semilla, orden)` —el worker reconstruye
la flota desde la semilla, que es determinista—, el mapa y los kernels de numba
se compilan una vez por proceso y no una por escenario, y el proceso padre es el
único que escribe.

**d) Todo es reanudable.**
Es lo que hace aprovechables las máquinas spot: se anotan las semillas ya
resueltas y las configuraciones ya evaluadas, y lo hecho se sube al bucket cada
pocos minutos, de modo que una interrupción cuesta minutos. El identificador de
cada run es su semilla, así que un escenario a medio escribir se descarta al
preparar los datos en lugar de colarse duplicado.

**e) Criba temprana** (`entrenar_nube.py --criba`).
A la mitad de su presupuesto de épocas, una configuración cuyo error de
validación supere el percentil 85 de las ya vistas en ese mismo punto se
abandona. Está calibrada del lado seguro: solo se descarta el ~15 % peor, así que
una red que arranque lenta pero vaya a remontar casi seguro sobrevive. Los hitos
son **fracciones del presupuesto de cada configuración**, no épocas absolutas,
porque con presupuestos de 40, 80 y 120 épocas la época 10 significa cosas muy
distintas en cada una. Se apaga con `--criba ""`, lo indicado en la rejilla fina,
donde todas las candidatas son buenas. No cambia el espacio de búsqueda: las
mismas configuraciones se prueban y se puntúan.

**f) Datos escalonados entre fases** (`entrenar_nube.py --fraccion-datos`).
La búsqueda amplia solo tiene que ordenar configuraciones entre sí, y para eso
basta una parte del dataset; la rejilla fina afina de verdad y merece bastante
más; y el modelo final se entrena con todo. Reparto: **0,35 → 0,65 → 1,0**. El
submuestreo es por filas y no por escenarios, para que con poca fracción se sigan
viendo todos los escenarios con menos instantes de cada uno: rinde más la
variedad de situaciones que el detalle de cada una.

**g) Submuestreo temporal** (`preparar_datos.py --paso`).
A DT = 0,1 s dos instantes seguidos son casi el mismo dato. Con `--paso 2` o `3`
el dataset se reduce a la mitad o a un tercio sin perder casi información. Por
defecto está a 1.

---

## 8. Prueba en local

Todo funciona igual en un PC, solo que más pequeño.

```bash
cd "Apr Superv nube"
python verificar.py                            # coherencia de los cálculos
python escenarios.py --n 40                    # escenarios de evaluación
python generador_nube.py --escenarios 4 --veh "2,3"
python preparar_datos.py
python entrenar_nube.py --n-configs 4 --max-muestras 200000
python finalizar.py
```

El circuito completo en GCP se valida por unos céntimos antes de lanzar los
10.000 escenarios:

```bash
TDR_ESCENARIOS=50 TDR_TAREAS_CPU=2 bash "Apr Superv nube/gcp/lanzar.sh" generar
```

---

## 9. Lo que no está comprobado

- **La fase 2 no se ha ejecutado nunca.** El código compila, pero en el entorno
  donde se escribió no había torch instalado. La estimación de coste de GPU
  (~70-100 €) es un cálculo a ojo, no una medida.
- **Los JSON de `gcp/` no se han validado contra la API real** de Batch.
- Los precios de spot varían por región y momento, así que conviene contrastarlos
  en la calculadora de Google. El orden de magnitud sí es fiable.

Sí están medidos el coste por orden del planificador (115 s + 35 s por vehículo a
calidad 5) y la coherencia exacta de los cálculos vectorizados.

### Dos cosas que hay que decidir antes de preparar los datos

- **`preparar_datos.N_VEC_MAX = 7`.** Es el máximo del eje de vecinos. Subirlo
  después obliga a rehacer el superset entero.
- **`--max-muestras` (12 M).** Es lo que cabe holgado en una L4 de 24 GB con
  entradas de 110 columnas. Por encima de eso hace falta una GPU mayor, o usar
  `--en-cpu` en el reentrenamiento final.
