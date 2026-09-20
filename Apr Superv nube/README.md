# Aprendizaje supervisado de la flota

Esta carpeta contiene el pipeline que enseña a una red neuronal a conducir los
vehículos del simulador. La red aprende de rutas calculadas por el planificador
clásico y después se prueba conduciendo por sí misma, en bucle cerrado.

El nombre «nube» es histórico: el código se preparó para poder repartir trabajo
entre máquinas, pero **los entrenamientos con GPU de este estudio se han hecho
en local**, dentro de esta misma carpeta. Los archivos de GCP son una vía de
despliegue opcional que todavía no se ha probado contra Batch real.

## Estado actual

Datos medidos en los archivos de `datos/`:

| elemento | estado |
|---|---:|
| Runs preparados | 924 |
| Muestras de entrenamiento | 612.770 |
| Lotes binarios | 64 |
| Tamaño de las muestras | 0,70 GiB |
| Escenarios de selección generados | 480 |
| Escenarios de test generados | 480 |
| Escenarios usados por defecto en cada selección | 400 |

La **fase 6 está en ejecución**. Es la última comparación de
hiperparámetros: cuatro candidatas fijadas de antemano, cada una con tres
semillas nuevas. Cuando termine se elegirá por la media de esas tres notas.

La mejor medida aislada hasta la fase 5 es `0,13120`, con 5 capas, 2.048
neuronas por capa, dropout 0,2 y 640 épocas. La segunda es `0,13082`, con la
misma red y dropout 0,4. Esa diferencia (`0,00038`) es menor que el ruido entre
semillas ya observado, por eso todavía no se considera ganadora a ninguna.

## Qué aprende la red

Cada muestra describe un vehículo en un instante. La entrada reúne su estado,
la meta, los controles anteriores, los vecinos más relevantes, el modo de
priorización y las distancias a obstáculos medidas con rayos. La salida son los
diez próximos pares de aceleración y giro.

Durante el entrenamiento se imitan los controles del planificador. Para elegir
modelos no basta con mirar ese error: la red se introduce en el simulador y
conduce flotas completas durante un máximo de 300 segundos simulados.

La nota final de cada vehículo distingue tres casos:

- Si llega sin chocar, obtiene entre 1 y 2 puntos según el ángulo final.
- Si no llega pero tampoco choca, obtiene como máximo 0,5 según la distancia y
  el ángulo restantes.
- Si choca, obtiene como máximo 0,1 y pierde más cuanto más tiempo permanece en
  contacto.

Así, una trayectoria limpia siempre queda por encima de una que atraviesa un
obstáculo, pero las redes malas todavía reciben una señal gradual que permite
compararlas.

## Separación de los datos

Hay tres conjuntos que no comparten semillas:

1. **Entrenamiento.** Son las rutas del planificador. Un 10 % de los runs se
   reserva de forma estable para medir el error de validación y conservar la
   mejor época de cada entrenamiento.
2. **Selección.** Son escenarios sin ruta objetivo. La red los conduce y esta
   nota decide qué configuración continúa. Las fases 1 a 6 usan este conjunto.
3. **Test.** No interviene en ninguna decisión. Solo se abrirá una vez, después
   de elegir la configuración de fase 6 y entrenar el modelo definitivo.

Los archivos `seleccion.json` y `test.json` contienen 480 escenarios cada uno.
Los barridos actuales limitan selección a los primeros 400 para mantener todas
las comparaciones anteriores sobre los mismos casos. El test final usará los
480.

## Recorrido del experimento

| paso | script | qué se hizo | resultado principal |
|---|---|---|---|
| Datos | `generador_nube.py` | El planificador produjo rutas con distintos tamaños de flota, prioridades y modos | 924 runs aprovechables |
| Preparación | `preparar_datos.py` | Las rutas se convirtieron en un superset binario reutilizable | 612.770 filas, dimensión máxima 277 |
| Fase 1 | `entrenar_nube.py` | Exploración amplia | 241 configuraciones registradas en el primer resumen |
| Fase 2 | `buscar_optuna.py` | Búsqueda guiada por Optuna y ampliación del eje de rayos | El estudio actual conserva 895 pruebas: 448 completas, 437 podadas y 10 pendientes o interrumpidas |
| Fase 3 | `fase3.py` | Las 12 mejores se repitieron con 3 semillas y 320 épocas | Ganó la configuración de 4 capas y ancho 2.048: media `0,11304` |
| Fase 4 | `fase4.py` | Se probaron redes de 7, 9 y 11 capas con atajos residuales | Las profundas empeoraron; el control de 4 capas obtuvo `0,12495` |
| Fase 5 | `fase5.py` | Rejilla de 3–5 capas, dropout 0,2–0,4 y 480/640 épocas | 640 épocas ganó 7 de 9 comparaciones; las dos mejores fueron las de 5 capas |
| Fase 6 | `fase6.py` | Reválida final de cuatro candidatas con semillas 1, 2 y 3 | En ejecución |
| Final | `finalizar.py` | Elegirá por la media de fase 6, entrenará con semilla 4 y medirá test una vez | Pendiente |

Las semillas de fase 6 son nuevas. La semilla 0 de fase 5 sirvió para escoger
las candidatas y no se reutiliza en su media; así no se premia dos veces una
medición afortunada.

## Archivos principales

| archivo | función |
|---|---|
| `comun.py` | Rutas compartidas, carpetas de datos y sincronización opcional |
| `generador_nube.py` | Generación paralela de rutas con CPU |
| `preparar_datos.py` | Conversión de CSV al superset de arrays NumPy |
| `vectorizado.py` | Construcción de entradas y rollout de muchas flotas a la vez |
| `entrenar_nube.py` | Entrenamiento y barrido amplio |
| `buscar_optuna.py` | Búsqueda guiada y reanudable con Optuna |
| `fase3.py` | Reválida de las mejores configuraciones iniciales |
| `fase4.py` | Experimento de profundidad y bloques residuales |
| `fase5.py` | Rejilla fina de capas, dropout y épocas |
| `fase6.py` | Reválida final con semillas independientes |
| `finalizar.py` | Entrenamiento definitivo y única evaluación de test |
| `escenarios.py` | Generación de selección y test con rangos de semilla separados |
| `verificar.py` | Comparación entre los cálculos directos y vectorizados |
| `panel.py`, `panel_fase1.py` | Paneles de seguimiento de los barridos antiguos |
| `gcp/`, `Dockerfile` | Despliegue opcional por lotes en Google Cloud |

Todo lo generado se guarda en `datos/`, que Git ignora:

```text
datos/
├── escenarios/   seleccion.json y test.json
├── muestras/     superset binario dividido en lotes
├── rutas/        CSV originales, si se conservan
└── modelos/      bases Optuna, CSV, modelos y registros
```

La variable `TDR_DATOS` permite usar otra ubicación. En esta copia están las
muestras preparadas, pero `datos/rutas/` no contiene ya los CSV originales. Se
puede seguir entrenando; para reconstruir las muestras desde cero habría que
recuperar o regenerar esas rutas.

## Ejecución local

Desde `Apr Superv nube`, usando el entorno virtual del proyecto:

```powershell
$py = "..\.venv\Scripts\python.exe"
& $py fase5.py --resumen
& $py fase6.py --resumen
```

### Ver el modelo conduciendo

Haz doble clic en `gui.bat`, o ejecuta:

```powershell
& $py gui.py
```

La interfaz abre el primer escenario de selección y elige, por este orden,
`datos/modelos/politica.pt`, el mejor modelo de fase 5 o el `.pt` más reciente.
El menú **Modelo** permite cargar otro fichero. **Escenarios de selección**
recorre los casos guardados y **Frecuencia de control** permite recalcular cada
10, 5, 2 o 1 pasos. Al pulsar **Calcular y simular**, la interfaz usa el mismo
rollout, mapa y comprobación de choques que la evaluación; un vehículo se pinta
con contorno rojo y una cruz mientras está tocando otro vehículo o el mapa.

También se puede elegir el modelo y el escenario al arrancar:

```powershell
& $py gui.py --modelo "datos/modelos/fase5/mejor_c5-d20-e640.pt" --escenario 25
```

La interfaz solo carga escenarios de selección. No abre los escenarios de test,
que quedan reservados para la evaluación final.

La fase 6 se reparte entre tres procesos ocultos. Cada combinación se reserva
mediante un fichero atómico, de modo que los procesos no repiten trabajo:

```powershell
& $py fase6.py --lanzar --frac-vram 0.27
```

No se abren ventanas nuevas. Los procesos escriben en
`datos/modelos/fase6/log_t0.txt`, `log_t1.txt` y `log_t2.txt`.

### Pausar y reanudar la fase 6

Para pedir una pausa segura:

```powershell
& $py fase6.py --pausar
```

Cada proceso termina su época actual, guarda un checkpoint completo y sale. El
checkpoint contiene los pesos actuales, el mejor estado, el optimizador, el
calendario de aprendizaje, la época, el tiempo acumulado y los estados
aleatorios. Así la reanudación continúa desde la época siguiente. Mientras está
activo también se guarda automáticamente cada 10 épocas, para limitar la pérdida
si se apaga el equipo sin pedir antes la pausa.

El estado se consulta y el trabajo se reanuda con:

```powershell
& $py fase6.py --estado
& $py fase6.py --lanzar --frac-vram 0.27
```

`--lanzar` elimina la orden de pausa, libera las reservas incompletas y abre tres
trabajadores ocultos. Al completar un modelo se borra su checkpoint; el `.pt` y
la fila del CSV pasan a ser el resultado definitivo.

No se debe ejecutar `finalizar.py` hasta que `fase6.py --resumen` muestre `3/3`
en las cuatro candidatas. El finalizador se niega a continuar si falta alguna.

## Rendimiento medido en local

Equipo usado: NVIDIA GeForce RTX 5060 Ti de 16.311 MiB, con tres procesos
simultáneos limitados al 27 % de VRAM cada uno.

Tiempos de fase 5 por entrenamiento, medidos mientras los tres procesos
compartían la GPU:

| red | 480 épocas | 640 épocas |
|---|---:|---:|
| 3 capas × 2.048 | 58,4 min | 77,7 min |
| 4 capas × 2.048 | 84,0 min | 112,0 min |
| 5 capas × 2.048 | 108,8 min | 145,1 min |

La fase 5 completa sumó 29,31 horas de proceso. Repartidas continuamente entre
tres procesos equivalen a 9 h 46 min de reloj. Para fase 6, la suma medida de
tareas equivalentes da 25,97 horas de proceso; repartida igual, la previsión
inicial es **8 h 40 min**. Es una extrapolación de esas medidas: la carga del
equipo puede mover el tiempo real.

## Decisiones técnicas importantes

- El superset guarda la representación máxima una sola vez. Cada configuración
  recorta columnas para variar vecinos, historia, Fourier y rayos sin reconstruir
  el dataset.
- La construcción de entradas y el rollout trabajan con flotas completas en
  arrays, evitando bucles Python por vehículo.
- Los rayos aportan distancias reales al mapa. La configuración finalista usa
  18 rayos; las ondas de Fourier quedaron en 0 porque no mostraron mejora útil.
- Los vecinos se ordenan por urgencia. La zona final usa 3 vecinos, un paso de
  controles anteriores y horizonte 20.
- Los modelos se comparan por rollout de selección. El MSE solo elige la mejor
  época dentro de un entrenamiento.
- Las fases largas son reanudables: los CSV marcan lo terminado y los cerrojos
  reparten tareas entre procesos.

## Comprobaciones y límites conocidos

- `verificar.py` llegó a medir diferencia máxima `0,000e+00` entre la vía
  directa y la vectorizada, tanto en muestras como en poses finales. Ahora no se
  puede repetir esa prueba porque los CSV originales no están en `datos/rutas/`.
- Los resultados de una sola semilla son ruidosos. En fase 3, repetir una misma
  configuración produjo diferencias de hasta aproximadamente `0,030`; por eso
  fase 6 decide con tres semillas nuevas.
- Los archivos de GCP no se han validado contra la API real de Batch. Tampoco
  representan las fases locales 3–6 completas; antes de usarlos conviene hacer
  una ejecución pequeña de extremo a extremo.
- El entorno local medido usa PyTorch `2.11.0+cu128`. La imagen Docker conserva
  una versión distinta fijada en `requirements_nube.txt`; su equivalencia no se
  ha medido.
- Las cifras antiguas de coste de nube eran estimaciones, no gasto observado, y
  se han retirado de este documento.

## Después de fase 6

No habrá otra búsqueda de hiperparámetros. El procedimiento final será:

1. Comprobar que las cuatro candidatas tienen tres resultados.
2. Elegir la mayor media, conservando también la desviación y cada nota.
3. Entrenar esa configuración desde cero con la semilla fija 4.
4. Evaluarla una sola vez sobre los 480 escenarios de test.
5. Guardar `datos/modelos/politica.pt` y copiarlo a
   `Apr Superv local/modelos/politica.pt` para usarlo en la interfaz.

Cuando el resumen esté completo, todo ese cierre se ejecuta con:

```powershell
& $py finalizar.py
```

El test no debe consultarse antes: en cuanto se usa para decidir algo deja de
ser una medida independiente del modelo final.
