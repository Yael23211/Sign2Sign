# Sign2Sign — Continuación de documentación técnica  
## LSM Canonical V2, entrenamiento, evaluación offline y pruebas en webcam

> **Continuación directa del documento anterior.**  
> El documento previo cerró la etapa de entrenamiento y evaluación del clasificador ASL, incluyendo sus pruebas en webcam.  
> Este archivo continúa a partir de ese punto y documenta el trabajo realizado para **LSM**, desde la primera selección de modelos hasta la construcción de **LSM Canonical V2**, el entrenamiento definitivo, la evaluación offline, las pruebas controladas en webcam y las auditorías finales.

---

# 1. Punto de partida de LSM

Al iniciar esta etapa ya se contaba con:

- Dataset LSM procesado a landmarks.
- `85,883` muestras genuinas LSM válidas.
- Clases genuinas disponibles: `23`.
- Letras faltantes en el dataset LSM original: `E`, `L` y `Z`.
- Sustitución provisional de `E`, `L` y `Z` con muestras provenientes de ASL.
- Provisionales:
  - `E`: 929
  - `L`: 930
  - `Z`: 900
  - Total: `2,759`
- Representaciones disponibles:
  - `img_norm`
  - `world_norm`
- Sin `NaN` ni `Inf`.
- Dos clasificadores independientes planeados:
  - ASL.
  - LSM.
- La evaluación del modelo debía mantenerse separada entre:
  - clases genuinas LSM;
  - clases provisionales E/L/Z.

La decisión metodológica se mantuvo: **E/L/Z pueden utilizarse durante entrenamiento para completar las 26 salidas del clasificador, pero no deben inflar ni contaminar las métricas principales de LSM genuina**.

---

# 2. Split LSM V1 utilizado inicialmente

El primer split utilizado fue:

| Split | Muestras |
|---|---:|
| Train | 60,907 |
| Validation | 14,791 |
| Test | 12,944 |

Características importantes:

- `26` clases en train debido a E/L/Z provisionales.
- `23` clases genuinas en validation.
- `23` clases genuinas en test.
- E/L/Z se mantuvieron exclusivamente en train.
- La separación se realizó por grupos/sesiones para evitar leakage.

La validación correspondía principalmente a la **sesión 3**, mientras que el test original utilizaba la **sesión 6**.

---

# 3. Primera selección de modelos LSM

Inicialmente se compararon cuatro experimentos, siguiendo la misma lógica usada para ASL:

- L1: Random Forest + `img_norm`.
- L2: Random Forest + `world_norm`.
- L3: MLP + `img_norm`.
- L4: MLP + `world_norm`.

Después se corrigió la evaluación para que las métricas macro se calcularan únicamente sobre las **23 clases genuinas LSM**.

## 3.1 Resultados corregidos en validation

| Experimento | Accuracy | Precision macro genuina | Recall macro genuina | F1 macro genuina | Top-3 | Predicciones E/L/Z |
|---|---:|---:|---:|---:|---:|---:|
| L1 RF img_norm | 0.873842 | 0.908359 | 0.870155 | 0.861162 | 0.994929 | 80 |
| L2 RF world_norm | 0.867825 | 0.899744 | 0.864853 | 0.855840 | 0.995741 | 102 |
| L3 MLP img_norm | **0.970117** | 0.969591 | 0.967523 | **0.967452** | 0.999797 | **4** |
| L4 MLP world_norm | 0.966669 | **0.970192** | 0.964145 | 0.966094 | **0.999865** | 61 |

### Conclusión inicial

- MLP fue claramente superior a Random Forest.
- `img_norm` y `world_norm` funcionaron bien.
- L3 fue seleccionado provisionalmente por su mejor F1 macro genuino y baja tasa de predicción provisional.
- A partir de esta evidencia se decidió que **no tenía sentido seguir utilizando Random Forest en iteraciones posteriores**.

---

# 4. Comportamiento de las curvas de entrenamiento

En L3 y L4 se observó:

- accuracy de entrenamiento cercana al 99%;
- accuracy de validación notablemente menor;
- `train_loss` descendiendo continuamente;
- `validation_loss` alcanzando mínimos tempranos y después aumentando/oscillando.

En L3:

- mejor época por `val_loss`: 5;
- mejor época por `val_accuracy`: 5.

La interpretación fue que existía cierto **overfitting**, pero no necesariamente un problema que debiera corregirse agresivamente. La validación estaba formada por una sesión distinta de train, por lo que parte de la brecha reflejaba diferencias de dominio entre sesiones.

Esto se volvió especialmente importante al abrir el test.

---

# 5. Primer test final LSM y detección del problema de dominio

Después de seleccionar L3 únicamente con validation, se abrió el test original.

## 5.1 Resultado

| Métrica | Resultado |
|---|---:|
| Accuracy | 72.71% |
| Precision macro genuina | 72.30% |
| Recall macro genuina | 73.76% |
| F1 macro genuina | 70.79% |
| Top-3 | 81.36% |
| Predicciones provisionales E/L/Z | 120 / 12,944 = 0.9271% |

Errores:

- Top-1 incorrectos: `3,533`.
- Errores Top-1 cuya clase real seguía en Top-3: `1,120`.
- Clase real fuera del Top-3: `2,413`.

El resultado era demasiado bajo respecto a validation y mostraba anomalías muy fuertes.

## 5.2 Clases colapsadas

Al revisar el reporte por clase se detectó:

- `G`: F1 = 0.
- `H`: F1 = 0.
- `Y`: F1 = 0.

En particular:

- `G → X`: 650 casos.
- `H → X`: 650 casos.
- `Y → X`: 650 casos.

Esto indicaba que el problema no era un error aleatorio del MLP.

---

# 6. Auditoría del dominio del dataset LSM

Se realizó una auditoría de:

- metadatos;
- grupos;
- sesiones;
- centroides;
- vecinos más cercanos;
- distancias entre grupos;
- imágenes originales.

El objetivo fue determinar si el split de test realmente representaba la misma referencia gestual que el resto del proyecto.

## 6.1 Hallazgo principal: sesión 6

Las clases problemáticas pertenecían a:

- `FRAM_G6`
- `FRAM_H6`
- `FRAM_Y6`

Todas estaban en sesión 6.

Las comparaciones de vecinos mostraron que sus landmarks estaban mucho más próximos a otras clases que a sus equivalentes G/H/Y de train.

Ejemplos:

- G test tenía vecinos cercanos X/Z.
- H test tenía vecinos cercanos Z/V.
- Y test tenía vecinos cercanos Z/V.

La tasa de vecino de la misma clase para G/H/Y era `0%`.

## 6.2 Auditoría visual

La revisión de imágenes originales confirmó que no era simplemente un problema de `handedness`.

El problema era de **orientación/configuración respecto a la referencia adoptada**.

Ejemplo particularmente claro:

- G en train mostraba una orientación de la palma consistente con la referencia principal.
- G de sesión 6 presentaba la palma orientada en sentido opuesto.

El mismo fenómeno se observó en H y Y.

### Importante

No se consideró incorrecto utilizar mano izquierda o derecha por sí mismo.

La política adoptada fue:

- **Handedness diferente:** puede conservarse si representa correctamente la misma configuración de referencia.
- **Configuración/orientación gestual fuera de la referencia definida para el proyecto:** puede excluirse.

Esto es consistente con la limitación previamente definida para Sign2Sign: **el proyecto no busca cubrir todas las variantes posibles de ASL/LSM, sino evaluar respecto a una referencia concreta**.

---

# 7. Auditoría general de grupos LSM

La auditoría se amplió a todos los grupos para verificar que el problema no estuviera limitado únicamente a G/H/Y.

Se analizaron:

- `141` grupos clase+grupo.
- `85,883` muestras genuinas LSM.
- distancias entre centroides;
- ranking del grupo de la misma clase;
- ratio de distancia respecto al vecino global;
- sesión;
- split;
- revisión visual.

Se generaron indicadores:

- `OK`
- `REVIEW`
- `STRONG_REVIEW`

La sesión 6 sí contenía varios grupos sospechosos, pero **no debía descartarse completamente**.

El análisis mostró que muchos grupos de otras sesiones también tenían distancias elevadas debido a:

- handedness;
- variación natural;
- perspectiva;
- pequeñas diferencias de ejecución;
- grupos con estilos visuales diferentes.

Por lo tanto, la distancia geométrica se usó únicamente como **señal de revisión**, no como criterio automático de exclusión.

---

# 8. Manifiesto Canonical V2

Se construyó un manifiesto a nivel de grupo.

Datos:

- Muestras genuinas: `85,883`.
- Clases genuinas: `23`.
- Grupos: `141`.
- Overrides manuales aplicados: `12`.

Decisiones iniciales:

| Decisión | Grupos | Muestras |
|---|---:|---:|
| EXCLUDE | 3 | 1,950 |
| KEEP | 136 | 82,678 |
| REVIEW | 2 | 1,255 |

Los tres grupos marcados para exclusión fueron:

- `G / FRAM_G6`
- `H / FRAM_H6`
- `Y / FRAM_Y6`

Los grupos en revisión y otros casos prioritarios fueron inspeccionados visualmente.

---

# 9. Revisión final de grupos

Se revisaron específicamente grupos sospechosos como:

- `Q / Q_NORMAL`
- `K / P_NORMAL`
- `A / PHONE_BURST`
- `P / K7_FRAME`

Las imágenes mostraron que correspondían a sus clases, aunque presentaban:

- handedness diferente;
- variación de orientación;
- cercanía geométrica con otras letras;
- diferencias de estilo respecto a otros grupos.

La decisión final fue **KEEP** para estos grupos.

## 9.1 Resultado definitivo

| Decisión | Grupos | Muestras |
|---|---:|---:|
| KEEP | 138 | 83,933 |
| EXCLUDE | 3 | 1,950 |

Excluidos definitivamente:

- `FRAM_G6`
- `FRAM_H6`
- `FRAM_Y6`

No se descartaron otros grupos únicamente por handedness.

---

# 10. Construcción de LSM Canonical V2

Se creó:

```text
D:\Sign2SignData\processed\landmarks\lsm_landmarks_canonical_v2.csv
```

Contenido:

- Genuinas LSM conservadas: `83,933`.
- Provisionales E/L/Z: `2,759`.
- Total: `86,692`.

## 10.1 Provisionales

| Clase | Muestras |
|---|---:|
| E | 929 |
| L | 930 |
| Z | 900 |

Estas muestras permanecen con restricción:

```text
TRAIN_ONLY
```

Las clases genuinas quedan como:

```text
FREE
```

## 10.2 Auditoría de integridad

Se verificó:

- `img_norm NaN = 0`
- `img_norm Inf = 0`
- `world_norm NaN = 0`
- `world_norm Inf = 0`

También se confirmó que G6/H6/Y6 quedaron completamente ausentes del dataset Canonical V2.

---

# 11. Splits Canonical V2

Se decidió reconstruir completamente los splits.

## Política

```text
TRAIN      = resto de sesiones/grupos válidos
VALIDATION = sesión 3
TEST       = sesión 5
E/L/Z      = TRAIN ONLY
```

La sesión 6 dejó de ser el test principal porque contenía grupos que no representaban la referencia adoptada.

**No se eliminó toda la sesión 6:** únicamente se retiraron G6/H6/Y6; el resto pudo utilizarse según la política de split.

## 11.1 Conteos

| Split | Muestras | Porcentaje |
|---|---:|---:|
| Train | 42,743 | 49.30% |
| Validation | 14,791 | 17.06% |
| Test | 29,158 | 33.63% |
| **Total** | **86,692** | **100%** |

Grupos:

- Train: 73.
- Validation: 23.
- Test: 45.

### Auditorías

- Grupos con leakage entre splits: `0`.
- E/L/Z fuera de train: `0`.
- 23 clases genuinas presentes en train.
- 23 clases genuinas presentes en validation.
- 23 clases genuinas presentes en test.
- Landmarks finitos: correcto.

---

# 12. Conteo por clase en Canonical V2

| Clase | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| A | 1887 | 650 | 1296 | 3833 |
| B | 1935 | 650 | 1295 | 3880 |
| C | 1950 | 650 | 1300 | 3900 |
| D | 1515 | 650 | 1300 | 3465 |
| E | 929 | 0 | 0 | 929 |
| F | 1901 | 650 | 1300 | 3851 |
| G | 1299 | 650 | 1300 | 3249 |
| H | 1300 | 650 | 1300 | 3250 |
| I | 1877 | 650 | 1300 | 3827 |
| J | 1942 | 650 | 1300 | 3892 |
| K | 1802 | 650 | 1300 | 3752 |
| L | 930 | 0 | 0 | 930 |
| M | 1371 | 600 | 1264 | 3235 |
| N | 990 | 553 | 1285 | 2828 |
| O | 1950 | 650 | 1300 | 3900 |
| P | 2582 | 650 | 650 | 3882 |
| Q | 1918 | 639 | 1268 | 3825 |
| R | 1479 | 650 | 1300 | 3429 |
| S | 1784 | 650 | 1300 | 3734 |
| T | 1846 | 650 | 1300 | 3796 |
| U | 1753 | 650 | 1300 | 3703 |
| V | 1738 | 650 | 1300 | 3688 |
| W | 1939 | 649 | 1300 | 3888 |
| X | 1926 | 650 | 1300 | 3876 |
| Y | 1300 | 650 | 1300 | 3250 |
| Z | 900 | 0 | 0 | 900 |

Archivo generado:

```text
D:\Sign2SignData\processed\landmarks\lsm_landmarks_canonical_v2_splits.csv
```

---

# 13. Nuevo notebook de entrenamiento

Se decidió crear un notebook nuevo para Canonical V2 y conservar el notebook anterior como antecedente experimental.

Motivos:

- preservar la trazabilidad;
- evitar mezclar resultados V1 y V2;
- facilitar la documentación;
- Colab pierde memoria/variables cuando se desconecta el runtime.

Los datos necesarios se copiaron a:

```text
Sign2SignColab/data/
```

incluyendo:

```text
asl_landmarks_final.csv
lsm_landmarks_final.csv
lsm_landmarks_canonical_v2_splits.csv
```

---

# 14. Decisión de utilizar únicamente MLP

No se repitieron Random Forest en Canonical V2.

Razón:

- en V1 los MLP fueron considerablemente superiores;
- RF no aportaba evidencia útil adicional;
- se prefirió utilizar el tiempo de experimentación en representación y regularización.

Se entrenaron:

- V2-M1: MLP + `img_norm`.
- V2-M2: MLP + `world_norm`.
- V2-M3: MLP + `world_norm` con mayor regularización.

---

# 15. V2-M1 — MLP + img_norm

Resultados en validation:

| Métrica | Resultado |
|---|---:|
| Accuracy | 0.909878 |
| Precision macro genuina | 0.932767 |
| Recall macro genuina | 0.906848 |
| F1 macro genuina | 0.901460 |
| Top-3 | 0.980461 |
| Predicciones E/L/Z | 51 / 14,791 = 0.3448% |

Épocas:

- mejor `val_loss`: 2.
- mejor `val_accuracy`: 4.

Problemas importantes por clase:

- N: recall ≈ 0.396.
- R: recall ≈ 0.203.
- U: recall ≈ 0.598.
- D: precision ≈ 0.499.

Las curvas mostraron una brecha marcada entre train y validation.

---

# 16. V2-M2 — MLP + world_norm

Resultados en validation:

| Métrica | Resultado |
|---|---:|
| Accuracy | **0.938273** |
| Precision macro genuina | **0.948427** |
| Recall macro genuina | **0.934266** |
| F1 macro genuina | **0.933878** |
| Top-3 | **0.996011** |
| Predicciones E/L/Z | 105 / 14,791 = 0.7099% |

Épocas:

- mejor `val_loss`: 5.
- mejor `val_accuracy`: 3.

V2-M2 superó claramente a V2-M1 en:

- accuracy;
- F1 macro;
- recall;
- Top-3.

La representación `world_norm` resultó más adecuada para Canonical V2.

---

# 17. V2-M3 — prueba con mayor regularización

Se probó una variante más regularizada para intentar reducir la brecha train-validation.

Resultados:

| Métrica | Resultado |
|---|---:|
| Accuracy | 0.899736 |
| Precision macro genuina | 0.896492 |
| Recall macro genuina | 0.896500 |
| F1 macro genuina | 0.885015 |
| Top-3 | 0.990264 |
| Predicciones E/L/Z | 173 |

Épocas:

- mejor `val_loss`: 2.
- mejor `val_accuracy`: 4.

### Conclusión

La regularización adicional **empeoró la generalización**.

La brecha train-validation no debía interpretarse únicamente como falta de regularización; también reflejaba diferencia real entre dominios/sesiones.

V2-M3 fue descartado.

---

# 18. Selección definitiva antes de test

Comparación:

| Modelo | Accuracy val | F1 macro genuina | Top-3 |
|---|---:|---:|---:|
| V2-M1 img_norm | 90.99% | 90.15% | 98.05% |
| **V2-M2 world_norm** | **93.83%** | **93.39%** | **99.60%** |
| V2-M3 regularized | 89.97% | 88.50% | 99.03% |

Se congeló:

```text
V2-M2 MLP + world_norm
```

como ganador **antes de abrir test**.

El test no se utilizó para seleccionar arquitectura ni hiperparámetros.

---

# 19. Test final de V2-M2

Resultados:

| Métrica | Resultado |
|---|---:|
| Accuracy | **87.54%** |
| Precision macro genuina | 89.87% |
| Recall macro genuina | 87.77% |
| F1 macro genuina | **87.35%** |
| Top-3 | **99.22%** |
| Predicciones E/L/Z | 144 / 29,158 = 0.4939% |

Errores:

- Top-1: `3,633`.
- Top-1 incorrectos con clase correcta en Top-3: `3,406`.
- Clase correcta fuera de Top-3: solamente `227`.

## 19.1 Brecha validation-test

- Accuracy test - validation: `-6.29 puntos porcentuales`.
- F1 test - validation: `-6.03 puntos porcentuales`.

La caída fue importante, pero muy inferior al problema observado en V1 y coherente con un test formado por una sesión distinta.

## 19.2 Confusiones principales

| Real | Predicha | Casos |
|---|---|---:|
| N | M | 1095 |
| R | U | 766 |
| V | U | 387 |
| W | D | 269 |
| M | N | 218 |
| I | A | 128 |
| T | S | 112 |
| H | G | 94 |
| O | C | 86 |
| Q | Z | 79 |
| U | R | 56 |
| G | X | 45 |
| Q | X | 45 |
| K | L | 39 |
| R | V | 39 |

La confusión M/N siguió apareciendo offline, pero posteriormente se observó que en webcam ambas podían distinguirse correctamente bajo una ejecución adecuada.

---

# 20. Matriz de confusión final

Se generó y guardó una matriz de confusión:

- gradiente azul;
- números dentro de cada celda;
- líneas divisorias entre celdas;
- etiquetas de clase;
- título indicando Canonical V2 y V2-M2.

Esta versión se conservó como la representación visual preferida para documentación.

---

# 21. Modelo LSM congelado

A partir de los resultados de validation y test se congeló provisionalmente:

```text
V2-M2
MLP
world_norm
26 salidas
```

Modelo utilizado posteriormente en local:

```text
V2_M2_mlp_world_norm.keras
```

junto con su `StandardScaler` de `world_norm`.

El modelo utiliza:

```text
63 features = 21 landmarks × (x, y, z)
```

---

# 22. Integración local con webcam

El modelo y scaler se descargaron desde Drive y se integraron al repositorio local.

Pipeline:

```text
Webcam
  ↓
detect_with_cascade()
  ↓
MediaPipe Hand Landmarker
  ↓
world_norm
  ↓
StandardScaler
  ↓
V2-M2
  ↓
Top-3
```

## Decisiones técnicas

- No se aplica flip horizontal.
- Se reutiliza exactamente el pipeline de preprocessing usado en dataset.
- `detect_with_cascade()` tiene firma:

```python
detect_with_cascade(landmarker, original)
```

y devuelve un diccionario.
- Las predicciones se realizan únicamente cuando el usuario presiona `ENTER`.

---

# 23. Problema de rendimiento de webcam

La primera versión realizaba inferencia continuamente.

Resultado:

- webcam visualmente trabada;
- consumo innecesario;
- inferencia repetida aunque el usuario todavía estuviera preparando la seña.

Se cambió a:

```text
video fluido continuamente
+
inferencia solo al presionar ENTER
```

Esto coincide además con la interacción planeada del sistema final, donde cada letra se captura deliberadamente.

---

# 24. Smoke test manual LSM

Las primeras pruebas cualitativas mostraron buen comportamiento.

Se observaron correctamente, entre otras:

- A
- B
- G
- M
- N
- R
- U
- V
- Y

M/N podían presentar cercanía, pero en webcam se lograban reconocer de forma independiente con confianza razonable.

También se probó informalmente E/L/Z:

- E: muy estable.
- L: muy estable.
- Z: algo más variable, pero funcional.

---

# 25. Evaluación webcam controlada — 23 clases genuinas

Se construyó un evaluador controlado:

- 23 clases genuinas.
- 5 intentos por clase.
- 115 intentos totales.
- predicciones ocultas durante la captura;
- imagen exacta guardada por intento;
- métricas Top-1 y Top-3;
- métricas condicionadas a detección;
- métricas end-to-end.

## 25.1 Resultados

| Métrica | Resultado |
|---|---:|
| Intentos | 115 |
| Detectadas | 113 |
| No detectadas | 2 |
| Tasa de detección | **98.26%** |
| Top-1 dado detección | **86.73%** |
| Top-3 dado detección | **92.04%** |
| End-to-end Top-1 | **85.22%** |
| End-to-end Top-3 | **90.43%** |

El resultado Top-1 dado detección (`86.73%`) quedó muy cerca del test offline (`87.54%`).

Esto fue una señal positiva de transferencia desde dataset a webcam real.

## 25.2 Confusiones dominantes

```text
J → B : 5
R → U : 5
P → K : 3
P → Z : 2
```

La mayoría de las demás clases tuvieron 5/5 Top-1.

### N

N fue detectada únicamente 3/5 veces:

- 3 detectadas.
- 3 Top-1 correctas.
- 3 Top-3 correctas.

Por lo tanto, sus dos fallos fueron de **detección de mano**, no del clasificador.

---

# 26. Interpretación de J, P/K y R/U

## J

J contiene movimiento en su ejecución convencional.

El proyecto modela letras dinámicas mediante una **configuración estática de referencia**.

En la primera prueba webcam, la J realizada por el usuario no coincidía con el snapshot esperado por el dataset, produciendo:

```text
J → B
```

Esto motivó una prueba focalizada usando exactamente la configuración estática de referencia.

## P/K

P y K presentan configuraciones manuales muy cercanas dentro de LSM.

En un sistema de un solo frame, eliminar información dinámica/orientacional puede hacer que su representación sea altamente ambigua.

Por ello no se consideró apropiado intentar “forzar” una separación artificial sin investigar antes la geometría del dataset.

## R/U

R y U comparten una región geométrica cercana.

R depende fuertemente de que el cruce de los dedos se represente correctamente en los landmarks.

Este mismo tipo de dificultad ya se había observado en ASL.

---

# 27. Evaluación focalizada J/K/P/R/U

Se realizó una prueba adicional:

```text
J
K
P
R
U
```

con:

```text
10 intentos por clase
50 intentos totales
```

## 27.1 Resultados globales

| Métrica | Resultado |
|---|---:|
| Intentos | 50 |
| Detectadas | 44 |
| No detectadas | 6 |
| Tasa detección | 88.00% |
| Top-1 dado detección | 56.82% |
| Top-3 dado detección | 79.55% |
| End-to-end Top-1 | 50.00% |
| End-to-end Top-3 | 70.00% |

Estas métricas **no deben compararse directamente con la prueba general**, porque esta evaluación seleccionó deliberadamente las clases más problemáticas.

## 27.2 Por clase

| Clase | Detectadas | Top-1 | Top-3 | Predicción dominante |
|---|---:|---:|---:|---|
| J | 4 | 3 | 4 | J |
| K | 10 | 6 | 8 | K |
| P | 10 | 0 | 3 | X |
| R | 10 | **10** | **10** | R |
| U | 10 | 6 | **10** | U |

Confusiones:

```text
U → W : 4
P → X : 4
P → N : 2
P → K : 2
K → Z : 2
P → Z : 2
J → B : 1
K → U : 1
K → X : 1
```

---

# 28. Conclusiones del focalizado

## R

Pasó de:

```text
R → U 5/5
```

en la prueba general a:

```text
10/10 Top-1
```

en la focalizada.

Conclusión:

- el problema anterior estaba fuertemente relacionado con ejecución/configuración;
- el modelo sí puede reconocer R.

## J

Cuando MediaPipe detectó la mano:

- 3/4 Top-1.
- 4/4 Top-3.

Conclusión:

- utilizar la configuración estática de referencia resolvió casi completamente el problema de clasificación;
- la dificultad restante fue principalmente de detección de mano.

## U

- 6/10 Top-1.
- 10/10 Top-3.
- 4 casos `U → W`.

Conclusión:

- U y W se encuentran cerca;
- la clase real permanece consistentemente dentro del Top-3.

## K

- 6/10 Top-1.
- 8/10 Top-3.

Conclusión:

- funcionamiento razonable;
- cierta inestabilidad respecto a clases geométricamente cercanas.

## P

- 0/10 Top-1.
- 3/10 Top-3.

Era el único caso que aún necesitaba investigación específica.

---

# 29. Evaluación independiente de E/L/Z

Las clases provisionales se evaluaron por separado para no mezclarlas con las métricas genuinas LSM.

Protocolo:

```text
E × 10
L × 10
Z × 10
```

Total:

```text
30 intentos
```

## 29.1 Resultado

| Métrica | Resultado |
|---|---:|
| Detectadas | **30/30** |
| Detection rate | **100%** |
| Top-1 dado detección | **100%** |
| Top-3 dado detección | **100%** |
| End-to-end Top-1 | **100%** |
| End-to-end Top-3 | **100%** |

## 29.2 Por clase

| Clase | Top-1 | Top-3 | Probabilidad media clase real | Rank medio |
|---|---:|---:|---:|---:|
| E | 10/10 | 10/10 | **0.993424** | 1.0 |
| L | 10/10 | 10/10 | **0.984053** | 1.0 |
| Z | 10/10 | 10/10 | **0.866800** | 1.0 |

No hubo errores Top-1.

### Interpretación

- E y L son extremadamente estables.
- Z tiene menor confianza media, coherente con su naturaleza dinámica/estática provisional, pero permaneció siempre en Top-1.
- Las tres clases son funcionalmente utilizables para esta etapa.
- **No se incorporan estas métricas a la métrica principal del clasificador LSM genuino.**

---

# 30. Auditoría específica P vs K

Para comprender el comportamiento de P se compararon las 10 capturas webcam de P con muestras TRAIN de:

- P.
- K.

La comparación se realizó usando exactamente:

```text
world_norm
+
StandardScaler de V2-M2
+
distancia euclidiana
```

Para cada captura se buscaron los vecinos P y K más cercanos.

## 30.1 Resultado principal

```text
Capturas detectadas: 10

Referencia geométrica más cercana:
K = 10/10
P = 0/10
```

Distancias medias:

```text
Distancia media a P = 4.4110
Distancia media a K = 3.0259
```

Detalle:

| Captura | Predicción MLP | d(P) | d(K) | Más cercana |
|---|---|---:|---:|---|
| P_01 | X | 4.166444 | 3.332441 | K |
| P_02 | N | 4.407212 | 3.032585 | K |
| P_03 | Z | 5.233953 | 2.411187 | K |
| P_04 | K | 5.071019 | 2.537266 | K |
| P_05 | K | 4.979630 | 2.723861 | K |
| P_06 | N | 4.644272 | 2.892490 | K |
| P_07 | Z | 4.675075 | 3.443956 | K |
| P_08 | X | 3.490186 | 3.478470 | K |
| P_09 | X | 3.726183 | 3.249253 | K |
| P_10 | X | 3.716469 | 3.157438 | K |

---

# 31. Interpretación final de P/K

Las diez P realizadas en webcam se encontraban geométricamente más cerca de la distribución TRAIN de K que de la de P.

Esto indica dos factores combinados:

1. P y K son altamente similares bajo la representación estática utilizada.
2. La ejecución webcam de P se aproxima más a la configuración que el dataset codificó como K.

No se consideró adecuado retrenar el clasificador únicamente para forzar una separación.

La diferencia puede depender de información que el sistema actual no modela explícitamente:

- movimiento;
- orientación;
- transición;
- configuración temporal.

Esto se conserva como **limitación conocida del sistema estático**.

Una formulación útil para documentación técnica es:

> La letra P presentó dificultades durante la evaluación en webcam. Un análisis de proximidad en el espacio de características `world_norm` mostró que las diez capturas realizadas para P se encontraban más próximas a muestras de entrenamiento correspondientes a K que a las propias muestras P. Este comportamiento es consistente con la elevada similitud entre ambas configuraciones bajo la representación estática utilizada por el sistema. Debido a que el proyecto no modela explícitamente el componente dinámico de las señas, esta ambigüedad se conserva como una limitación conocida y podrá ser tratada posteriormente por el módulo de construcción y corrección textual.

---

# 32. Estado final del clasificador LSM

Clasificador congelado para la siguiente fase:

```text
V2-M2
MLP
world_norm
26 clases
```

## Evidencia principal

### Offline

```text
TEST Canonical V2
Accuracy Top-1: 87.54%
F1 macro genuina: 87.35%
Top-3: 99.22%
```

### Webcam — 23 clases genuinas

```text
Detection rate: 98.26%
Top-1 dado detección: 86.73%
Top-3 dado detección: 92.04%
End-to-end Top-1: 85.22%
End-to-end Top-3: 90.43%
```

### Webcam — provisionales

```text
E/L/Z:
30/30 detecciones
30/30 Top-1
30/30 Top-3
```

### Focalizado

- R: 10/10 Top-1.
- J: funcional cuando se utiliza la referencia estática esperada.
- K: razonable, con ambigüedad.
- U: siempre en Top-3.
- P: ambigüedad explicada mediante auditoría geométrica.

---

# 33. Lecciones técnicas importantes

## 33.1 Un split correcto importa tanto como el modelo

Un test puede producir resultados aparentemente muy malos si contiene una variante fuera de la referencia que el sistema pretende cubrir.

El primer 72.71% no significaba simplemente que el MLP fuera malo.

La auditoría reveló un cambio fuerte de dominio en G/H/Y de sesión 6.

## 33.2 No excluir datos solamente por distancia

Los centroides y nearest-neighbors sirven como herramientas de auditoría.

No deben utilizarse como regla automática de eliminación.

Handedness, perspectiva y variación natural pueden generar distancias grandes sin que la seña sea incorrecta.

## 33.3 Handedness no equivale a orientación incorrecta

Se decidió aceptar tanto mano izquierda como derecha cuando:

- la configuración de la seña se conserva;
- la orientación corresponde a una variante aceptable de la referencia.

Lo que se excluyó en G6/H6/Y6 fue una configuración/orientación fuera de la referencia adoptada, no el simple hecho de usar otra mano.

## 33.4 Validation y test deben representar dominios diferentes pero válidos

El objetivo no es hacer validation/test artificialmente fáciles.

El objetivo es que ambos sean:

- independientes;
- sin leakage;
- diferentes de train;
- pero pertenecientes al alcance real definido para el proyecto.

## 33.5 La brecha train-validation no siempre se arregla con dropout

La variante más regularizada V2-M3 rindió peor.

Parte de la brecha provenía del cambio de sesión/dominio, no únicamente de overfitting.

## 33.6 Top-3 es extremadamente valioso

En test Canonical V2:

```text
Top-1 = 87.54%
Top-3 = 99.22%
```

De `3,633` errores Top-1:

```text
3,406
```

todavía tenían la clase correcta dentro del Top-3.

Esto confirma la utilidad futura de conservar:

- Top-k;
- probabilidades;
- margen;
- contexto de palabra;

para el módulo de corrección textual.

## 33.7 No todas las confusiones deben resolverse en el clasificador

Casos como:

- P/K;
- R/U;
- M/N;

pueden depender de diferencias geométricas muy pequeñas.

El sistema completo dispone posteriormente de:

- construcción textual;
- vocabulario;
- contexto;
- corrección restringida;
- LLM.

Por ello no es necesario exigir que el clasificador resuelva todas las ambigüedades aisladamente.

## 33.8 Las letras dinámicas siguen siendo una limitación explícita

El proyecto trabaja con snapshots estáticos.

Por lo tanto letras con movimiento como J/Z y diferencias temporales como K/P pueden perder información discriminativa.

Esta limitación es intencional y debe documentarse.

## 33.9 Evaluar webcam de manera controlada es indispensable

El test offline no es suficiente.

La evaluación webcam permitió separar:

```text
detección de mano
vs
clasificación
```

Ejemplo:

- N tuvo 2 fallos de detección.
- Cuando fue detectada, las capturas fueron clasificadas correctamente.

## 33.10 La inferencia debe ejecutarse bajo demanda

La versión con inferencia continua hacía que la webcam se sintiera trabada.

Para la interacción final:

```text
ENTER → capturar → detectar → clasificar
```

es más eficiente y además coincide con el alcance del sistema.

---

# 34. Archivos y scripts relevantes de esta etapa

Entre los scripts utilizados/generados se encuentran:

```text
scripts\build_lsm_canonical_v2.py
scripts\evaluate_lsm_webcam.py
scripts\evaluate_lsm_webcam_focused.py
scripts\evaluate_lsm_webcam_provisional.py
scripts\audit_lsm_p_vs_k.py
```

Datasets principales:

```text
D:\Sign2SignData\processed\landmarks\
    lsm_landmarks_final.csv
    lsm_landmarks_canonical_v2.csv
    lsm_landmarks_canonical_v2_splits.csv
```

Auditoría Canonical V2:

```text
D:\Sign2SignData\inventory\lsm_canonical_v2\
```

Evaluación webcam general:

```text
D:\Sign2SignData\inventory\webcam_eval\lsm\
```

Evaluación focalizada:

```text
D:\Sign2SignData\inventory\webcam_eval\lsm_focused\
```

Evaluación provisional:

```text
D:\Sign2SignData\inventory\webcam_eval\lsm_provisional\
```

Auditoría P/K:

```text
D:\Sign2SignData\inventory\webcam_eval\lsm_p_vs_k_audit\
```

---

# 35. Errores técnicos encontrados durante la implementación

## Colab

Al desconectar el runtime:

- se pierden imports;
- se pierden variables;
- Drive debe montarse nuevamente;
- los archivos persisten únicamente si fueron guardados en Drive.

Por esta razón Canonical V2 se trabajó en un notebook nuevo y autocontenido.

## Construcción de Canonical V2

Durante el desarrollo aparecieron problemas por:

- identificación inicial incorrecta de E/L/Z provisionales;
- columnas opcionales como `notes` no presentes;
- diferencias entre esquemas de archivos intermedios.

Se corrigieron haciendo el script más robusto:

- detección explícita de fuente provisional;
- no asumir columnas opcionales;
- auditorías posteriores a cada construcción.

## Integración webcam

Se corrigieron diferencias reales de API:

- `detect_with_cascade()` devuelve `dict`.
- Firma real: `(landmarker, original)`.
- `create_hand_landmarker()` recibe la ruta como `str`.
- Se reutiliza la función de extracción normalizada correspondiente a la representación del modelo.
- No se aplica flip.

---

# 36. Decisiones que NO deben revertirse sin nueva evidencia

1. **Mantener ASL y LSM como modelos separados.**
2. **Mantener V2-M2 como clasificador LSM actual.**
3. **Utilizar `world_norm` para LSM V2-M2.**
4. **E/L/Z son provisionales y TRAIN_ONLY.**
5. **No mezclar E/L/Z con métricas genuinas LSM.**
6. **G6/H6/Y6 permanecen excluidos del Canonical V2.**
7. **No eliminar grupos únicamente por handedness.**
8. **No hacer flip horizontal en webcam.**
9. **No ajustar el modelo usando el test final.**
10. **No reentrenar solamente para corregir P/K sin nueva evidencia.**
11. **Conservar Top-3 y probabilidades para etapas posteriores.**
12. **Inferencia principal por captura deliberada con ENTER.**

---

# 37. Cierre de la fase de reconocimiento LSM

Con Canonical V2 se logró:

- auditar el dominio del dataset;
- identificar grupos fuera de referencia;
- reconstruir un dataset consistente;
- crear splits sin leakage;
- entrenar un MLP competitivo;
- seleccionar el modelo únicamente con validation;
- abrir test una sola vez después de selección;
- validar el comportamiento con webcam;
- separar fallos de detección y clasificación;
- investigar las clases difíciles;
- validar por separado E/L/Z;
- explicar cuantitativamente la ambigüedad P/K.

Por lo tanto, **el módulo de reconocimiento de letras LSM se considera cerrado para esta etapa del proyecto**.

El modelo queda congelado provisionalmente como:

```text
LSM Canonical V2 — V2-M2 — MLP + world_norm
```

---

# 38. Siguiente fase

El siguiente módulo a desarrollar es el:

# Constructor textual

Objetivo general:

```text
captura de seña
    ↓
letra clasificada
    ↓
confirmación
    ↓
acumulación de caracteres
    ↓
palabras / frase
```

La construcción textual será el puente entre:

```text
reconocimiento de letras
```

y:

```text
corrección / traducción / vocabulario / LLM
```

Este documento termina exactamente antes de comenzar la implementación formal del constructor textual.
