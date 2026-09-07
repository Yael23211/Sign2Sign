# Sign2Sign — Entrenamiento, selección, validación e integración del reconocedor ASL

**Estado:** ASL congelado como baseline funcional con limitaciones conocidas  
**Periodo documentado:** desde el entrenamiento en Google Colab hasta las evaluaciones y diagnósticos en webcam  
**Objetivo de este documento:** conservar decisiones, resultados, aprendizajes, problemas encontrados y conclusiones técnicas para reutilizarlos en la documentación final del TT y como referencia durante la integración del sistema completo.

---

# 1. Contexto de esta fase

El objetivo de esta etapa fue construir y validar el reconocedor de letras ASL de Sign2Sign.

El flujo completo evaluado fue:

```text
imagen / webcam
    ↓
preprocesamiento
    ↓
MediaPipe Hand Landmarker
    ↓
21 landmarks de mano
    ↓
normalización
    ↓
vector de 63 características
    ↓
StandardScaler
    ↓
modelo de clasificación
    ↓
Top-1 / Top-3 + probabilidades
```

El clasificador ASL trabaja con las 26 letras:

```text
A B C D E F G H I J K L M
N O P Q R S T U V W X Y Z
```

En el alcance actual del proyecto, las letras con movimiento se mantienen como una aproximación estática. Esta decisión constituye una limitación conocida del prototipo.

---

# 2. Dataset ASL utilizado

El dataset ASL original fue almacenado fuera del repositorio, dentro de:

```text
D:\Sign2SignData\raw\asl\ASL_original.zip
```

Las imágenes originales permanecen comprimidas para evitar duplicar almacenamiento.

El dataset procesado con landmarks se encuentra en:

```text
D:\Sign2SignData\processed\landmarks\asl_landmarks_final.csv
```

## 2.1 Tamaño original

Se trabajó con:

```text
26,000 imágenes
26 clases
1,000 imágenes por letra
```

## 2.2 Extracción de landmarks

Después del pipeline MediaPipe + cascade:

```text
Filas totales:        26,000
Landmarks válidos:    24,778
Fallos de extracción:  1,222
```

Esto representa aproximadamente:

```text
95.30 % de imágenes con landmarks válidos
```

## 2.3 Split final

El split utilizado para entrenamiento y selección fue:

| Split | Muestras | Porcentaje aproximado |
|---|---:|---:|
| Train | 20,061 | 80.96 % |
| Validation | 2,234 | 9.02 % |
| Test | 2,483 | 10.02 % |
| **Total válido** | **24,778** | **100 %** |

Las 26 clases están presentes en train, validation y test.

El conjunto de test fue preservado y no se utilizó durante la selección de modelos.

---

# 3. Pipeline de landmarks

El pipeline compartido vive en:

```text
src/landmarks/pipeline.py
```

El objetivo desde el inicio fue utilizar el mismo procesamiento tanto para:

- extracción masiva offline;
- entrenamiento;
- evaluación;
- webcam;
- integración final.

Esto evita discrepancias entre entrenamiento e inferencia.

## 3.1 Funciones principales reutilizadas

Entre las funciones relevantes del pipeline se encuentran:

```python
create_hand_landmarker(...)
detect_with_cascade(...)
image_landmarks_to_arrays(...)
normalize_points(...)
```

También existen transformaciones auxiliares como:

```python
center_crop(...)
center_crop_4_3(...)
letterbox(...)
transform_image(...)
```

---

# 4. Cascade de detección

Durante las pruebas iniciales se observó que MediaPipe no detectaba todas las imágenes del dataset con la misma facilidad.

Se probaron distintas transformaciones para aumentar la tasa de detección.

El orden final del cascade fue:

```text
1. baseline
2. center_4_3
3. crop_85
4. crop_70
5. letterbox
```

El `stretch` fue probado, pero no se conservó en el cascade final porque no aportó rescates incrementales relevantes en la muestra evaluada.

## 4.1 Resultado de una muestra de 150 imágenes

Antes del cascade:

```text
103 / 150 = 68.67 %
```

Después del cascade:

```text
132 / 150 = 88.00 %
```

Esto justificó mantener un pipeline escalonado para la extracción masiva.

---

# 5. Normalización

Se trabajó con dos representaciones de entrada:

```text
img_norm
world_norm
```

Cada una contiene:

```text
21 landmarks × 3 coordenadas = 63 características
```

La normalización fue verificada mediante pruebas numéricas.

Se comprobó que:

- la muñeca queda utilizada como referencia;
- el origen se desplaza correctamente;
- la escala relativa se mantiene;
- no se generaron errores de normalización;
- no existen vectores incompletos ni valores NaN/Inf en el conjunto final válido.

---

# 6. Entorno de entrenamiento

El entrenamiento de modelos se realizó en **Google Colab**.

La razón principal fue evitar instalar frameworks pesados localmente y mantener el equipo local dedicado a:

- extracción;
- evaluación;
- webcam;
- integración.

Los artefactos entrenados se guardaron posteriormente en Google Drive y se copiaron al repositorio/local para inferencia.

---

# 7. Estrategia de selección de modelos ASL

Se definieron cuatro experimentos iniciales:

```text
A1 = Random Forest + img_norm
A2 = Random Forest + world_norm
A3 = MLP + img_norm
A4 = MLP + world_norm
```

La comparación se realizó únicamente sobre validation.

El test permaneció cerrado hasta congelar el modelo ganador.

---

# 8. Experimento A1 — Random Forest + img_norm

Resultados sobre validation:

| Métrica | Resultado |
|---|---:|
| Accuracy | 0.977171 |
| Precision macro | 0.977172 |
| Recall macro | 0.977011 |
| F1 macro | 0.976978 |
| Top-3 accuracy | 0.991943 |

Interpretación:

- buen baseline;
- desempeño alto;
- demostró que `img_norm` contiene suficiente información discriminativa;
- fue superado posteriormente por el MLP.

---

# 9. Experimento A2 — Random Forest + world_norm

Resultados sobre validation:

| Métrica | Resultado |
|---|---:|
| Accuracy | 0.972695 |
| Precision macro | 0.972777 |
| Recall macro | 0.972548 |
| F1 macro | 0.972542 |
| Top-3 accuracy | 0.990152 |

Conclusión:

```text
img_norm > world_norm
```

al menos para Random Forest y para este dataset.

---

# 10. Experimento A3 — MLP + img_norm

Este modelo resultó ganador.

## 10.1 Arquitectura

```text
Input: 63
    ↓
Dense 128
    ↓
Batch Normalization
    ↓
ReLU
    ↓
Dropout 0.20
    ↓
Dense 64
    ↓
Batch Normalization
    ↓
ReLU
    ↓
Dropout 0.20
    ↓
Dense 26
    ↓
Softmax
```

## 10.2 Configuración de entrenamiento

Se utilizó:

```text
Optimizer: Adam
Learning rate inicial: 1e-3
Batch size: 128
```

Callbacks:

```text
EarlyStopping sobre val_loss
restore_best_weights=True

ReduceLROnPlateau
```

El escalado se realizó mediante:

```text
StandardScaler
```

Regla metodológica importante:

```text
scaler.fit(...) SOLO sobre train
validation/test → scaler.transform(...)
```

Nunca se ajustó el scaler con validation o test.

## 10.3 Resultados validation

| Métrica | Resultado |
|---|---:|
| Accuracy | 0.987914 |
| Precision macro | 0.988058 |
| Recall macro | 0.987983 |
| F1 macro | 0.987941 |
| Top-3 accuracy | 0.994181 |

Mejor época:

```text
37
```

Aproximadamente:

```text
Train loss: 0.04979
Validation loss: 0.06134

Train accuracy: 98.594 %
Validation accuracy: 98.791 %
```

Las curvas se consideraron saludables y no mostraron un sobreajuste preocupante.

---

# 11. Errores de A3 en validation

Se registraron:

```text
27 errores Top-1
```

De éstos:

```text
14 → clase correcta presente en Top-3
13 → clase correcta fuera del Top-3
```

Confusiones repetidas observadas:

```text
A → I : 2
W → V : 2
```

La mayoría del resto de errores fueron aislados.

También se detectaron errores con alta confianza Softmax.

Esto produjo un aprendizaje importante:

> Un umbral basado únicamente en la probabilidad máxima de Softmax no es suficiente para detectar todos los errores.

Por ello se dejó anotado investigar posteriormente:

- margen Top-1 vs Top-2;
- entropía;
- calibración;
- consenso temporal/multiframe;
- uso del Top-k en la capa de corrección.

---

# 12. Experimento A4 — MLP + world_norm

Resultados sobre validation:

| Métrica | Resultado |
|---|---:|
| Accuracy | 0.982990 |
| Precision macro | 0.983215 |
| Recall macro | 0.982893 |
| F1 macro | 0.982915 |
| Top-3 accuracy | 0.992838 |

Mejor época:

```text
33
```

Aproximadamente:

```text
Train accuracy: 98.1905 %
Validation accuracy: 98.299 %
```

Las curvas también fueron saludables.

Sin embargo:

```text
A3 > A4
```

---

# 13. Selección formal del modelo ASL

Después de A1-A4 se concluyó:

```text
MLP > Random Forest
img_norm > world_norm
```

Por lo tanto:

```text
A3 = MLP + img_norm
```

fue congelado como candidato seleccionado antes de abrir test.

Se decidió no continuar probando arquitecturas únicamente para intentar subir décimas porcentuales, ya que A3 ofrecía:

- alta precisión;
- arquitectura pequeña;
- facilidad de despliegue;
- inferencia rápida;
- resultados reproducibles;
- suficiente capacidad para el alcance académico del TT.

---

# 14. Evaluación final sobre test

Una vez congelado A3, se abrió el conjunto test original.

Total:

```text
2,483 muestras
```

Resultados:

| Métrica | Resultado |
|---|---:|
| Accuracy | 0.991542 |
| Precision macro | 0.991615 |
| Recall macro | 0.991411 |
| F1 macro | 0.991468 |
| Top-3 accuracy | 0.997181 |

En porcentaje:

```text
Top-1: 99.1542 %
Top-3: 99.7181 %
```

Errores Top-1:

```text
21
```

De éstos:

```text
14 → recuperados dentro del Top-3
7  → fuera del Top-3
```

No se observó colapso de ninguna clase.

Todas las clases presentaron F1 alto, aproximadamente superior al 98 %, con varias clases perfectas.

Confusiones repetidas destacables:

```text
E → M : 2
W → B : 2
```

También aparecieron errores de alta confianza, por ejemplo:

```text
I → S ≈ 96 %
E → M ≈ 86 %
L → G ≈ 80 %
```

Esto reforzó nuevamente la conclusión de que Softmax máximo no debe interpretarse automáticamente como una medida perfecta de certeza.

---

# 15. Artefactos guardados del entrenamiento ASL

Entre los artefactos conservados se encuentran:

```text
A3_mlp_img_norm.keras
A4_mlp_world_norm.keras

scaler_img_norm.joblib
scaler_world_norm.joblib

A3_training_history.csv
A4_training_history.csv

A3_mlp_img_norm_confusion.png
A3_classification_report.csv
A3_validation_errors.csv
A3_confusion_pairs.csv

A3_validation_vs_test.csv

A3_test_metrics.json
A3_test_classification_report.csv
A3_test_confusion_matrix.png
A3_test_errors.csv
A3_test_confusion_pairs.csv
```

Notebook:

```text
01_asl_model_selection.ipynb
```

Los artefactos principales para inferencia local son:

```text
models\asl\A3_mlp_img_norm.keras
models\asl\scaler_img_norm.joblib
```

---

# 16. Integración local del modelo

Se creó un smoke test local:

```text
scripts\test_asl_model_webcam.py
```

El flujo final quedó:

```text
webcam
   ↓
detect_with_cascade()
   ↓
MediaPipe result
   ↓
image_landmarks_to_arrays()
   ↓
img_norm (63)
   ↓
scaler.transform()
   ↓
A3
   ↓
Top-3
```

---

# 17. Problemas de integración encontrados

Durante la integración se descubrieron varias incompatibilidades prácticas.

## 17.1 create_hand_landmarker y pathlib

Localmente:

```python
create_hand_landmarker(Path(...))
```

provocó:

```text
AttributeError:
'WindowsPath' object has no attribute 'encode'
```

Solución:

```python
create_hand_landmarker(str(path))
```

---

## 17.2 Salida real de detect_with_cascade

Se confirmó que:

```python
detect_with_cascade(...)
```

devuelve un `dict`, no una tupla.

Claves utilizadas:

```text
ok
variant
processed
result
```

Los scripts posteriores fueron ajustados a este contrato real.

---

## 17.3 Firma de image_landmarks_to_arrays

Para mantener compatibilidad con el pipeline local se utilizó inspección defensiva de firma con:

```python
inspect.signature(...)
```

Esto permitió soportar variantes donde la función pudiera recibir:

```text
landmarks
```

o:

```text
landmarks, width, height
```

---

## 17.4 Versión de scikit-learn

El scaler fue serializado con:

```text
scikit-learn 1.6.1
```

El equipo local llegó a tener:

```text
scikit-learn 1.9.0
```

Esto generó:

```text
InconsistentVersionWarning
```

Se decidió alinear el entorno local con:

```text
scikit-learn==1.6.1
```

para evitar incompatibilidades de serialización y asegurar reproducibilidad.

---

## 17.5 No utilizar flip horizontal

Se decidió no aplicar `flip` horizontal en webcam.

Razón:

- se quiere conservar la misma representación y orientación utilizada durante extracción y entrenamiento;
- pruebas previas mostraron que `flip` podía ayudar algunas muestras pero perjudicar otras;
- no se desea alterar el dominio de inferencia sin justificarlo.

---

# 18. Smoke test informal con webcam

Después de corregir la integración, el modelo comenzó a reconocer letras reales mediante webcam.

Durante pruebas manuales se observaron buenos resultados en varias letras, entre ellas:

```text
A
B
D
K
M
N
P
S
U
V
W
Y
```

cuando la configuración se realizaba de manera suficientemente estable.

Se observó sensibilidad a la pose en algunas letras, especialmente:

```text
D
S
```

Una consideración importante es que la persona que realizó las pruebas no es usuario habitual de ASL y reprodujo las configuraciones a partir de referencias visuales.

Por lo tanto, las evaluaciones webcam mezclan dos factores:

```text
robustez del modelo
+
calidad de ejecución de la seña
```

Esta limitación debe mantenerse explícita en la documentación.

---

# 19. Idea futura: captura multiframe

Durante las pruebas surgió una posible mejora:

Al presionar ENTER, en lugar de utilizar únicamente un frame:

```text
capturar 5–10 frames
durante aproximadamente 300 ms
    ↓
predecir cada frame
    ↓
agregar por voto / promedio / consenso
```

Esto permitiría estabilizar la predicción sin convertir A3 en un modelo temporal.

Decisión actual:

> No implementar esta mejora antes de cuantificar correctamente el baseline single-frame.

---

# 20. Evaluación controlada general en webcam

Se creó:

```text
scripts\evaluate_asl_webcam.py
```

Características de la prueba:

- letras A-Z;
- 5 intentos por letra;
- 130 intentos totales;
- predicciones ocultas durante la prueba;
- imagen exacta de cada intento guardada;
- CSV de resultados;
- resumen;
- métricas por clase;
- matriz de confusión.

Ruta de la primera evaluación:

```text
D:\Sign2SignData\inventory\webcam_eval\asl\
run_20260906_121444
```

---

# 21. Resultados de evaluación general webcam

Total:

```text
26 letras × 5 intentos = 130
```

Detección:

```text
130 / 130 = 100 %
```

No detecciones:

```text
0
```

Clasificación:

```text
Top-1 dado detección: 73.85 %
Top-3 dado detección: 93.08 %
```

Como la detección fue 100 %:

```text
End-to-end Top-1 = 73.85 %
End-to-end Top-3 = 93.08 %
```

En números:

```text
Top-1 correctos: 96 / 130
Errores Top-1:   34 / 130
```

De los 34 errores:

```text
25 tenían la clase verdadera dentro del Top-3
9 estaban fuera del Top-3
```

---

# 22. Resultado por clase — prueba general

| Clase | Top-1 | Top-3 |
|---|---:|---:|
| A | 5/5 | 5/5 |
| B | 5/5 | 5/5 |
| C | 5/5 | 5/5 |
| D | 5/5 | 5/5 |
| E | 5/5 | 5/5 |
| F | 5/5 | 5/5 |
| G | 5/5 | 5/5 |
| H | 5/5 | 5/5 |
| I | 5/5 | 5/5 |
| J | 5/5 | 5/5 |
| K | 3/5 | 5/5 |
| L | 5/5 | 5/5 |
| M | 5/5 | 5/5 |
| N | 5/5 | 5/5 |
| O | 2/5 | 3/5 |
| P | 1/5 | 5/5 |
| Q | 5/5 | 5/5 |
| R | 0/5 | 0/5 |
| S | 5/5 | 5/5 |
| T | 0/5 | 5/5 |
| U | 0/5 | 5/5 |
| V | 5/5 | 5/5 |
| W | 5/5 | 5/5 |
| X | 0/5 | 4/5 |
| Y | 3/5 | 5/5 |
| Z | 2/5 | 4/5 |

Resultado relevante:

```text
18 / 26 clases fueron perfectas Top-1 (5/5)
```

---

# 23. Confusiones principales — prueba general

Las confusiones más repetidas fueron:

```text
U → W : 5
X → D : 5
P → G : 4
T → A : 4
O → C : 3
Z → P : 3
R → U : 2
R → V : 2
K → U : 2
Y → L : 2
T → N : 1
R → W : 1
```

---

# 24. Interpretación preliminar por clases problemáticas

## K

En la primera prueba:

```text
K → U en 2/5
```

pero K siempre apareció dentro del Top-3.

Esto indicó inicialmente una frontera K/U.

---

## O

En la primera prueba:

```text
O → C
```

fue una confusión importante.

Se observó que O/C depende fuertemente del grado de cierre de los dedos.

---

## P

En la primera prueba:

```text
P → G
```

P permanecía generalmente en Top-3.

Se sospechó una fuerte sensibilidad a la orientación.

---

## R

Primera prueba:

```text
R → U / V / W
```

R nunca apareció en Top-3.

Desde ese momento se identificó como la clase más preocupante.

---

## T

Primera prueba:

```text
T → A / N
```

T siempre estaba dentro de Top-3.

Se sospechó sensibilidad a la posición interna del pulgar.

---

## U

Primera prueba:

```text
U → W 5/5
```

pero U siempre estaba en Top-3.

Se observó experimentalmente que pequeñas diferencias en la colocación del pulgar modifican mucho la predicción.

---

## X

Primera prueba:

```text
X → D 5/5
```

X apareció en Top-3 en 4/5.

La hipótesis fue que la curvatura del índice no se estaba representando con suficiente claridad.

---

## Y

Primera prueba:

```text
Y → L en 2/5
```

pero Y siempre apareció en Top-3.

Se consideró una frontera recuperable.

---

## Z

Primera prueba:

```text
Z → P
```

Se recordó que Z es dinámica en ASL estándar, pero dentro del alcance del proyecto se trata mediante una aproximación estática.

---

# 25. Evaluación focalizada

Se decidió realizar una segunda prueba únicamente sobre las letras problemáticas.

Se creó:

```text
scripts\evaluate_asl_webcam_focused.py
```

Clases:

```text
K O P R T U X Y Z
```

Intentos:

```text
10 por clase
90 totales
```

Las predicciones permanecieron ocultas durante la captura.

---

# 26. Resultados de prueba focalizada

Detección:

```text
90 / 90 = 100 %
```

Resultados globales de la ejecución original:

```text
Top-1: 27.78 %
Top-3: 33.33 %
```

Sin embargo, posteriormente se detectó que los 10 intentos etiquetados como K fueron realizados accidentalmente con una configuración de P.

Por lo tanto:

> Los resultados de K en esta prueba son inválidos y no deben utilizarse para evaluar K.

---

# 27. Resultados focalizados válidos

Excluyendo K:

| Letra | Top-1 | Top-3 | Lectura inicial |
|---|---:|---:|---|
| O | 0/10 | 0/10 | problema sistemático |
| P | 0/10 | 0/10 | problema sistemático |
| R | 0/10 | 0/10 | problema sistemático |
| T | 0/10 | 0/10 | problema sistemático |
| U | 5/10 | 10/10 | frontera U/V |
| X | 0/10 | 0/10 | problema sistemático |
| Y | 10/10 | 10/10 | recuperada |
| Z | 10/10 | 10/10 | recuperada |

Excluyendo K:

```text
Top-1 = 25 / 80 = 31.25 %
Top-3 = 30 / 80 = 37.50 %
```

Importante:

> Este porcentaje NO sustituye el resultado general de 73.85 %, porque esta prueba fue deliberadamente sesgada hacia las clases problemáticas.

---

# 28. Confusiones de la prueba focalizada

Patrones principales:

```text
P → Q : 10/10

K inválida → P : 7/10
(el usuario estaba realizando P)

T → P : 6
T → N : 4

U → V : 5
U → U : 5

X → Z : 5
X → G : 3
X → P : 2

R → V : 4
R → U : 4
R → W : 2

O → C : 4
O → D : 3
O → E : 2
O → W : 1
```

Y/Z quedaron completamente correctas en esta segunda prueba.

---

# 29. Decisión después de la prueba focalizada

No se decidió reentrenar inmediatamente.

Razón:

La persona realizando las señas no es usuario habitual de ASL, por lo que primero era necesario separar:

```text
error de ejecución
vs
diferencia de dominio
vs
limitación de MediaPipe
vs
limitación del clasificador
```

Se decidió realizar un diagnóstico geométrico.

---

# 30. Diagnóstico dataset vs webcam

Se creó:

```text
scripts\diagnose_asl_webcam_domain.py
```

Se analizaron únicamente:

```text
O P R T X
```

Fuentes utilizadas:

```text
asl_landmarks_final.csv
focused_webcam_results.csv
capturas webcam guardadas
scaler_img_norm.joblib
A3_mlp_img_norm.keras
```

El objetivo fue comparar cada captura webcam con:

1. predicción de A3;
2. vecinos más cercanos del TRAIN;
3. centroides de cada clase.

---

# 31. Método del diagnóstico geométrico

Para cada captura webcam:

```text
captura
   ↓
pipeline MediaPipe
   ↓
img_norm 63
   ↓
StandardScaler original
   ↓
comparación en espacio escalado
```

Se calculó:

```text
Nearest Neighbors sobre TRAIN
distancia euclidiana
```

y también:

```text
centroide promedio por clase
```

---

# 32. Resultado general del diagnóstico

Se observó que A3, KNN y centroides contaban una historia consistente.

Esto fue importante porque descartó la hipótesis de que A3 estuviera tomando decisiones completamente arbitrarias.

---

# 33. Diagnóstico de O

En los 100 vecinos analizados para O:

```text
C = 45
D = 40
Z = 6
O = 3
otros = resto
```

O sí apareció ocasionalmente cerca.

El centroide real O quedó en Top-3 en aproximadamente:

```text
3 / 10 capturas
```

Interpretación:

- O se encuentra cerca de fronteras con otras configuraciones;
- pequeñas diferencias de cierre/orientación alteran la geometría;
- no parece únicamente un problema de ejecución;
- existe sensibilidad real de representación/generalización.

---

# 34. Diagnóstico de P

En los 100 vecinos más cercanos de las 10 capturas P:

```text
J = 57
Q = 32
G = 10
M = 1
P = 0
```

Resultado importante:

```text
0 muestras P entre los 100 vecinos
```

La clase P tenía un rango medio de centroide aproximado de:

```text
4.9
```

Interpretación:

- la P webcam no estaba cayendo en la región P del entrenamiento;
- geométricamente se parecía más a Q/J/G;
- la predicción Q de A3 era coherente con el espacio del TRAIN;
- esto apunta principalmente a diferencia de configuración/orientación/dominio.

---

# 35. Diagnóstico de R

En los 100 vecinos:

```text
U = 55
V = 22
D = 8
Z = 6
N = 5
R = 0
```

Resultado:

```text
0 muestras R entre los 100 vecinos
```

El centroide R obtuvo un rango medio aproximado de:

```text
12.6 / 26
```

Interpretación:

- la geometría webcam estaba muy lejos de la distribución R del TRAIN;
- el cruce índice-medio es una característica crítica;
- la vista frontal puede ocultar profundidad/solapamiento;
- MediaPipe representa la geometría observada, no la intención de la seña.

---

# 36. Diagnóstico de T

En los 100 vecinos:

```text
P = 68
N = 31
A = 1
T = 0
```

Resultado:

```text
0 muestras T entre los 100 vecinos
```

Además, A3 y KNN mostraron prácticamente el mismo patrón:

```text
P 6/10
N 4/10
```

La distancia promedio a vecinos globales fue especialmente baja.

Interpretación:

- las T realizadas caían cómodamente dentro de regiones P/N;
- el problema no parecía ser una decisión absurda del MLP;
- la configuración del pulgar y del puño era crítica.

---

# 37. Diagnóstico de X

En los 100 vecinos:

```text
P = 38
Z = 36
G = 19
D = 6
E = 1
X = 0
```

Resultado:

```text
0 muestras X entre los 100 vecinos
```

Interpretación:

- las capturas X caían principalmente cerca de Z/P/G;
- la curvatura y orientación del índice no se estaban trasladando de forma robusta a la región X;
- existe una diferencia de representación/dominio importante.

---

# 38. Comparación visual con imágenes originales

Después del diagnóstico numérico se decidió recuperar directamente las imágenes originales del TRAIN desde:

```text
D:\Sign2SignData\raw\asl\ASL_original.zip
```

No fue necesario descomprimir el ZIP.

Se creó:

```text
scripts\visualize_asl_domain_neighbors.py
```

Este script produjo comparaciones para:

```text
O
P
R
T
X
```

Cada fila contiene:

```text
webcam
+
3 vecinos globales más cercanos
+
3 vecinos más cercanos de la clase verdadera
```

Archivos generados:

```text
O_visual_comparison.png
P_visual_comparison.png
R_visual_comparison.png
T_visual_comparison.png
X_visual_comparison.png
```

---

# 39. Interpretación visual de P

Las capturas webcam P presentaron distancias aproximadas como:

```text
Q d=3.84
Q d=5.17
G d=9.69
```

Mientras que las P verdaderas más cercanas se encontraban alrededor de:

```text
P d=14.92
P d=15.51
P d=15.59
```

Esto representa una separación muy grande.

Visualmente:

- la orientación de la mano difería;
- la disposición de los dedos difería;
- las capturas webcam se parecían más a Q que a las P representadas por el TRAIN.

Conclusión:

> No se justifica modificar A3 por el caso P en este momento.

---

# 40. Interpretación visual de R

Ejemplo:

```text
Webcam R
↓
V d=1.82
V d=1.82
V d=1.83

R correcta más cercana:
d≈2.48
```

Visualmente:

- en las capturas webcam los dedos se percibían más paralelos;
- en las R del TRAIN el cruce índice-medio era mucho más visible.

Conclusión:

> Las predicciones U/V/W son coherentes con la geometría visible de las capturas.

---

# 41. Interpretación visual de T

Ejemplo:

```text
T webcam
↓
P d=1.29
P d=1.38
P d=1.39

T correcta más cercana:
d≈3.27
```

Visualmente las T del TRAIN muestran una configuración cerrada muy concreta.

Las capturas webcam se parecían más a P/N.

Conclusión:

> A3 estaba clasificando de forma coherente con el espacio aprendido.

---

# 42. Interpretación visual de O

Ejemplo observado:

```text
O webcam
D global  d≈2.79
O real    d≈2.95
```

En este caso las distancias eran mucho más cercanas.

Visualmente la captura sí parecía razonablemente O.

Conclusión:

> O presenta una frontera real de generalización y sensibilidad a pequeños cambios geométricos.

Este caso no se explica únicamente como “seña mal realizada”.

---

# 43. Interpretación visual de X

Ejemplo:

```text
X webcam

Z:
d≈1.69
d≈1.70
d≈2.14

X más cercana:
d≈3.68
```

Visualmente las capturas sí contenían cierta curvatura del índice, por lo que no se consideró un caso trivial de ejecución incorrecta.

Conclusión:

> X parece presentar una diferencia real entre la representación webcam y la distribución X del dataset sintético.

Posibles factores:

- grado de flexión;
- orientación;
- perspectiva;
- componente Z estimada por MediaPipe;
- domain shift;
- variabilidad insuficiente del TRAIN.

---

# 44. Resumen de clases problemáticas después del diagnóstico

## K

```text
Prueba focalizada inválida.
Se realizó P accidentalmente.
No utilizar esos 10 intentos para evaluar K.
```

---

## O

```text
Sensibilidad real webcam/dataset.
Frontera especialmente con D/C.
Caso de robustez/generalización.
```

---

## P

```text
Configuración webcam muy diferente del TRAIN.
Se aproxima fuertemente a Q/J.
Principalmente diferencia de dominio/ejecución.
```

---

## R

```text
Cruce índice-medio poco visible en webcam.
Landmarks próximos a U/V.
Principalmente diferencia de ejecución/vista.
```

---

## T

```text
Configuración webcam diferente del TRAIN.
Landmarks muy próximos a P/N.
Principalmente diferencia de configuración.
```

---

## U

```text
Focused:
5/10 Top-1
10/10 Top-3

Muy sensible a la posición del pulgar.
Clase fronteriza pero recuperable con Top-k.
```

---

## X

```text
Diferencia de dominio/representación.
Landmarks próximos a Z/P/G.
Caso de robustez real.
```

---

## Y

```text
Focused:
10/10 Top-1
10/10 Top-3

El problema inicial se redujo al ejecutar la configuración con mayor cuidado.
```

---

## Z

```text
Focused:
10/10 Top-1
10/10 Top-3

Se mantiene la limitación conceptual:
Z es dinámica en ASL estándar,
pero el proyecto utiliza una aproximación estática.
```

---

# 45. Conclusión técnica general de ASL

La evaluación muestra dos realidades complementarias.

## 45.1 Dentro del dominio del dataset

A3 obtuvo:

```text
99.15 % Top-1
99.72 % Top-3
```

Esto demuestra que el modelo aprende y discrimina correctamente el espacio de características utilizado en entrenamiento.

## 45.2 Fuera del dominio, con webcam real

Evaluación general:

```text
100 % detección de mano
73.85 % Top-1
93.08 % Top-3
```

Esto muestra una caída de generalización al pasar del dataset al entorno real.

Sin embargo, el diagnóstico posterior demostró que una parte importante de los errores no corresponde a decisiones arbitrarias del clasificador.

En múltiples casos:

```text
A3
KNN
centroides
```

coinciden en que la captura webcam se encuentra geométricamente dentro de regiones asociadas a otras clases.

Por tanto, la explicación principal incluye:

- domain shift;
- orientación;
- configuración manual;
- sensibilidad geométrica;
- limitaciones de MediaPipe;
- variación insuficiente en ciertas clases del dataset.

---

# 46. Conclusión redactable para reporte técnico

Una formulación útil para documentación futura es:

> El modelo ASL seleccionado presentó un rendimiento elevado sobre el conjunto de prueba del mismo dominio, alcanzando aproximadamente 99.15 % de exactitud Top-1 y 99.72 % Top-3. Al trasladar el sistema a un escenario de captura mediante webcam, la exactitud Top-1 disminuyó a 73.85 %, mientras que Top-3 se mantuvo en 93.08 %, con una tasa de detección de mano del 100 %. Un análisis posterior mediante vecinos más cercanos, centroides y comparación visual de muestras mostró que varias de las capturas erróneas se ubican geométricamente en regiones correspondientes a otras clases del conjunto de entrenamiento. Este comportamiento evidencia una diferencia de dominio entre las imágenes utilizadas para entrenamiento y las capturas reales, además de sensibilidad a orientación, disposición de dedos y perspectiva. Las configuraciones P, R y T mostraron diferencias claras respecto a las muestras del entrenamiento, mientras que O y X presentaron mayor sensibilidad de representación y generalización.

Esta redacción deberá adaptarse posteriormente al formato final del reporte y complementar con tablas/figuras.

---

# 47. Decisión actual sobre A3

A3 queda:

```text
CONGELADO POR AHORA
```

No se reentrenará antes de avanzar con LSM.

Razones:

1. presenta excelente rendimiento dentro del dominio;
2. el pipeline local funciona;
3. la detección MediaPipe en webcam fue 100 % en las pruebas;
4. la mayoría de las clases funcionan correctamente;
5. los errores restantes están parcialmente explicados;
6. continuar optimizando ASL podría consumir tiempo necesario para completar el sistema Sign2Sign.

Esto NO significa que ASL sea definitivo.

Significa:

> A3 es el baseline seleccionado y funcional del prototipo actual.

---

# 48. Posibles mejoras futuras de ASL

Si el calendario lo permite después de completar el flujo end-to-end:

## 48.1 Fine-tuning con capturas reales

Crear un dataset webcam independiente y agregar variabilidad de:

- usuarios;
- iluminación;
- distancia;
- orientación;
- tamaño de mano;
- fondo.

Nunca utilizar las capturas de evaluación actual como entrenamiento sin separar nuevos conjuntos.

---

## 48.2 Data augmentation

Evaluar augmentations consistentes con producción:

- rotaciones pequeñas;
- perspectiva;
- cambios de iluminación;
- ruido;
- variación de escala;
- orientación controlada.

---

## 48.3 Multiframe consensus

Al presionar ENTER:

```text
capturar varios frames
↓
predicciones individuales
↓
promedio / voto
↓
letra final
```

---

## 48.4 Mejor criterio de confianza

No utilizar únicamente:

```text
max(Softmax)
```

Investigar:

```text
Top-1 probability
Top-1 - Top-2 margin
entropy
calibration
Top-k
```

---

## 48.5 Revisión de O y X

Si se decide robustecer clases específicas, O y X son actualmente las candidatas más interesantes porque sus fallos no se explicaron únicamente por una configuración claramente distinta.

---

# 49. Reglas metodológicas que deben mantenerse

## Test original congelado

No ajustar hiperparámetros, arquitectura ni reglas utilizando:

```text
ASL original test
```

Ese test ya fue abierto una vez después de la selección.

---

## Evaluaciones webcam preservadas

No reutilizar como entrenamiento:

```text
run_20260906_121444
focused_run_20260906_131838
```

Estas ejecuciones deben mantenerse como evidencia experimental.

---

## Pipeline idéntico

Mantener:

```text
offline extraction
=
train representation
=
webcam representation
```

salvo que se diseñe formalmente una nueva versión y se reevalúe completamente.

---

## Top-k

Conservar siempre:

```text
Top-1
Top-2
Top-3
probabilidades
```

porque la futura capa de corrección podrá aprovechar ambigüedades.

---

# 50. Lecciones principales de esta fase

## 50.1 Un test de 99 % no garantiza 99 % en webcam

El dataset y la webcam pertenecen a dominios distintos.

Por ello:

```text
evaluación offline ≠ evaluación end-to-end
```

Ambas son necesarias.

---

## 50.2 La detección y la clasificación son problemas distintos

MediaPipe obtuvo:

```text
100 % detección
```

en las evaluaciones webcam.

Sin embargo, la clasificación no fue perfecta.

Por tanto:

```text
mano detectada
≠
seña correctamente clasificada
```

---

## 50.3 Top-3 es extremadamente útil

Webcam general:

```text
Top-1 = 73.85 %
Top-3 = 93.08 %
```

Existe una diferencia de casi 20 puntos porcentuales.

Esto respalda el diseño original de conservar:

```text
Top-k + probabilidades
```

para la capa de corrección contextual.

---

## 50.4 Softmax alto puede estar equivocado

Se observaron errores de alta confianza tanto en validation/test como durante pruebas.

Por ello:

> Probabilidad Softmax alta no equivale automáticamente a certeza real.

---

## 50.5 La geometría explica muchos errores

El análisis KNN/centroides mostró que múltiples capturas mal clasificadas ya estaban ubicadas cerca de otra clase antes de que A3 tomara la decisión.

Esto demuestra que es necesario analizar:

```text
representación
+
clasificador
```

y no culpar únicamente al modelo final.

---

## 50.6 La ejecución del usuario importa

Al no contar todavía con un evaluador experto en ASL, la forma de realizar una letra puede introducir errores.

Ejemplos:

- R depende del cruce visible;
- T depende del pulgar;
- U depende de configuración muy precisa;
- P depende de orientación;
- O depende del cierre;
- X depende de curvatura.

Esta limitación deberá mencionarse en cualquier evaluación webcam realizada hasta ahora.

---

## 50.7 Las imágenes originales deben conservarse

Aunque el modelo trabaje únicamente con landmarks, conservar las imágenes originales permitió posteriormente explicar qué estaba ocurriendo geométricamente.

Por eso se mantuvo:

```text
ASL_original.zip
```

sin eliminarlo.

---

# 51. Scripts relevantes creados durante esta fase

```text
scripts\test_asl_model_webcam.py
```

Smoke test interactivo.

```text
scripts\evaluate_asl_webcam.py
```

Evaluación controlada A-Z.

```text
scripts\evaluate_asl_webcam_focused.py
```

Evaluación focalizada de letras problemáticas.

```text
scripts\diagnose_asl_webcam_domain.py
```

Análisis KNN + centroides webcam vs TRAIN.

```text
scripts\visualize_asl_domain_neighbors.py
```

Comparación visual con vecinos del dataset original dentro del ZIP.

---

# 52. Estructura de resultados recomendada a conservar

```text
D:\Sign2SignData\
│
├── raw\
│   └── asl\
│       └── ASL_original.zip
│
├── processed\
│   └── landmarks\
│       └── asl_landmarks_final.csv
│
└── inventory\
    └── webcam_eval\
        ├── asl\
        │   └── run_20260906_121444\
        │
        ├── asl_focused\
        │   └── focused_run_20260906_131838\
        │
        └── asl_domain_diagnosis\
            └── diagnosis_focused_run_20260906_131838\
                └── visual_comparison\
```

---

# 53. Estado de cierre de ASL

Checklist:

```text
[x] Dataset organizado
[x] Extracción de landmarks
[x] Normalización
[x] Split train/validation/test
[x] Random Forest baseline
[x] MLP baseline
[x] Comparación img_norm/world_norm
[x] Selección formal usando validation
[x] Test abierto una sola vez
[x] Modelo guardado
[x] Scaler guardado
[x] Inferencia local
[x] Webcam smoke test
[x] Evaluación webcam controlada
[x] Evaluación focalizada
[x] Diagnóstico geométrico
[x] Comparación visual dataset/webcam
[x] Limitaciones identificadas
[x] A3 congelado como baseline actual
```

---

# 54. Siguiente fase: LSM

A partir de este punto la prioridad cambia a LSM.

El flujo previsto es reutilizar la metodología ASL:

```text
L1 = Random Forest + img_norm
L2 = Random Forest + world_norm
L3 = MLP + img_norm
L4 = MLP + world_norm
```

pero con una consideración importante.

LSM tiene:

```text
TRAIN:
26 clases
incluye E/L/Z provisionales tomadas de ASL

VALIDATION:
23 clases genuinas LSM

TEST:
23 clases genuinas LSM
```

Por lo tanto:

> El notebook LSM no debe copiarse ciegamente del notebook ASL.

Será necesario adaptar correctamente:

- métricas;
- matrices;
- reportes;
- clases presentes;
- selección;
- evaluación final.

Las clases provisionales:

```text
E
L
Z
```

deben permanecer train-only dentro del esquema actual.

---

# 55. Criterio de avance del proyecto

El objetivo no es perfeccionar indefinidamente el clasificador ASL.

El objetivo final de Sign2Sign es construir el flujo:

```text
ASL / LSM
    ↓
reconocimiento de letras
    ↓
constructor textual
    ↓
corrección / traducción
    ↓
vocabulario cerrado
    ↓
mapeo a recursos visuales
    ↓
videos de lengua destino
    ↓
interfaz final
```

Por lo tanto, ASL se considera suficientemente avanzado para continuar con LSM.

---

# 56. Nota para documentación final

Este documento contiene tanto:

- resultados formales;
- observaciones experimentales;
- decisiones prácticas;
- hipótesis técnicas;
- limitaciones.

Para el reporte técnico final será necesario separar claramente:

```text
Resultados
Discusión
Limitaciones
Trabajo futuro
```

y evitar presentar hipótesis como hechos demostrados.

La evidencia más fuerte disponible actualmente es:

1. métricas validation/test;
2. métricas webcam;
3. KNN sobre TRAIN;
4. centroides;
5. comparación visual con muestras originales.

---

# 57. Resumen ejecutivo de la fase ASL

El reconocedor ASL de Sign2Sign se desarrolló mediante landmarks de mano de MediaPipe y un clasificador MLP entrenado sobre la representación `img_norm`. Después de comparar Random Forest y MLP con `img_norm` y `world_norm`, el modelo A3 fue seleccionado utilizando validation. A3 alcanzó 99.15 % de exactitud Top-1 y 99.72 % Top-3 en test. Posteriormente fue integrado con el pipeline real de webcam, donde MediaPipe alcanzó 100 % de detección de mano y el clasificador obtuvo 73.85 % Top-1 y 93.08 % Top-3 en una evaluación de 130 intentos. Las clases problemáticas fueron estudiadas mediante una prueba focalizada, vecinos más cercanos, centroides y comparación visual con el dataset original. El análisis mostró que muchas capturas erróneas se ubican geométricamente dentro de regiones correspondientes a otras clases, evidenciando diferencias de dominio, orientación y ejecución más que un fallo aislado del MLP. A3 queda congelado como baseline funcional mientras el proyecto avanza hacia LSM.

---

**Fin del documento de fase ASL.**
