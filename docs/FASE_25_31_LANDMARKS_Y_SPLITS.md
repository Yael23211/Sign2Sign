# Fase 3 — Conversión de imágenes a landmarks y preparación de datasets

**Proyecto:** Sign2Sign  
**Periodo de referencia:** 25–31 de agosto de 2026  
**Estado:** Cerrada  
**Salida del plan:** Dos datasets de landmarks listos para entrenamiento.

---

## 1. Objetivo de la fase

El objetivo de esta fase fue transformar los datasets de imágenes de American Sign Language (ASL) y Lengua de Señas Mexicana (LSM) en representaciones numéricas basadas en landmarks de la mano, normalizarlas, validar su integridad y construir conjuntos de `train`, `validation` y `test` adecuados para el entrenamiento posterior de dos clasificadores independientes.

De acuerdo con el plan de ejecución de Sign2Sign, el hito asociado a esta fase consiste en contar con los datasets ASL y LSM limpios, convertidos a landmarks y separados en conjuntos de entrenamiento, validación y prueba.

La fase se consideró terminada únicamente cuando los datos quedaron:

- convertidos a landmarks de manera reproducible;
- normalizados;
- auditados numéricamente;
- trazables respecto a su dataset y muestra de origen;
- separados sin introducir contaminación evidente entre entrenamiento y evaluación;
- disponibles en archivos finales para el entrenamiento de los modelos.

---

## 2. Estado de entrada

La fase comenzó después del cierre de la limpieza e inventario de datos.

### ASL

El dataset ASL contenía:

- 26 clases correspondientes a las letras `A–Z`;
- una clase adicional `Blank`;
- división original en `train` y `test`;
- 27,000 imágenes válidas a nivel de archivo.

La clase `Blank` se mantuvo como muestra negativa y no se incorporó al clasificador de letras. La arquitectura definida establece que una imagen sin una mano detectable debe ser rechazada antes de llegar al clasificador, por lo que el modelo supervisado se limita a las 26 clases `A–Z`.

### LSM

El dataset LSM limpio contenía:

- 89,683 imágenes registradas originalmente;
- 183 duplicados exactos identificados y conservados únicamente para trazabilidad;
- 89,500 muestras aceptadas para procesamiento;
- ausencia de las clases `E`, `L`, `Ñ` y `Z` en el conjunto original.

Las muestras rechazadas conservaron su registro en el manifest mediante:

```text
status = rejected
reject_reason = exact_duplicate
```

mientras que únicamente las muestras con:

```text
status = accepted
```

fueron utilizadas para extracción de landmarks.

---

## 3. Tecnología utilizada para extracción

Se utilizó **MediaPipe Tasks HandLandmarker**, debido a que la versión instalada de MediaPipe no expone la interfaz clásica `mp.solutions`.

Modelo utilizado:

```text
models/mediapipe/hand_landmarker.task
```

Configuración base:

```text
num_hands = 1
min_hand_detection_confidence = 0.5
min_hand_presence_confidence = 0.5
min_tracking_confidence = 0.5
```

Cada detección válida produce **21 landmarks de la mano**.

Para cada muestra se conservaron las siguientes representaciones:

- `img_raw`: landmarks expresados respecto a la imagen procesada;
- `img_norm`: landmarks de imagen normalizados;
- `world_raw`: coordenadas tridimensionales entregadas por MediaPipe;
- `world_norm`: coordenadas tridimensionales normalizadas.

Además se conservaron metadatos como:

- clase;
- ruta original;
- dataset de procedencia;
- grupo o sesión;
- variante de detección utilizada;
- lateralidad detectada;
- confianza de lateralidad;
- dimensiones originales y procesadas;
- información de trazabilidad del manifest.

---

## 4. Normalización de landmarks

La normalización utilizada toma como referencia la muñeca (`landmark 0`) y una escala basada en la distancia entre la muñeca y el `landmark 9`.

De forma conceptual:

```text
punto_normalizado = (punto - muñeca) / distancia(muñeca, landmark_9)
```

Con ello se busca disminuir la dependencia respecto a:

- posición absoluta de la mano en la imagen;
- resolución de la imagen;
- tamaño aparente de la mano;
- distancia respecto a la cámara.

La misma lógica se aplicó tanto a los landmarks de imagen como a los landmarks tridimensionales.

Como criterio de validación se comprobó que:

```text
muñeca normalizada = (0, 0, 0)
distancia normalizada entre landmark 0 y 9 = 1
```

---

## 5. Problema de detección en LSM

Durante las primeras pruebas se observó una diferencia importante entre ASL y LSM.

Ejemplos iniciales:

```text
ASL A -> aproximadamente 98 % de detección
LSM M -> aproximadamente 66 %
LSM N -> aproximadamente 42 %
```

Las pruebas con webcam, utilizando el mismo HandLandmarker, mostraron detecciones de hasta 100 % en señas equivalentes. Esto indicó que el problema no se encontraba directamente en MediaPipe, sino en características de determinadas imágenes del dataset LSM, entre ellas:

- encuadre;
- proporción de la imagen;
- posición de la mano;
- escala relativa de la mano;
- características específicas de determinadas sesiones de captura.

---

## 6. Cascade de preprocesamiento

En lugar de elegir una única transformación para todas las imágenes, se implementó un esquema de **fallback o cascade**.

La lógica definida fue:

```text
1. baseline
2. center_4_3
3. crop_85
4. crop_70
5. letterbox
```

Cada imagen se intenta primero sin transformación adicional. Si MediaPipe no detecta una mano, se prueban de manera secuencial las variantes restantes hasta obtener una detección o agotar el cascade.

La decisión se tomó después de comprobar experimentalmente que ninguna transformación era superior en todos los grupos. Algunas técnicas mejoraban clases o sesiones específicas, pero perjudicaban otras.

### Resultado del cascade en LSM

Sobre las 85,883 muestras de LSM con detección válida:

| Variante | Muestras | Porcentaje de detecciones válidas |
|---|---:|---:|
| baseline | 82,086 | 95.58 % |
| center_4_3 | 2,758 | 3.21 % |
| crop_70 | 382 | 0.44 % |
| letterbox | 375 | 0.44 % |
| crop_85 | 282 | 0.33 % |

El cascade permitió recuperar **3,797 imágenes adicionales** que no habían sido resueltas mediante el procesamiento base.

La misma estrategia debe conservarse posteriormente durante la captura e inferencia, ya que el entrenamiento y la operación del sistema deben utilizar un preprocesamiento compatible.

---

## 7. Extracción masiva de ASL

Se procesaron las 26,000 imágenes correspondientes únicamente a las letras `A–Z`.

Resultado:

```text
Imágenes procesadas : 26,000
Landmarks válidos   : 24,778
Sin landmarks       : 1,222
Tasa de extracción  : 95.30 % aproximadamente
```

La comparación entre los splits originales mostró tasas muy similares:

```text
train original : 22,295 landmarks válidos
test original  :  2,483 landmarks válidos
```

La tasa de detección fue cercana entre ambos conjuntos, por lo que no se observó evidencia de un sesgo relevante provocado por MediaPipe entre el `train` y el `test` originales.

---

## 8. Extracción masiva de LSM

Después de aplicar el filtro definitivo del manifest:

```text
status == accepted
```

se procesaron exactamente:

```text
89,500 muestras
```

Resultado final:

```text
Landmarks válidos : 85,883
Sin landmarks     :  3,617
Tasa extracción   : 95.96 %
```

### Resultado por clase

| Clase | Válidas | Fallidas | Tasa |
|---|---:|---:|---:|
| A | 3,833 | 59 | 98.48 % |
| B | 3,880 | 11 | 99.72 % |
| C | 3,900 | 0 | 100.00 % |
| D | 3,465 | 435 | 88.85 % |
| F | 3,851 | 49 | 98.74 % |
| G | 3,899 | 0 | 100.00 % |
| H | 3,900 | 0 | 100.00 % |
| I | 3,827 | 73 | 98.13 % |
| J | 3,892 | 8 | 99.79 % |
| K | 3,752 | 148 | 96.21 % |
| M | 3,235 | 665 | 82.95 % |
| N | 2,828 | 1,072 | 72.51 % |
| O | 3,900 | 0 | 100.00 % |
| P | 3,882 | 18 | 99.54 % |
| Q | 3,825 | 74 | 98.10 % |
| R | 3,429 | 471 | 87.92 % |
| S | 3,734 | 151 | 96.11 % |
| T | 3,796 | 104 | 97.33 % |
| U | 3,703 | 178 | 95.41 % |
| V | 3,688 | 66 | 98.24 % |
| W | 3,888 | 11 | 99.72 % |
| X | 3,876 | 24 | 99.38 % |
| Y | 3,900 | 0 | 100.00 % |

Las clases con mayor dificultad para MediaPipe fueron principalmente `N`, `M`, `R` y `D`.

No se eliminaron clases completas por estas tasas, ya que incluso los casos más difíciles conservaron una cantidad suficiente de muestras útiles para iniciar experimentación.

---

## 9. Auditoría numérica de LSM

Se ejecutó una auditoría completa sobre los 85,883 vectores válidos.

Resultado:

```text
Válidas con NaN        : 0
Válidas con infinito   : 0

img_raw incompletas    : 0
img_norm incompletas   : 0
world_raw incompletas  : 0
world_norm incompletas : 0
```

También se verificó la normalización:

```text
Image muñeca incorrecta : 0
Image escala incorrecta : 0
World muñeca incorrecta : 0
World escala incorrecta : 0
```

Errores máximos observados:

```text
image wrist : 0
image scale : 0
world wrist : 0
world scale : 0
```

Por lo tanto, los vectores aceptados se consideran numéricamente consistentes y aptos para entrenamiento.

---

## 10. Incorporación provisional de E, L y Z en LSM

El dataset original LSM no contiene las clases `E`, `L`, `Ñ` y `Z`.

Después de revisar las configuraciones manuales y las necesidades del prototipo se tomó la decisión de incorporar provisionalmente `E`, `L` y `Z` desde ASL.

Las muestras añadidas fueron:

```text
E : 929
L : 930
Z : 900
```

Total:

```text
2,759 muestras provisionales
```

Estas muestras se conservan explícitamente identificadas:

```text
dataset = LSM
source_dataset = ASL
provisional = 1
provisional_source = ASL_missing_LSM_class
```

No se mezclan silenciosamente con las muestras LSM genuinas.

Además, las muestras provisionales se restringieron a:

```text
split_final = train
```

y no se utilizarán para validación o prueba formal del modelo LSM.

El dataset candidato resultante contiene:

```text
LSM genuino válido : 85,883
ASL provisional    :  2,759
Total              : 88,642
```

---

## 11. Tratamiento de Ñ

No se generó artificialmente una clase `Ñ` duplicando los vectores de `N`.

La razón es que, dentro de la simplificación estática del prototipo, `N` y `Ñ` no disponen de información suficiente para formar dos clases geométricamente distinguibles.

Duplicar las mismas entradas con dos etiquetas diferentes introduciría una contradicción para el clasificador.

La decisión actual es:

```text
captura estática
      ↓
clasificador
      ↓
N / alternativas top-k
      ↓
vocabulario + corrección contextual
      ↓
N o Ñ según la palabra plausible
```

La incapacidad para representar correctamente el movimiento de `Ñ` se mantiene como una limitación explícita del prototipo.

---

## 12. Construcción del split final de LSM

Inicialmente se evaluó una división por grupos individuales buscando una proporción cercana a `80/10/10`.

La propuesta obtenida fue aproximadamente:

```text
train      : 71.78 %
validation : 14.48 %
test       : 13.74 %
```

Sin embargo, una auditoría posterior mediante `session_hint` detectó posible contaminación entre splits:

```text
sesiones únicas       : 7
sesiones en >1 split  : 5
```

Por ejemplo, grupos asociados a una misma sesión aparecían simultáneamente en `train`, `validation` y `test`.

Por este motivo se descartó la propuesta por clase/grupo y se adoptó una división a nivel de sesión.

### Estructura identificada

Las sesiones con cobertura completa de las 23 clases LSM genuinas fueron:

- sesión 3;
- sesión 5;
- sesión 6.

Se decidió utilizar:

```text
TRAIN
- sesión 5
- sesión 1
- grupos NORMAL
- GENERIC_FRAME
- sesión 7
- BURST
- grupos parciales sin session_hint
- E/L/Z provisionales provenientes de ASL

VALIDATION
- sesión 3 completa

TEST
- sesión 6 completa
```

La sesión 6 se utilizó como `test` porque contiene las 23 clases genuinas y además incluye varios de los grupos que mostraron mayores dificultades de detección, lo que permite una evaluación más exigente de generalización.

### Resultado final LSM

```text
train       : 60,907  (68.71 %)
validation  : 14,791  (16.69 %)
test        : 12,944  (14.60 %)
total       : 88,642
```

Cobertura:

```text
train       : 26 clases
validation  : 23 clases LSM genuinas
test        : 23 clases LSM genuinas
```

`E`, `L` y `Z` aparecen únicamente en entrenamiento porque no existen muestras LSM genuinas disponibles para evaluarlas.

El split final verificó:

- sesión 3 aislada en `validation`;
- sesión 6 aislada en `test`;
- cero muestras provisionales fuera de `train`.

### Limitación del split

`session_hint` se deriva de la estructura y nomenclatura disponibles en el dataset. No se cuenta con evidencia suficiente para afirmar que represente necesariamente un identificador único de persona o firmante.

Por lo tanto, el split reduce la contaminación entre sesiones de adquisición identificables, pero no permite garantizar independencia absoluta por sujeto.

---

## 13. Construcción del split final de ASL

ASL ya incluía una separación original en `train` y `test`.

Se decidió conservar completamente el `test` original y obtener `validation` únicamente desde el `train` original.

Procedimiento:

```text
train original
      ↓
90 % train final
10 % validation

test original
      ↓
100 % test final
```

La extracción de `validation` fue estratificada por clase y reproducible mediante una selección determinista.

Resultado:

```text
train       : 20,061  (80.96 %)
validation  :  2,234  ( 9.02 %)
test        :  2,483  (10.02 %)
total       : 24,778
```

Validaciones:

```text
Test original movido      : 0
Validation fuera de train : 0
Filas sin split final     : 0
```

Las 26 clases `A–Z` están presentes en los tres conjuntos.

---

## 14. Tratamiento de Blank

La clase `Blank` del dataset ASL no se incorporó como clase número 27 del clasificador.

La arquitectura adoptada es:

```text
imagen
   ↓
MediaPipe
   ↓
¿mano detectada?
   ├─ no -> rechazo / mensaje de error
   └─ sí
        ↓
     landmarks
        ↓
     clasificador A-Z
```

Por lo tanto, `Blank` se conserva como conjunto negativo para posibles pruebas del módulo de detección, pero no forma parte del problema supervisado `A–Z`.

---

## 15. Archivos principales generados

### Datos procesados

```text
D:\Sign2SignData\processed\landmarks\asl_landmarks.csv
D:\Sign2SignData\processed\landmarks\asl_landmarks_final.csv

D:\Sign2SignData\processed\landmarks\lsm_landmarks.csv
D:\Sign2SignData\processed\landmarks\lsm_landmarks_provisional.csv
D:\Sign2SignData\processed\landmarks\lsm_landmarks_final.csv
```

### Reportes

```text
D:\Sign2SignData\inventory\landmark_analysis\lsm_landmarks_by_class.csv
D:\Sign2SignData\inventory\landmark_analysis\lsm_landmarks_by_group.csv
D:\Sign2SignData\inventory\landmark_analysis\lsm_landmark_integrity_problems.csv

D:\Sign2SignData\processed\landmarks\lsm_final_split_summary.csv
D:\Sign2SignData\processed\landmarks\asl_final_split_summary.csv
```

### Scripts de esta etapa

Entre los scripts creados o utilizados durante la fase se encuentran:

```text
scripts\test_webcam_detection_rate.py
scripts\extract_lsm_landmarks.py
scripts\analyze_lsm_landmarks.py
scripts\build_lsm_provisional_landmarks.py
scripts\propose_lsm_group_split.py
scripts\check_lsm_session_leakage.py
scripts\analyze_lsm_sessions.py
scripts\build_lsm_final_split.py
scripts\build_asl_final_split.py
```

Además, la lógica compartida de extracción, cascade y normalización se centralizó en el módulo de landmarks del proyecto para evitar diferencias entre los procesos de preparación y la futura inferencia.

---

## 16. Decisiones técnicas consolidadas

Al cierre de la fase quedan establecidas las siguientes decisiones:

1. Se utilizarán **dos clasificadores separados**, uno para ASL y otro para LSM.
2. La entrada principal de los modelos serán landmarks de MediaPipe.
3. Se conservarán tanto `img_norm` como `world_norm` hasta comparar experimentalmente su desempeño.
4. El mismo preprocesamiento utilizado durante preparación deberá reutilizarse durante inferencia.
5. El cascade solo aplica transformaciones cuando la detección base falla.
6. Las muestras sin landmarks no se utilizan para entrenar el clasificador.
7. `Blank` se mantiene fuera del clasificador ASL.
8. `E`, `L` y `Z` de LSM son provisionales y proceden de ASL.
9. Las muestras provisionales de LSM se utilizan exclusivamente en entrenamiento.
10. `Ñ` no se entrena como clase duplicada de `N`; su ambigüedad se delega al contexto y al vocabulario.
11. El split de LSM se realiza a nivel de sesiones identificables para reducir leakage.
12. El test original de ASL se preserva.
13. Las decisiones de representación y modelo se tomarán mediante `validation`, evitando utilizar el conjunto de prueba para selección de hiperparámetros.

---

## 17. Limitaciones identificadas

Al cierre de esta etapa permanecen las siguientes limitaciones:

- Algunas configuraciones de mano, especialmente en `N`, `M`, `R` y `D` de LSM, presentan menor tasa de detección con MediaPipe.
- Las clases `E`, `L` y `Z` del clasificador LSM dependen provisionalmente de muestras ASL.
- No existe una clase independiente de `Ñ` dentro del reconocedor estático.
- Las letras que dependen de movimiento se modelan mediante una representación estática.
- La estructura de sesiones del dataset LSM fue inferida a partir de metadatos y nombres de archivo; no equivale necesariamente a una separación certificada por firmante.
- La distribución del conjunto LSM no es perfectamente balanceada, aspecto que será tratado durante entrenamiento y evaluación sin modificar artificialmente el dataset maestro.

Estas limitaciones no impiden continuar con el prototipo, pero deberán conservarse explícitamente en el reporte técnico y en la interpretación de resultados.

---

## 18. Evidencias obtenidas

La fase produjo evidencia reproducible de:

- detección de mano mediante MediaPipe Tasks;
- funcionamiento del pipeline sobre webcam;
- diferencia de comportamiento entre imágenes ASL y determinadas sesiones LSM;
- mejora obtenida mediante cascade;
- extracción masiva sobre ambos datasets;
- conteos por clase;
- integridad numérica de los vectores;
- validación matemática de la normalización;
- trazabilidad de las clases LSM provisionales;
- detección y corrección de una primera propuesta de split con posible leakage;
- construcción definitiva de los splits ASL y LSM.

---

## 19. Criterios de cierre

Checklist de la fase:

- [x] MediaPipe HandLandmarker integrado.
- [x] Extracción de 21 landmarks reproducible.
- [x] Cascade de detección definido.
- [x] Landmarks normalizados.
- [x] Extracción masiva ASL completada.
- [x] Extracción masiva LSM completada.
- [x] Integridad numérica comprobada.
- [x] Duplicados LSM excluidos del procesamiento.
- [x] Clases faltantes de LSM documentadas.
- [x] E/L/Z provisionales identificadas y trazables.
- [x] Tratamiento de Ñ documentado.
- [x] `Blank` excluido justificadamente del clasificador.
- [x] Split final ASL creado.
- [x] Test original ASL preservado.
- [x] Split final LSM creado.
- [x] Sesiones de validación y prueba LSM aisladas.
- [x] Datasets finales disponibles para entrenamiento.

**Resultado:** fase cerrada.

---

## 20. Estado final y siguiente fase

La salida obligatoria del periodo 25–31 de agosto se considera cumplida:

> **Dos datasets de landmarks listos para entrenamiento.**

Archivos de entrada para la siguiente fase:

```text
ASL:
D:\Sign2SignData\processed\landmarks\asl_landmarks_final.csv

LSM:
D:\Sign2SignData\processed\landmarks\lsm_landmarks_final.csv
```

La siguiente fase corresponde al **entrenamiento de los clasificadores ASL y LSM**.

El primer experimento consistirá en comparar, bajo la misma configuración de modelo:

```text
img_norm
vs.
world_norm
```

La selección se realizará con el conjunto de validación. El conjunto de prueba se reservará para la evaluación final del modelo seleccionado.

Las métricas previstas incluyen, al menos:

- accuracy;
- precision;
- recall;
- F1-score;
- matriz de confusión;
- top-1 accuracy;
- top-k accuracy.

Con ello se inicia la fase de entrenamiento partiendo de datasets cuya procedencia, estructura, normalización y separación han sido previamente verificadas.
