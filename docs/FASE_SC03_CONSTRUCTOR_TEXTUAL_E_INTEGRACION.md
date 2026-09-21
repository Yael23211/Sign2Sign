# Fase SC-03 — Constructor textual e integración del reconocimiento

## 1. Objetivo de la fase

Esta fase tuvo como objetivo completar el subsistema **SC-03 Reconocimiento de señas** mediante la incorporación de un constructor textual capaz de recibir las predicciones producidas por los clasificadores ASL y LSM, acumularlas como letras y palabras, conservar las alternativas Top-k asociadas a cada captura y producir una salida textual estructurada que pueda ser consumida por el siguiente subsistema.

El alcance de esta fase no incluye corrección ortográfica, selección contextual de alternativas, traducción ni generación de la salida visual. Estas responsabilidades corresponden a los subsistemas posteriores.

---

## 2. Decisiones de diseño

### 2.1 Separación de responsabilidades

Se decidió mantener separados los siguientes componentes:

- **Pipeline de landmarks**: procesamiento de imagen, detección de mano y normalización.
- **SignRecognizer**: carga del modelo y scaler, inferencia y generación de Top-k.
- **TextBuilder**: construcción y edición de la secuencia textual.
- **Script de ejecución SC-03**: manejo de webcam, teclado y visualización temporal.

De esta manera, `TextBuilder` no depende de MediaPipe, TensorFlow, OpenCV ni de la webcam, mientras que `SignRecognizer` no administra la cadena textual.

---

## 3. Constructor textual

Se creó el módulo:

```text
src/text/
├── __init__.py
└── text_builder.py
```

### 3.1 Responsabilidades

`TextBuilder` permite:

- agregar una letra reconocida;
- conservar Top-1 y Top-k con sus probabilidades;
- insertar separadores entre palabras;
- eliminar el último elemento;
- limpiar la secuencia completa;
- finalizar la secuencia y producir una representación inmutable;
- generar una salida serializable mediante `to_dict()`.

### 3.2 Controles definidos

La interacción adoptada para el flujo real de SC-03 es:

| Tecla | Acción |
|---|---|
| `ENTER` | Capturar una seña, reconocerla y agregar la letra Top-1 |
| `SPACE` | Insertar un separador entre palabras |
| `BACKSPACE` | Eliminar la última letra o espacio |
| `ESC` | Limpiar la secuencia actual |
| `T` | Finalizar y producir la salida textual |
| `Q` | Salir sin finalizar |

Reglas adicionales:

- no se permiten espacios iniciales;
- no se permiten espacios consecutivos;
- un `BACKSPACE` sobre una secuencia vacía no produce error;
- `finalize()` elimina espacios finales;
- no se permite finalizar una secuencia vacía;
- `finalize()` no limpia automáticamente el contenido, para evitar perder la captura si el siguiente subsistema falla.

---

## 4. Estructura de salida

Cada letra reconocida se conserva como un token con:

- clase Top-1;
- probabilidad Top-1;
- alternativas Top-k;
- probabilidades asociadas.

Ejemplo conceptual:

```json
{
  "source_language": "LSM",
  "text": "HOLA MUNDO",
  "character_count": 9,
  "word_count": 2,
  "tokens": [
    {
      "kind": "letter",
      "value": "H",
      "top_k": [
        {"label": "H", "probability": 0.98},
        {"label": "G", "probability": 0.01},
        {"label": "N", "probability": 0.01}
      ]
    }
  ]
}
```

La conservación de Top-k se considera importante porque algunas clases presentan geometrías cercanas y pueden generar probabilidades repartidas entre alternativas plausibles.

---

## 5. Pruebas unitarias de TextBuilder

Se creó:

```text
tests/
├── __init__.py
└── test_text_builder.py
```

Las pruebas se ejecutaron con:

```bash
python -m unittest tests.test_text_builder -v
```

Resultado:

```text
Ran 23 tests in 0.005s

OK
```

Se verificaron, entre otros, los siguientes casos:

- estado inicial;
- agregar una predicción;
- construir una palabra;
- construir una frase;
- impedir espacio inicial;
- impedir espacios consecutivos;
- borrar una letra;
- borrar un espacio;
- borrar sobre secuencia vacía;
- limpiar la secuencia;
- finalizar correctamente;
- remover espacio final;
- conservar el contenido después de finalizar;
- impedir finalizar una secuencia vacía;
- conservar Top-k;
- limitar el tamaño de Top-k;
- normalizar letras a mayúsculas;
- rechazar idiomas no soportados;
- rechazar caracteres inválidos;
- rechazar entradas de más de una letra;
- rechazar probabilidades fuera de `[0, 1]`;
- producir una salida serializable.

Todas las pruebas resultaron satisfactorias.

---

## 6. Componente de reconocimiento reutilizable

Se creó:

```text
src/recognition/
├── __init__.py
└── sign_recognizer.py
```

### 6.1 Responsabilidades

`SignRecognizer` encapsula:

- carga del modelo MediaPipe;
- carga del clasificador ASL o LSM;
- carga del `StandardScaler`;
- ejecución de `detect_with_cascade()`;
- extracción de landmarks;
- selección de `img_norm` o `world_norm` según el idioma;
- escalado de características;
- inferencia del modelo;
- generación de Top-k.

### 6.2 Configuración por idioma

#### ASL

- modelo: `models/asl/A3_mlp_img_norm.keras`;
- scaler: `models/asl/scaler_img_norm.joblib`;
- representación utilizada: `img_norm`;
- dimensión de entrada: 63 características.

#### LSM

- modelo: `models/lsm/V2_M2_mlp_world_norm.keras`;
- scaler: `models/lsm/scaler_world_norm.joblib`;
- representación utilizada: `world_norm`;
- dimensión de entrada: 63 características.

Ambos reconocedores producen el mismo contrato de salida.

---

## 7. Flujo integrado de SC-03

Se creó:

```text
scripts/run_sc03_webcam.py
```

Este script funciona como punto de ejecución del subsistema durante su verificación funcional.

Flujo:

```text
Webcam
  ↓
ENTER
  ↓
detect_with_cascade()
  ↓
landmarks
  ↓
img_norm / world_norm
  ↓
StandardScaler
  ↓
A3 ASL / V2-M2 LSM
  ↓
Top-1 + Top-k
  ↓
TextBuilder
  ↓
Texto intermedio estructurado
```

No se aplica inversión horizontal de la imagen, manteniendo la orientación utilizada durante el entrenamiento e inferencia previa.

---

## 8. Pruebas funcionales con webcam

Se ejecutó el flujo integrado tanto para ASL como para LSM.

### 8.1 ASL

El flujo produjo correctamente una secuencia de dos palabras y una salida estructurada completa.

Ejemplo obtenido:

```text
HELLC WCULD
```

La secuencia contenía errores de clasificación ya conocidos del modelo ASL. Entre las capturas se observaron confusiones de geometrías cercanas, pero el sistema:

- agregó cada predicción;
- conservó Top-k;
- insertó espacios;
- permitió edición;
- finalizó correctamente;
- generó la estructura serializable esperada.

La prueba confirmó el funcionamiento del flujo completo de construcción textual aunque el reconocimiento individual conserve las limitaciones ya documentadas del clasificador ASL.

### 8.2 LSM

El flujo produjo:

```text
HOLA MRNDO
```

La mayor parte de la secuencia fue reconocida correctamente. En una de las capturas se obtuvo:

```text
R = 54.70 %
U = 44.82 %
W = 0.31 %
```

Este caso confirmó la utilidad de conservar Top-k, ya que la alternativa correcta puede mantenerse próxima a la predicción Top-1 aun cuando no sea seleccionada directamente.

Al igual que en ASL, el sistema construyó y finalizó correctamente la secuencia textual.

---

## 9. Verificación de controles

Durante las pruebas funcionales se confirmó:

- `ENTER` agrega una única predicción;
- `SPACE` inserta separadores;
- `BACKSPACE` elimina letras;
- `BACKSPACE` elimina espacios;
- `ESC` limpia la secuencia;
- `T` finaliza y produce `BuiltText`;
- ASL y LSM producen la misma estructura de salida;
- Top-k y probabilidades se conservan por letra.

---

## 10. Decisión sobre umbral de confianza

Inicialmente se había contemplado utilizar un umbral de confianza para rechazar predicciones de baja probabilidad.

Después de observar el comportamiento real de los clasificadores se decidió **no aplicar un umbral duro que impida agregar una predicción válida**.

La razón principal es que el vocabulario de traducción será delimitado y el subsistema posterior podrá utilizar:

- texto Top-1;
- alternativas Top-k;
- probabilidades;
- contexto de la palabra o frase;
- vocabulario permitido.

Un umbral fijo podría descartar información útil en casos como:

```text
R = 54.70 %
U = 44.82 %
```

En este caso, rechazar la captura elimina una alternativa potencialmente correcta, mientras que conservar la distribución permite que el componente textual posterior utilice el contexto.

Por lo tanto, la política adoptada es:

```text
detección válida
    ↓
clasificación válida
    ↓
conservar Top-1 + Top-k + probabilidades
    ↓
agregar Top-1 al texto intermedio
    ↓
resolver ambigüedades en el procesamiento textual posterior
```

Los únicos casos que impiden agregar una letra son:

- ausencia de detección válida de mano;
- salida incompleta o inválida del pipeline;
- error de inferencia.

---

## 11. Relación con el procesamiento mediante LLM

La decisión anterior se alinea con el diseño previsto para el siguiente subsistema.

El LLM recibirá una entrada restringida y contextual que podrá incluir:

- texto reconocido;
- idioma de origen;
- vocabulario permitido;
- alternativas Top-k;
- probabilidades por captura.

El objetivo no será permitir una traducción libre, sino seleccionar o corregir la secuencia dentro del vocabulario y reglas definidas por Sign2Sign.

Ejemplo conceptual:

```text
Texto reconocido:
HOLA MRNDO

Posición conflictiva:
Top-1: R 54.70 %
Top-2: U 44.82 %

Vocabulario permitido:
HOLA
MUNDO
GRACIAS
...
```

En este escenario, el contexto y el vocabulario delimitado pueden apoyar la selección de la secuencia válida sin depender de un umbral rígido de confianza.

---

## 12. Estado de SC-03

Con los componentes y verificaciones realizadas, SC-03 cuenta con:

- preprocesamiento de imagen;
- detección de mano;
- extracción de landmarks;
- normalización;
- carga de modelos;
- reconocimiento ASL;
- reconocimiento LSM;
- Top-k y probabilidades;
- constructor de secuencia textual;
- edición de la secuencia;
- generación de texto intermedio estructurado;
- pruebas unitarias del constructor;
- verificación funcional mediante webcam en ambos idiomas.

El subsistema produce una salida común que puede ser consumida por el siguiente bloque de procesamiento textual.

---

## 13. Archivos incorporados

```text
src/
├── recognition/
│   ├── __init__.py
│   └── sign_recognizer.py
│
└── text/
    ├── __init__.py
    └── text_builder.py

scripts/
└── run_sc03_webcam.py

tests/
├── __init__.py
└── test_text_builder.py
```

Los scripts previos de evaluación de webcam se conservan como herramientas de prueba y evidencia experimental; no fueron convertidos en componentes de ejecución del sistema.

---

## 14. Siguiente bloque

El siguiente subsistema a desarrollar es **SC-04 Corrección y traducción textual**.

La entrada disponible desde SC-03 está compuesta por:

```text
texto intermedio
+
idioma de origen
+
Top-k
+
probabilidades
+
vocabulario delimitado
```

El siguiente trabajo deberá definir el contrato de entrada/salida del LLM, el vocabulario permitido, las reglas de corrección y traducción y la forma de restringir la respuesta para evitar resultados fuera del dominio de Sign2Sign.
