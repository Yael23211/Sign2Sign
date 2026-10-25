# Fase SC-05 y SC-06 — Mapeo de recursos visuales y reproducción de salida

**Proyecto:** Sign2Sign  
**Estado:** Cerrada para el flujo local del prototipo  
**Subsistemas:** SC-05 Mapeo visual + SC-06 Generación/reproducción de salida visual  
**Integración alcanzada:** SC-03 → SC-04 → SC-05 → SC-06  
**Direcciones validadas:** LSM → ASL y ASL → LSM  

---

## 1. Objetivo de la fase

El objetivo de esta fase fue completar el tramo final del flujo local de Sign2Sign después de la corrección y traducción textual realizada por SC-04.

La entrada disponible para este bloque es un `TranslationResult` que contiene, entre otros campos:

```text
target_language
concepts
translated_text
unresolved
```

A partir de esta información se implementaron dos responsabilidades independientes:

```text
SC-05
concept_id
→ recurso visual asociado
→ validación de disponibilidad
→ secuencia ordenada de recursos

SC-06
secuencia de recursos
→ validación de reproducción
→ reproducción ordenada
→ presentación de estado al usuario
```

El flujo local completo alcanzado es:

```text
Usuario
  ↓
Webcam
  ↓
SC-03
Reconocimiento + construcción textual
  ↓
BuiltText
  ↓
SC-04
Corrección + traducción + conceptos canónicos
  ↓
TranslationResult
  ↓
SC-05
Mapeo de concept_id a recursos visuales
  ↓
ResourceMappingResult
  ↓
SC-06
Reproducción secuencial
  ↓
Salida visual en la lengua destino
```

---

## 2. Decisiones de diseño consolidadas

Durante esta fase se mantuvieron las siguientes decisiones:

1. SC-05 y SC-06 permanecen desacoplados.
2. SC-05 no reproduce videos; únicamente resuelve recursos.
3. SC-06 no interpreta significado ni modifica conceptos; únicamente reproduce la secuencia recibida.
4. El orden entregado por SC-04 se conserva hasta la salida visual.
5. El prototipo utiliza español signado e inglés signado, por lo que SC-06 no intenta aplicar una gramática completa de LSM o ASL.
6. Todos los conceptos disponibles en el flujo local cuentan con un recurso MP4 directo por `concept_id`.
7. Los recursos locales se consideran la salida principal del prototipo.
8. Los recursos online permanecen como alternativa futura y no forman parte del cierre local de SC-06.
9. Los elementos incluidos en `unresolved` no detienen necesariamente el flujo si SC-04 también produjo conceptos válidos.
10. Ante una falla externa de SC-04, la cadena construida por SC-03 se conserva para permitir un reintento.

---

# Parte I — SC-05: Mapeo de recursos visuales

## 3. Responsabilidad de SC-05

SC-05 transforma la lista de conceptos canónicos producidos por SC-04 en una secuencia de recursos visuales reproducibles.

Ejemplo:

```text
TranslationResult

target_language:
LSM

concepts:
FIRST_PERSON_SINGULAR
WANT
WATER
```

SC-05 produce:

```text
resources/lsm/FIRST_PERSON_SINGULAR.mp4
resources/lsm/WANT.mp4
resources/lsm/WATER.mp4
```

El subsistema no modifica el significado, no traduce nuevamente el texto y no altera el orden de los conceptos.

---

## 4. Estructura implementada para SC-05

Se incorporó:

```text
src/
└── visual_mapping/
    ├── __init__.py
    ├── contracts.py
    └── resource_mapper.py

tests/
└── test_resource_mapper.py

scripts/
├── test_sc05_mapper.py
├── test_sc04_sc05_integration.py
└── run_sc03_sc04_sc05_webcam.py
```

---

## 5. Contratos de SC-05

En `contracts.py` se definieron:

```text
VisualMappingError
InvalidLanguageError
InvalidDeliveryError
InvalidCatalogError
ResourceNotFoundError
ResourceUnavailableError
ResourceFileMissingError
```

También se definieron las estructuras:

```text
VisualResource
ResourceMappingRequest
ResourceMappingResult
```

### 5.1 VisualResource

Representa un recurso individual.

Campos principales:

```text
concept_id
language
delivery
location
type
available
```

### 5.2 ResourceMappingRequest

Representa una solicitud normalizada hacia SC-05.

Contiene:

```text
concepts
target_language
delivery
```

La construcción normaliza:

```text
concept_id → mayúsculas
language   → mayúsculas
delivery   → minúsculas
```

### 5.3 ResourceMappingResult

Representa la salida formal de SC-05.

Contiene:

```text
target_language
delivery
resources
```

Además expone:

```text
locations
concept_ids
to_dict()
```

---

## 6. VisualResourceMapper

El componente principal de SC-05 es:

```text
VisualResourceMapper
```

El catálogo por defecto es:

```text
data/resources/visual_resources.json
```

Al inicializarse:

1. carga el catálogo;
2. valida su estructura;
3. crea un índice interno;
4. rechaza llaves duplicadas;
5. conserva la independencia entre ASL y LSM.

La llave lógica de cada recurso es:

```text
(concept_id, language, delivery)
```

---

## 7. Resolución de recursos

El método principal es:

```python
resolve(
    concepts,
    target_language,
    delivery="local",
)
```

Su comportamiento es:

```text
conceptos de entrada
       ↓
normalización
       ↓
validación idioma/delivery
       ↓
búsqueda de cada recurso
       ↓
validación available
       ↓
validación física si es local
       ↓
ResourceMappingResult
```

SC-05 conserva:

- orden original;
- conceptos repetidos;
- separación ASL/LSM.

---

## 8. Integración directa con SC-04

Se incorporó:

```python
resolve_translation_result(
    translation_result,
    delivery="local",
)
```

El método utiliza directamente:

```text
translation_result.target_language
translation_result.concepts
```

De esta manera el contrato entre ambos subsistemas queda:

```text
SC-04
TranslationResult
       ↓
SC-05
ResourceMappingResult
```

No es necesario volver a pasar manualmente la lengua destino.

---

## 9. Validación de rutas locales

Para recursos con:

```text
delivery = local
```

SC-05 comprueba que:

1. la ruta sea relativa al proyecto;
2. la ruta no pueda escapar del directorio raíz;
3. el archivo exista físicamente;
4. el recurso esté marcado como disponible.

Esto evita aceptar entradas de catálogo que no puedan reproducirse posteriormente.

---

## 10. Pruebas unitarias de SC-05

La suite implementada cubre nueve casos:

```text
1. Independencia ASL / LSM.
2. Normalización de mayúsculas, minúsculas y espacios.
3. Conservación del orden y conceptos repetidos.
4. Detección de archivo físico faltante.
5. Recurso no registrado.
6. Recurso marcado como no disponible.
7. Llave duplicada en el catálogo.
8. Idioma inválido.
9. Resolución individual de recurso LSM.
```

La ejecución realizada fue satisfactoria:

```text
Ran 9 tests
OK
```

---

## 11. Smoke test de SC-05

Se implementó:

```text
scripts/test_sc05_mapper.py
```

Caso utilizado:

```text
target_language = LSM

concepts:
FIRST_PERSON_SINGULAR
WANT
WATER
```

Resultado:

```text
resources/lsm/FIRST_PERSON_SINGULAR.mp4
resources/lsm/WANT.mp4
resources/lsm/WATER.mp4
```

Los recursos fueron localizados correctamente y se verificó su disponibilidad física.

---

## 12. Prueba SC-04 → SC-05

Se implementó:

```text
scripts/test_sc04_sc05_integration.py
```

Se construyó un `TranslationResult` válido con:

```text
source_language:
LSM

target_language:
ASL

recognized_text:
HOLA COMT ESTAA

corrected_text:
HOLA CÓMO ESTÁS

translated_text:
HELLO HOW ARE YOU

concepts:
HELLO
HOW_ARE_YOU

unresolved:
ninguno
```

SC-05 resolvió:

```text
resources/asl/HELLO.mp4
resources/asl/HOW_ARE_YOU.mp4
```

Resultado:

```text
OK: integración SC-04 -> SC-05 correcta
```

---

## 13. Integración SC-03 → SC-04 → SC-05

Posteriormente SC-05 fue incorporado al pipeline real de webcam.

### 13.1 LSM → ASL

Caso observado:

```text
Reconocido:
HOLS COMC ESTAS

Corregido:
HOLA CÓMO ESTÁS

Traducido:
HELLO, HOW ARE YOU

Conceptos:
HELLO
HOW_ARE_YOU

Unresolved:
ninguno
```

SC-05 produjo:

```text
HELLO.mp4
HOW_ARE_YOU.mp4
```

Con ello se comprobó que una secuencia construida mediante webcam podía llegar hasta recursos visuales reales de ASL.

### 13.2 ASL → LSM

También se probó el sentido inverso.

En una prueba previa se obtuvo:

```text
Reconocido:
I WANT WANEU

Corregido:
I WANT

Traducido:
YO QUIERO

Conceptos:
FIRST_PERSON_SINGULAR
WANT

Unresolved:
WANEU
```

SC-05 resolvió únicamente los conceptos válidos:

```text
FIRST_PERSON_SINGULAR.mp4
WANT.mp4
```

Este caso permitió verificar que los elementos parcialmente irresolubles no producen recursos inventados.

---

# Parte II — SC-06: Generación y reproducción de salida visual

## 14. Responsabilidad de SC-06

SC-06 recibe el resultado de SC-05 y reproduce los recursos visuales en el mismo orden.

Ejemplo:

```text
ResourceMappingResult

HELLO.mp4
HOW_ARE_YOU.mp4
```

produce:

```text
reproducir HELLO.mp4
       ↓
esperar finalización
       ↓
reproducir HOW_ARE_YOU.mp4
       ↓
fin de secuencia
```

SC-06 no altera el orden semántico recibido.

---

## 15. Estructura implementada para SC-06

Se incorporó:

```text
src/
└── visual_output/
    ├── __init__.py
    ├── contracts.py
    └── sequence_player.py

tests/
└── test_sequence_player.py

scripts/
├── test_sc06_player.py
└── run_sc03_sc04_sc05_sc06_webcam.py
```

---

## 16. Contratos de SC-06

Se definieron las excepciones:

```text
VisualOutputError
InvalidPlaybackPlanError
UnsupportedDeliveryError
VideoFileMissingError
VideoOpenError
```

También se definieron:

```text
PlaybackItem
PlaybackPlan
PlaybackResult
```

### 16.1 PlaybackItem

Representa un elemento individual de la secuencia.

Campos:

```text
concept_id
language
location
position
total
```

### 16.2 PlaybackPlan

Contiene:

```text
target_language
items
```

y permite conocer:

```text
total
```

### 16.3 PlaybackResult

Conserva el resultado final de la reproducción:

```text
completed
interrupted
played_count
total_count
```

---

## 17. VisualSequencePlayer

El componente principal de SC-06 es:

```text
VisualSequencePlayer
```

Responsabilidades:

- construir el plan de reproducción;
- validar cada recurso;
- abrir los archivos MP4;
- reproducirlos secuencialmente;
- conservar el orden recibido;
- mostrar la lengua destino;
- mostrar el concepto actual;
- mostrar el texto traducido;
- mostrar contenido `unresolved`;
- permitir pausa e interrupción.

---

## 18. Construcción del plan de reproducción

El método:

```python
build_plan(mapping_result)
```

valida que:

1. exista `target_language`;
2. existan `resources`;
3. la secuencia no esté vacía;
4. cada recurso sea local;
5. cada recurso sea de tipo video;
6. cada recurso esté disponible;
7. el archivo exista físicamente.

La versión local de SC-06 rechaza intencionalmente:

```text
delivery = online
```

porque el consumo de recursos externos pertenece a una ampliación posterior.

---

## 19. Reproducción

El método principal es:

```python
play(
    mapping_result,
    translated_text="",
    unresolved=(),
)
```

Los controles implementados durante reproducción son:

```text
SPACE → pausa / continuar
Q     → detener
ESC   → detener
```

Al terminar devuelve un `PlaybackResult`.

---

## 20. Tamaño y presentación

Durante las pruebas se aumentó la superficie de reproducción a:

```text
1280 × 720
```

manteniendo relación:

```text
16:9
```

Los videos se redimensionan conservando el aspect ratio.

Si el video no llena completamente el área disponible, se utiliza un fondo negro tipo letterbox.

---

## 21. Soporte para videos con diferentes FPS

El repositorio contiene videos con diferentes tasas de cuadros.

Se observaron ejemplos aproximados de:

```text
24 FPS
30 FPS
60 FPS
```

Por ello no se fijó una velocidad global para SC-06.

El reproductor obtiene dinámicamente:

```python
cv2.CAP_PROP_FPS
```

para cada archivo.

---

## 22. Problema inicial de velocidad

La primera versión realizaba aproximadamente:

```text
procesamiento del frame
+
waitKey(intervalo completo)
```

Esto provocaba que el tiempo de procesamiento se sumara al intervalo esperado del video.

Por ejemplo:

```text
video de 30 FPS
intervalo ideal ≈ 33.3 ms

procesamiento ≈ varios ms
+
waitKey(33 ms)

→ reproducción ligeramente más lenta
```

El efecto era más perceptible en videos con mayor FPS.

---

## 23. Sincronización dinámica con reloj real

Para corregir este comportamiento se incorporó:

```python
time.perf_counter()
```

La reproducción utiliza un reloj de referencia real.

Conceptualmente:

```text
FPS del archivo
       ↓
duración ideal de frame
       ↓
tiempo objetivo de cada frame
       ↓
comparación con reloj real
       ↓
espera únicamente el tiempo restante
```

Por ejemplo:

```text
24 FPS
≈ 41.67 ms/frame

30 FPS
≈ 33.33 ms/frame

60 FPS
≈ 16.67 ms/frame
```

La sincronización se calcula individualmente para cada video.

Después del cambio, la reproducción observada mejoró de manera considerable.

---

## 24. Pausa y reloj de reproducción

Cuando el usuario pausa el video:

```text
SPACE
```

SC-06 registra el inicio de la pausa.

Al continuar:

```text
SPACE
```

el tiempo pausado se suma al origen del reloj de reproducción.

Con ello la pausa no provoca que todos los frames posteriores se consideren retrasados.

---

## 25. Pruebas aisladas de SC-06

Se definió una suite para validar principalmente:

```text
- conservación de orden;
- posiciones dentro de la secuencia;
- rechazo de delivery online en la versión local;
- detección de archivo faltante.
```

También se creó:

```text
scripts/test_sc06_player.py
```

para ejecutar un smoke test con archivos reales.

---

## 26. Smoke test SC-05 → SC-06

Caso:

```text
HELLO
HOW_ARE_YOU
```

SC-05 resolvió:

```text
resources/asl/HELLO.mp4
resources/asl/HOW_ARE_YOU.mp4
```

SC-06 reprodujo automáticamente:

```text
HELLO.mp4
       ↓
HOW_ARE_YOU.mp4
```

Resultado observado:

```text
Completado: True
Interrumpido: False
Videos reproducidos: 2/2
```

Con esto se comprobó el funcionamiento aislado de:

```text
SC-05 → SC-06
```

---

# Parte III — Integración completa

## 27. Script del pipeline completo

Se creó:

```text
scripts/run_sc03_sc04_sc05_sc06_webcam.py
```

Este script integra:

```text
SC-03
→ SC-04
→ SC-05
→ SC-06
```

sin modificar los scripts anteriores de cada etapa.

---

## 28. Flujo operativo

El flujo de usuario es:

```text
ENTER
capturar letra

SPACE
separar palabras

BACKSPACE
corregir cadena

ESC
limpiar / iniciar nueva cadena

T
finalizar
→ SC-04
→ SC-05
→ SC-06

P
reproducir nuevamente la salida visual

Q
salir
```

---

## 29. Repetición de salida sin nueva llamada al LLM

Después de completar una traducción se permite:

```text
P → reproducir otra vez
```

Esto utiliza el `ResourceMappingResult` ya existente.

Por lo tanto:

```text
P
```

no:

- vuelve a ejecutar SC-03;
- vuelve a llamar a Gemini;
- vuelve a ejecutar la traducción;
- modifica la secuencia.

Únicamente repite SC-06.

---

## 30. Manejo de falla temporal de Gemini

Durante una de las pruebas completas se presentó:

```text
GeminiRequestError

Elapsed time:
21.240 s

Cause:
APIConnectionError:
Server disconnected without sending a response
```

El sistema mostró:

```text
Error en SC-04.
La cadena se conserva.
```

El contenido construido por SC-03 permaneció disponible.

Al volver a ejecutar:

```text
T
```

la llamada funcionó correctamente.

Este comportamiento confirma la política previamente definida:

```text
fallo externo de SC-04
→ no destruir BuiltText
→ permitir reintento
```

SC-05 y SC-06 no se ejecutan cuando SC-04 no produce un `TranslationResult` válido.

---

# Parte IV — Pruebas extremo a extremo

## 31. Prueba completa LSM → ASL

Se realizó una prueba mediante webcam.

### Entrada reconocida

```text
HOLS COMC ESTAA
```

SC-03 introdujo múltiples errores respecto a la intención original.

### SC-04

Resultado:

```text
Corregido:
HOLA CÓMO ESTÁS

Traducido:
HELLO HOW ARE YOU

Conceptos:
HELLO
HOW_ARE_YOU

Unresolved:
ninguno
```

Latencia observada:

```text
2.687 s
```

### SC-05

Recursos:

```text
HELLO.mp4
HOW_ARE_YOU.mp4
```

### SC-06

Resultado:

```text
secuencia completada
2/2
```

Por lo tanto:

```text
LSM
→ webcam
→ reconocimiento
→ corrección
→ traducción
→ ASL
→ recursos
→ reproducción
```

funcionó correctamente.

---

## 32. Prueba completa ASL → LSM

Se realizó posteriormente el flujo inverso.

### Entrada reconocida

```text
I WANT WATEV
```

### SC-04

Resultado:

```text
Corregido:
I WANT WATER

Traducido:
YO QUIERO AGUA

Conceptos:
FIRST_PERSON_SINGULAR
WANT
WATER

Unresolved:
ninguno
```

Latencia observada:

```text
3.063 s
```

### SC-05

Recursos:

```text
FIRST_PERSON_SINGULAR.mp4
WANT.mp4
WATER.mp4
```

### SC-06

Resultado:

```text
secuencia completada
3/3
```

Con ello se comprobó el sentido:

```text
ASL → LSM
```

de extremo a extremo.

---

## 33. Segunda prueba de tolerancia ASL

Se realizó otra prueba con mayor degradación del reconocimiento.

### Entrada reconocida

```text
I WANT WAVAN
```

### SC-04

Resultado:

```text
Corregido:
I WANT WATER

Traducido:
YO QUIERO AGUA

Conceptos:
FIRST_PERSON_SINGULAR
WANT
WATER

Unresolved:
ninguno
```

Latencia observada:

```text
1.455 s
```

### SC-06

Resultado:

```text
secuencia completada
3/3
```

Esta prueba mostró que SC-04 puede recuperar una palabra válida incluso cuando el reconocimiento ASL contiene varios errores, siempre que exista suficiente contexto y el resultado permanezca dentro del vocabulario cerrado.

---

## 34. Observación sobre tolerancia ASL

Durante las pruebas webcam se ha observado mayor variabilidad práctica en el reconocedor ASL que en el reconocedor LSM.

Por ello se considera conveniente conservar una política de corrección contextual relativamente tolerante para ASL.

SC-04 puede aprovechar:

```text
- contexto de la oración;
- alternativas Top-k;
- probabilidades de clasificación;
- vocabulario cerrado;
- longitud de palabra;
- similitud ortográfica;
- distancia de edición.
```

No obstante, no se implementó una regla adicional específica durante esta fase porque el comportamiento actual ya logró corregir satisfactoriamente casos como:

```text
WATEV → WATER
WAVAN → WATER
```

sin producir conceptos fuera del vocabulario.

Como mejora futura puede evaluarse una política explícitamente diferenciada por lengua origen.

---

## 35. Orden lingüístico de la secuencia visual

SC-06 reproduce los conceptos en el mismo orden entregado por SC-04.

No realiza reordenamiento para intentar aproximarse a la gramática completa de ASL o LSM.

Por ejemplo:

```text
I WANT WATER

→ FIRST_PERSON_SINGULAR
→ WANT
→ WATER
```

y:

```text
YO QUIERO AGUA

→ FIRST_PERSON_SINGULAR
→ WANT
→ WATER
```

Esta decisión es consistente con la limitación del prototipo:

```text
español signado
+
inglés signado
```

La implementación no pretende constituir una transformación gramatical normativa de LSM o ASL.

---

## 36. Tratamiento de unresolved

El contrato de SC-04 conserva:

```text
unresolved
```

para contenido que no puede representarse de forma segura mediante el vocabulario.

SC-05 sólo recibe y resuelve:

```text
concepts
```

Por lo tanto:

```text
contenido resuelto
→ puede producir videos

contenido unresolved
→ se conserva como advertencia
→ no produce recursos inventados
```

SC-06 puede mostrar los elementos `unresolved` durante la presentación visual.

---

# Parte V — Estado de cierre

## 37. Estado de SC-05

Checklist:

```text
[x] Catálogo de recursos consumido.
[x] Contratos formales definidos.
[x] ASL y LSM independientes.
[x] Rutas locales validadas.
[x] Archivos físicos validados.
[x] Orden de conceptos preservado.
[x] Conceptos repetidos preservados.
[x] Errores controlados.
[x] Integración directa con TranslationResult.
[x] Pruebas unitarias.
[x] Smoke test con catálogo real.
[x] Integración SC-04 → SC-05.
[x] Integración webcam SC-03 → SC-04 → SC-05.
[x] Validación LSM → ASL.
[x] Validación ASL → LSM.
```

**Resultado:** SC-05 local cerrado.

---

## 38. Estado de SC-06

Checklist:

```text
[x] Contratos formales definidos.
[x] Construcción de plan de reproducción.
[x] Validación de recursos locales.
[x] Reproducción secuencial.
[x] Conservación de orden.
[x] Presentación de concepto actual.
[x] Presentación de lengua destino.
[x] Presentación de texto traducido.
[x] Soporte de unresolved.
[x] Pausa / continuar.
[x] Interrupción controlada.
[x] Repetición de salida.
[x] Soporte dinámico de FPS.
[x] Sincronización mediante reloj real.
[x] Videos de diferentes resoluciones.
[x] Smoke test con MP4 reales.
[x] Integración SC-05 → SC-06.
[x] Integración completa mediante webcam.
[x] LSM → ASL validado.
[x] ASL → LSM validado.
```

**Resultado:** SC-06 local cerrado.

---

## 39. Flujo funcional alcanzado

Al cierre de esta fase, Sign2Sign dispone de:

```text
Usuario
  ↓
Webcam
  ↓
SC-03
  ↓
detección
  ↓
landmarks
  ↓
clasificador ASL / LSM
  ↓
Top-1 + Top-k
  ↓
TextBuilder
  ↓
BuiltText
  ↓
SC-04
  ↓
corrección
  ↓
traducción
  ↓
conceptos canónicos
  ↓
SC-05
  ↓
mapeo visual
  ↓
ResourceMappingResult
  ↓
SC-06
  ↓
reproducción ordenada
  ↓
videos de la lengua destino
```

El núcleo local de traducción Sign2Sign funciona en ambos sentidos previstos:

```text
LSM → ASL
ASL → LSM
```

bajo las condiciones controladas del prototipo.

---

## 40. Limitaciones actuales

Permanecen las siguientes limitaciones:

- el reconocimiento se basa principalmente en deletreo manual estático;
- el clasificador ASL presenta mayor variabilidad durante uso real con webcam;
- SC-04 depende de conectividad y de un servicio LLM externo;
- una falla temporal del servicio externo puede retrasar la traducción;
- la secuencia visual conserva el orden de español/inglés signados;
- no se implementa gramática normativa completa de ASL o LSM;
- SC-06 local todavía no consume recursos online;
- la reproducción se realiza actualmente mediante OpenCV como implementación funcional del prototipo;
- los recursos visuales permanecen como clips independientes;
- no se genera animación continua ni transición lingüística automática entre señas;
- algunos videos pueden tener características heterogéneas de resolución, FPS o codificación.

---

## 41. Trabajo futuro identificado

Las mejoras posteriores pueden incluir:

```text
1. Política diferenciada de tolerancia SC-04 para ASL.
2. Uso explícito de longitud de palabra y distancia de edición.
3. Mayor aprovechamiento de Top-k durante reconstrucción ASL.
4. Recursos online como alternativa opcional.
5. Integración con backend/orquestador.
6. Exposición de recursos mediante FastAPI.
7. Interfaz final React.
8. Reproductor visual integrado al frontend.
9. Pruebas funcionales y de sistema más amplias.
10. Métricas de latencia extremo a extremo.
11. Evaluación de casos parcialmente unresolved.
12. Mejoras de transición visual entre clips.
```

Estas mejoras no impiden considerar cerrado el flujo local de SC-05 y SC-06.

---

## 42. Archivos principales al cierre de la fase

```text
data/
└── resources/
    └── visual_resources.json

resources/
├── asl/
│   └── *.mp4
└── lsm/
    └── *.mp4

src/
├── visual_mapping/
│   ├── __init__.py
│   ├── contracts.py
│   └── resource_mapper.py
│
└── visual_output/
    ├── __init__.py
    ├── contracts.py
    └── sequence_player.py

tests/
├── test_resource_mapper.py
└── test_sequence_player.py

scripts/
├── test_sc05_mapper.py
├── test_sc04_sc05_integration.py
├── run_sc03_sc04_sc05_webcam.py
├── test_sc06_player.py
└── run_sc03_sc04_sc05_sc06_webcam.py
```

---

## 43. Conclusión de la fase

SC-05 y SC-06 completan el tramo de salida visual del prototipo Sign2Sign.

SC-05 permite transformar los conceptos semánticos controlados producidos por SC-04 en recursos visuales verificables, conservando idioma, disponibilidad y orden.

SC-06 consume esa secuencia y reproduce los videos de manera ordenada, con controles básicos de usuario y soporte para archivos con diferentes tasas de cuadros.

La integración completa fue comprobada mediante webcam en ambos sentidos:

```text
LSM → ASL
ASL → LSM
```

incluyendo casos donde SC-03 introdujo errores de reconocimiento que fueron corregidos posteriormente por SC-04.

Con ello, Sign2Sign dispone por primera vez de un flujo local funcional de extremo a extremo:

```text
seña
→ reconocimiento
→ texto
→ corrección
→ traducción
→ conceptos
→ recursos visuales
→ videos de salida
```

La fase SC-05 + SC-06 se considera cerrada para el alcance actual del prototipo.

---

**Fin del documento de fase SC-05 + SC-06.**
