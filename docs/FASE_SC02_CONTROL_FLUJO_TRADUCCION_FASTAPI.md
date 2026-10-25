# Fase SC-02 — Control del flujo de traducción mediante FastAPI

**Proyecto:** Sign2Sign  
**Fecha de documentación:** 25 de octubre de 2026  
**Estado:** Implementación local completada y pruebas aprobadas; pendiente de revisión e integración con SC-01  
**Subsistema principal:** SC-02 Control del flujo de traducción  
**Soporte temporal:** SC-07 Gestión de datos temporales y recursos  
**Integración alcanzada:** HTTP → SC-03 → SC-04 → SC-05 → preparación SC-06  
**Interfaz React:** pendiente; no implementada en esta fase  

Continuación del trabajo descrito en:

- [Fase SC-03: constructor textual e integración](FASE_SC03_CONSTRUCTOR_TEXTUAL_E_INTEGRACION.md).
- [Fase SC-04: corrección, traducción e integración](FASE_SC04_CORRECCION_TRADUCCION_E_INTEGRACION.md).
- [Fase SC-07: recursos visuales y registro local](FASE_SC07_RECURSOS_VISUALES_Y_REGISTRO_LOCAL.md).
- [Fase SC-05 y SC-06: mapeo y salida visual](FASE_SC05_SC06_MAPEO_Y_SALIDA_VISUAL.md).

Los resultados de pruebas consignados corresponden a las ejecuciones realizadas durante la implementación de SC-02. La elaboración de este documento no constituye una nueva validación con cámara, navegador o servicio LLM real.

---

## 1. Objetivo de la fase

El objetivo fue incorporar una capa HTTP local para coordinar el núcleo de traducción ASL ↔ LSM, reutilizando sus contratos y componentes actuales.

Antes de esta fase, el flujo se coordinaba mediante scripts interactivos de webcam, controles de teclado y ventanas OpenCV. SC-02 permite invocar directamente los componentes desde FastAPI, mantener sesiones temporales y devolver una salida consumible por el futuro frontend.

```text
Petición HTTP
    ↓
SC-02: validación y control del flujo
    ↓
SC-03: reconocimiento y construcción textual
    ↓
SC-04: corrección, traducción y conceptos canónicos
    ↓
SC-05: mapeo de recursos visuales
    ↓
SC-06: construcción del plan de reproducción
    ↓
Respuesta JSON con textos y secuencia de URLs HTTP
```

SC-07 mantiene las sesiones temporales y proporciona acceso al repositorio local. La reproducción en navegador corresponde a una fase posterior.

## 2. Arquitectura disponible al inicio

| Subsistema | Componentes existentes reutilizados |
|---|---|
| SC-03 | `SignRecognizer`, `RecognitionResult`, pipeline de landmarks, modelos y escaladores |
| SC-03: construcción textual | `TextBuilder`, `BuiltText`, `TextToken`, `PredictionCandidate` |
| SC-04 | `CorrectionRequest`, `GeminiClient`, `Vocabulary`, `PromptBuilder`, `ResponseValidator`, `TranslationResult` |
| SC-05 | `VisualResourceMapper`, `ResourceMappingResult` y contratos relacionados |
| SC-06 | `VisualSequencePlayer`, `PlaybackPlan`, `PlaybackItem` |
| SC-07 | `ResourceRepository`, catálogo visual y videos locales |

La revisión previa comprobó 71 conceptos canónicos y 142 recursos locales: 71 ASL y 71 LSM. Los archivos registrados como disponibles y los artefactos de reconocimiento utilizados por el script completo estaban presentes.

SC-05 y SC-07 comparten el catálogo, pero mantienen lectores e índices independientes. SC-05 no fue refactorizado para utilizar internamente `ResourceRepository` en esta fase.

## 3. Decisiones de alcance

Las decisiones aprobadas fueron:

1. Implementar únicamente el backend SC-02; posponer SC-01 en React.
2. Reutilizar las clases y contratos de los subsistemas existentes.
3. Agregar el almacenamiento temporal dentro de SC-07.
4. Mantener sesiones en memoria y ejecutar un solo worker.
5. Mantener abierta la petición de traducción hasta devolver el resultado final.
6. Ejecutar operaciones bloqueantes fuera del event loop.
7. Cargar una sola vez las dependencias pesadas mediante lifespan.
8. Serializar el acceso a componentes compartidos cuando sea necesario.
9. Mantener captura deliberada de una imagen por acción.
10. Construir el plan de salida sin abrir ventanas OpenCV.
11. Entregar videos mediante URLs HTTP del backend.
12. Conservar los scripts de integración como herramientas de validación.

No se incorporaron Redis, Celery, bases de datos persistentes, WebSockets, SSE, múltiples workers, trabajos con `202 Accepted`, `operation_id` ni polling.

---

# Parte I — Arquitectura y archivos

## 4. Responsabilidades de los subsistemas

| Subsistema | Responsabilidad |
|---|---|
| SC-01 | Futura interacción, captura desde navegador y presentación web |
| SC-02 | Validación HTTP, coordinación de sesiones, concurrencia, invocación del núcleo y entrega del resultado |
| SC-03 | Reconocimiento y construcción de texto con confianza y Top-k |
| SC-04 | Corrección, traducción y generación de conceptos canónicos |
| SC-05 | Resolución de conceptos a recursos visuales |
| SC-06 | Generación de salida visual y construcción del plan de reproducción |
| SC-07 | Gestión de datos temporales y recursos |

SC-02 no agrega reglas lingüísticas, inferencia alternativa, selección semántica de videos ni reordenamiento de conceptos.

## 5. Estructura agregada

```text
src/
├── api/
│   ├── __init__.py
│   ├── main.py
│   ├── settings.py
│   ├── schemas.py
│   ├── dependencies.py
│   ├── routes.py
│   └── translation_controller.py
│
└── resources/
    ├── resource_repository.py          # Existente, conservado
    └── temporary_session_store.py      # Nuevo soporte temporal de SC-07

tests/
└── test_sc02.py
```

| Archivo nuevo | Responsabilidad |
|---|---|
| `src/api/__init__.py` | Define el paquete SC-02 |
| `src/api/main.py` | Expone `create_app()` y `app`, configura lifespan, rutas y manejo HTTP de `FlowError` |
| `src/api/settings.py` | Define raíz del proyecto, límites y selección del modelo Gemini |
| `src/api/schemas.py` | Contratos Pydantic de lenguas, comandos, imágenes, estados, sesiones y resultados |
| `src/api/dependencies.py` | Carga y agrupa componentes compartidos; libera reconocedores y cliente Gemini |
| `src/api/routes.py` | Ocho endpoints y recepción limitada de capturas |
| `src/api/translation_controller.py` | Coordina captura, edición, traducción, preparación visual y errores |
| `src/resources/temporary_session_store.py` | Define `TranslationSession` y `TemporarySessionStore` |
| `tests/test_sc02.py` | Pruebas del controlador y API con dependencias sustituibles |

Las pruebas del controlador y de la API se agruparon en un archivo, con clases separadas.

## 6. Archivos existentes actualizados y conservados

Se actualizaron:

- `requirements.txt`: dependencias del backend y dependencias del núcleo que faltaban por declarar.
- `.env.example`: configuración de Gemini sin credenciales reales.
- `README.md`: instalación, arranque, endpoints, comportamiento, límites y pruebas.

Este documento registra posteriormente el trabajo en `docs`.

No se modificaron las implementaciones existentes de SC-03, SC-04, SC-05, SC-06 ni `ResourceRepository`. Los scripts interactivos no fueron eliminados ni reescritos. FastAPI importa los componentes de `src`, no los scripts de webcam.

---

# Parte II — Inicialización y concurrencia

## 7. Ciclo de vida de FastAPI

El lifespan prepara las dependencias antes de atender peticiones:

```text
Configuración
    ↓
Vocabulary y GeminiClient
    ↓
VisualResourceMapper
    ↓
VisualSequencePlayer y ResourceRepository
    ↓
Reconocedores ASL y LSM
    ↓
TemporarySessionStore y TranslationController
    ↓
Servicio disponible
```

`load_components()` se ejecuta mediante `run_in_threadpool`. Los imports de reconocimiento se realizan dentro de esa función para permitir pruebas con dobles sin cargar TensorFlow.

Al cerrar el proceso se vacía el almacén y se llama a `Components.close()`, que libera los reconocedores y cierra el cliente Gemini cuando dispone de ese método. Si la inicialización falla después de cargar componentes, se intenta cerrar lo ya creado.

Una dependencia faltante impide completar el arranque. Los modelos no se descargan automáticamente.

## 8. Artefactos reutilizados

| Lengua | Modelo | Escalador |
|---|---|---|
| ASL | `models/asl/A3_mlp_img_norm.keras` | `models/asl/scaler_img_norm.joblib` |
| LSM | `models/lsm/V2_M2_mlp_world_norm.keras` | `models/lsm/scaler_world_norm.joblib` |

Ambos reconocedores utilizan `models/mediapipe/hand_landmarker.task` y `top_k_size=3`.

Los modelos entrenados deben aprovisionarse localmente porque sus formatos están excluidos del control de versiones existente.

## 9. Operaciones bloqueantes

Las rutas declaradas con `def`, incluida la traducción, se ejecutan en el threadpool de FastAPI.

La captura utiliza una ruta asíncrona para recibir el cuerpo por fragmentos y limitar su tamaño. Después delega al controlador mediante `run_in_threadpool`:

```text
Recepción HTTP asíncrona y limitada
    ↓
Threadpool
    ↓
Verificación de imagen + decodificación + inferencia
```

`GeminiClient.process()` también se ejecuta dentro del threadpool. La petición del cliente permanece abierta hasta obtener el resultado, mientras el event loop puede atender otras solicitudes.

## 10. Estrategia de bloqueos

| Bloqueo | Alcance y comportamiento |
|---|---|
| `TemporarySessionStore._lock` | Protege el diccionario de sesiones con `RLock` |
| `TranslationSession.lock` | Protege una sesión; adquisición sin espera y HTTP 409 si está ocupada |
| `Components.recognition_lock` | Serializa la inferencia de los reconocedores ASL y LSM |
| `Components.translation_lock` | Serializa el flujo compartido SC-04 → SC-05 → preparación SC-06 |

Durante una traducción se mantiene bloqueada su sesión, incluso si espera el bloqueo global. Una consulta, captura, comando, traducción adicional o eliminación simultánea de esa sesión recibe 409.

El endpoint de salud y la creación de otras sesiones pueden responder durante una llamada LLM bloqueante. Esta política prioriza simplicidad y seguridad. La espera de bloqueos globales ocupa hilos y limita el rendimiento bajo carga; no se implementó una solución de alta concurrencia.

---

# Parte III — Estado temporal en SC-07

## 11. Datos de cada sesión

```text
TranslationSession
├── id
├── source_language
├── target_language
├── builder
├── state
├── last_result
└── lock
```

El identificador se genera con `uuid4().hex`. La lengua destino se deriva automáticamente como la opuesta:

```text
ASL → LSM
LSM → ASL
```

Cada sesión tiene un `TextBuilder` propio. No se comparten cadenas ni tokens entre sesiones.

`last_result` conserva el resultado final, incluyendo una traducción textual válida cuando falla la preparación visual. Al iniciar otra traducción se limpia el resultado anterior. Si falla SC-04, el resultado queda vacío y el builder conserva la entrada para reintentar.

## 12. Estados implementados

| Estado | Significado |
|---|---|
| `capturing` | Disponible para construir o editar la cadena |
| `processing` | Traducción en curso o esperando el bloqueo compartido |
| `ready` | Resultado válido, que puede contener `unresolved` |
| `failed` | Fallo de SC-04 o de preparación visual |
| `closed` | Estado interno transitorio durante la eliminación |

```text
capturing → processing → ready
                       ↘ failed

ready / failed → captura válida o comando → capturing

sesión disponible → DELETE → closed → eliminación
```

Una captura sin mano no agrega letras ni invalida el resultado anterior. Los comandos válidos limpian `last_result` y dejan `capturing`, incluso si no cambian efectivamente el texto.

No hay polling de `processing`: la petición de traducción espera su resultado y las consultas simultáneas de la sesión ocupada se rechazan con 409.

## 13. Límites y limpieza

| Límite | Valor predeterminado |
|---|---|
| Sesiones almacenadas | 100 |
| Letras por cadena | 500 |
| Captura | 5 MiB: 5 × 1024 × 1024 bytes |
| Área de imagen | 16,777,216 píxeles: 4096 × 4096 |

El límite de imagen se aplica a `ancho × alto`; no establece un máximo individual de 4096 por dimensión. Los límites se definen en `Settings`, sin variables de entorno específicas en esta versión.

La limpieza ocurre mediante `DELETE` o al cerrar el proceso. No hay persistencia ni expiración automática. Las sesiones abandonadas cuentan contra el límite hasta eliminarlas o reiniciar.

Las imágenes se procesan en memoria y no se guardan como archivos. Los videos permanentes no se eliminan con las sesiones.

---

# Parte IV — Contrato HTTP

## 14. Endpoints implementados

Todas las rutas utilizan `/api/v1`.

| Método | Ruta | Resultado |
|---|---|---|
| GET | `/health` | Disponibilidad local |
| POST | `/sessions` | Creación, HTTP 201 |
| GET | `/sessions/{id}` | Estado, texto y último resultado |
| POST | `/sessions/{id}/captures` | Reconocimiento de una imagen |
| POST | `/sessions/{id}/commands` | Edición de la cadena |
| POST | `/sessions/{id}/translations` | Resultado final del flujo |
| DELETE | `/sessions/{id}` | Eliminación, HTTP 204 |
| GET | `/resources/{language}/{concept_id}` | Video MP4 registrado y disponible |

## 15. Salud y creación de sesión

`GET /api/v1/health` devuelve:

```json
{
  "status": "ok",
  "storage": "memory",
  "workers": 1
}
```

No llama a Gemini ni comprueba conectividad externa. El campo `workers` describe la configuración prevista; no impide iniciar accidentalmente varios procesos.

Creación de sesión:

```http
POST /api/v1/sessions
Content-Type: application/json
```

```json
{
  "source_language": "LSM"
}
```

Ejemplo ilustrativo de respuesta:

```json
{
  "id": "identificador-de-sesion",
  "source_language": "LSM",
  "target_language": "ASL",
  "state": "capturing",
  "text": "",
  "last_result": null
}
```

Las lenguas admitidas son exactamente `ASL` y `LSM`. Pydantic rechaza campos adicionales en creación y comandos.

## 16. Captura deliberada

```http
POST /api/v1/sessions/{id}/captures
Content-Type: image/png

[bytes PNG]
```

También se admite `image/jpeg`. Se envía el archivo binario directamente: no se usa multipart ni base64 en JSON y no se requiere `python-multipart`.

Durante la recepción se comprueba el tamaño acumulado. El controlador verifica el tipo declarado, el formato real, la integridad, el área y la posibilidad de decodificación.

```text
CaptureImage
    ↓
Pillow: verificación de formato e integridad
    ↓
OpenCV: decodificación BGR
    ↓
SignRecognizer.recognize(frame)
    ↓
RecognitionResult o None
    ↓
TextBuilder.add_prediction(letter, confidence, top_k)
```

La respuesta contiene `session` y `recognition`. Si no hay mano, `recognition` es `null` y no cambia la cadena. Una detección válida conserva letra, confianza, Top-k, lengua y variante entregadas por SC-03.

No se agregó un umbral de aceptación adicional ni reconocimiento continuo por video.

## 17. Orientación y color

La decodificación utiliza:

```text
cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
```

Produce una imagen BGR compatible con el pipeline existente. El backend no aplica reflejo horizontal ni rotación EXIF.

La futura vista previa puede mostrarse espejada mediante CSS, pero debe enviar los píxeles en la orientación acordada con SC-03. Esto requiere validación con capturas reales del navegador al integrar SC-01.

## 18. Comandos de edición

```http
POST /api/v1/sessions/{id}/commands
Content-Type: application/json
```

```json
{
  "command": "space"
}
```

| Comando | Método existente |
|---|---|
| `space` | `TextBuilder.add_space()` |
| `backspace` | `TextBuilder.backspace()` |
| `clear` | `TextBuilder.clear()` |

Se conservan las reglas del builder para espacios iniciales, repetidos y borrado de cadenas vacías. SC-02 no reconstruye manualmente los tokens.

## 19. Traducción y contratos internos

`POST /api/v1/sessions/{id}/translations` no necesita cuerpo adicional. Usa la cadena de la sesión y ejecuta:

```text
TextBuilder.finalize()
    ↓ BuiltText
CorrectionRequest.from_built_text(..., target_language)
    ↓ CorrectionRequest
GeminiClient.process(request)
    ↓ GeminiCallResult.result: TranslationResult
VisualResourceMapper.resolve_translation_result(..., delivery="local")
    ↓ ResourceMappingResult
VisualSequencePlayer.build_plan(mapping)
    ↓ PlaybackPlan
Secuencia HTTP para el cliente
```

Se omite el tramo visual cuando no hay conceptos. Nunca se invoca `VisualSequencePlayer.play()`.

## 20. Respuesta de traducción

Ejemplo ilustrativo; textos y latencia dependen de la entrada y del servicio LLM:

```json
{
  "source_language": "LSM",
  "target_language": "ASL",
  "recognized_text": "HOLA",
  "corrected_text": "HOLA",
  "translated_text": "HELLO",
  "concepts": ["HELLO"],
  "unresolved": [],
  "resources": [
    {
      "concept_id": "HELLO",
      "language": "ASL",
      "url": "/api/v1/resources/ASL/HELLO",
      "position": 1,
      "total": 1
    }
  ],
  "latency_seconds": 0.25,
  "state": "ready",
  "error": null
}
```

Las posiciones comienzan en 1. Se conservan orden y repeticiones del plan. La latencia procede de SC-04: no representa el tiempo total HTTP, la espera por bloqueos ni la preparación visual.

Las URLs son relativas al origen del backend. No se publican ubicaciones físicas absolutas ni `raw_response` de Gemini.

## 21. Recursos visuales mediante SC-07

```http
GET /api/v1/resources/ASL/HELLO
```

El controlador consulta `ResourceRepository.get_available_resource()` para entrega local y obtiene la ruta con `resolve_path()`. El cliente elige lengua y concepto, no una ubicación arbitraria del sistema.

Se utiliza `FileResponse` con `video/mp4`. Se comprobó soporte de solicitudes Range con respuesta 206. Un cliente puede repetir los videos sin invocar nuevamente a Gemini.

La entrega HTTP no garantiza compatibilidad de todos los códecs con el navegador.

---

# Parte V — Manejo de errores y resultados parciales

## 22. Fallo de SC-04 y reintento

Si falla SC-04, la sesión queda en `failed`, se conserva el builder y no se ejecutan SC-05 ni SC-06. La respuesta HTTP es 502:

```json
{
  "error": {
    "code": "sc04_failed",
    "message": "SC-04 correction/translation failed; input preserved",
    "stage": "SC-04"
  }
}
```

Puede reintentarse la misma cadena. Aunque el núcleo distingue errores de comunicación y validación, esta primera capa HTTP agrupa los fallos de SC-04 en un único código.

## 23. Contenido unresolved

Si únicamente existe contenido no resuelto:

```text
concepts = []
unresolved = [contenido no resuelto]
resources = []
state = ready
```

No se invocan el mapper ni un plan vacío. Si hay conceptos y `unresolved`, se conservan ambos y solo se generan videos para los conceptos válidos.

## 24. Fallo visual

Un fallo de SC-05 o de `build_plan()` conserva el resultado textual validado de SC-04 en la respuesta y en `last_result`.

Se devuelve HTTP 200 con `state: failed` y un error que identifica la etapa. Ejemplo de los campos de error dentro de la respuesta completa:

```json
{
  "state": "failed",
  "error": {
    "stage": "SC-05",
    "code": "visual_preparation_failed",
    "message": "SC-05 visual preparation failed; SC-04 result preserved"
  }
}
```

Si falla el plan, la etapa es `SC-06`. SC-05 conserva su política de fallo ante el primer recurso no resoluble; no se implementa mapeo parcial y la lista de recursos permanece vacía ante ese error.

## 25. Códigos HTTP relevantes

| Código | Uso |
|---|---|
| 200 | Consulta, captura, comando y traducción; revisar `state` y `error` |
| 201 | Sesión creada |
| 204 | Sesión eliminada |
| 206 | Entrega parcial de video mediante Range |
| 404 | Sesión inexistente o recurso no disponible |
| 409 | Sesión ocupada |
| 413 | Captura mayor al límite |
| 415 | Tipo de contenido no admitido |
| 422 | Validación, imagen inválida, texto vacío o límite de letras |
| 502 | Fallo de SC-04 |
| 503 | Capacidad máxima de sesiones |
| 500 | Fallo inesperado durante la traducción |

Los errores controlados no publican mensajes internos de excepciones. Los errores estándar de Pydantic/FastAPI mantienen su formato de validación propio.

---

# Parte VI — Configuración y ejecución

## 26. Dependencias declaradas

Se agregaron las siguientes versiones presentes en el entorno de implementación:

```text
fastapi==0.141.1
uvicorn==0.53.0
pydantic==2.13.5
tensorflow==2.21.0
google-genai==2.25.0
httpx==0.28.1
```

TensorFlow y Google Gen AI ya eran necesarios para el núcleo, pero faltaban por declarar. `httpx` se usa en pruebas de API y también es dependencia del SDK de Google.

La validación utilizó el entorno virtual existente; no equivale a una instalación limpia de todos los requisitos en otro equipo.

## 27. Variables de entorno y artefactos

`.env.example` conserva `SIGN2SIGN_DATA_ROOT` y documenta:

```dotenv
GEMINI_API_KEY=
GEMINI_MODEL=gemini-3.5-flash-lite
```

La clave es obligatoria para inicializar el cliente real. El modelo es opcional y debe ser compatible con SC-04. No se incorporaron credenciales reales.

Los artefactos del backend se resuelven desde la raíz del proyecto, no desde `SIGN2SIGN_DATA_ROOT`.

## 28. Inicio y documentación HTTP

Desde la raíz del repositorio:

```powershell
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --workers 1
```

Documentación interactiva:

```text
http://127.0.0.1:8000/docs
```

OpenAPI describe el cuerpo de captura como binario JPEG o PNG.

El servicio es local, sin autenticación ni CORS habilitado. Debe ejecutarse con un worker: varios procesos tendrían almacenes de sesiones independientes.

---

# Parte VII — Validación realizada

## 29. Organización de pruebas

Las pruebas nuevas se encuentran en `tests/test_sc02.py`, organizadas en `ControllerTests` y `ApiTests`.

La fábrica de `create_app()` admite componentes sustituibles. Las pruebas automáticas usan mocks para Gemini y reconocimiento; no requieren cámara física, credenciales ni llamadas reales.

Se reutiliza `build_plan()` con archivos temporales ficticios para comprobar contratos y entrega HTTP. Esto no prueba la decodificación real de video.

## 30. Pruebas del controlador

Las 11 pruebas verifican:

1. Confianza, Top-k, color BGR y ausencia de espejo mediante imagen asimétrica.
2. Contratos enviados a SC-04, URLs del plan y ausencia de llamadas a `play()`.
3. Conservación de entrada ante fallo SC-04 y reintento.
4. Omisión visual cuando solo hay `unresolved`.
5. Conservación conjunta de conceptos y `unresolved`.
6. Conservación textual e identificación de etapa ante fallos SC-05 y SC-06.
7. Cadena intacta cuando no hay mano.
8. Aislamiento de sesiones y rechazo de operaciones simultáneas.
9. Límites de imagen y letras.
10. Serialización del reconocimiento entre sesiones.
11. Rechazo de traducción vacía.

## 31. Pruebas de API

Las 5 pruebas verifican:

1. Endpoints, comandos, traducción, video, Range y eliminación.
2. Limpieza del almacén y cierre de dependencias al terminar lifespan.
3. Error HTTP seguro de SC-04 y entrada conservada.
4. Lenguas, comandos, imágenes, tipos, tamaño, capacidad y recursos ausentes.
5. Salud y creación de otra sesión durante una llamada bloqueante; HTTP 409 ante edición o eliminación de la sesión ocupada.

## 32. Resultado de la suite completa

Comandos ejecutados durante la implementación:

```powershell
python -m unittest discover -s tests -v
python -m unittest discover -s tests -q
```

Resultado final registrado:

```text
Ran 103 tests
OK
```

| Grupo | Cantidad |
|---|---|
| Pruebas existentes | 87 |
| Pruebas nuevas SC-02 | 16 |
| Total aprobado | 103 |

`git diff --check` tampoco detectó errores de whitespace durante la implementación.

## 33. Comprobación con componentes reales

Se realizó una prueba adicional de inicialización con reconocedores ASL/LSM, modelos, escaladores, mapper, constructor de plan y repositorio reales. Únicamente se sustituyó `GeminiClient` para evitar una llamada externa.

Se comprobó:

- Carga de ambos reconocedores.
- Resolución de `HELLO` para ASL.
- Construcción de un plan de un elemento.
- Disponibilidad local del recurso.
- Cierre de componentes al terminar.

Esta comprobación no incluyó captura física, inferencia de una mano real, traducción LLM real ni reproducción web.

## 34. Advertencias observadas

Starlette emitió una advertencia de deprecación del adaptador `TestClient` para `httpx`, sugiriendo `httpx2`. Las pruebas finalizaron correctamente con las versiones declaradas.

La carga real produjo mensajes de TensorFlow y MediaPipe, incluida la ausencia de soporte GPU nativo de esa versión de TensorFlow en Windows. La carga de modelos y construcción del plan terminaron correctamente.

---

# Parte VIII — Integración futura y cierre

## 35. Comunicación prevista con SC-01

```text
Crear sesión
    ↓
Mostrar vista previa de cámara
    ↓
Enviar una imagen al solicitar captura
    ↓
Mostrar texto y Top-k devueltos
    ↓
Enviar comandos de edición
    ↓
Solicitar traducción y esperar el resultado final
    ↓
Mostrar textos y advertencias
    ↓
Reproducir los videos en el orden recibido
    ↓
Eliminar sesión
```

El futuro cliente podrá enviar un `Blob` con su tipo MIME como cuerpo de `fetch`. La reproducción será una adaptación web de la presentación de SC-06 integrada con SC-01; SC-02 entrega recursos y orden.

El cliente debe revisar `state` y `error`, además del código HTTP. Para Vite se prevé un proxy `/api`; si se usan orígenes diferentes sin proxy, habrá que configurar CORS.

## 36. Limitaciones y pendientes

- SC-01 en React permanece pendiente.
- Falta validar permisos de cámara, orientación y reproducción de los MP4 en navegador.
- No hay autenticación, persistencia ni expiración automática.
- Solo se admite un worker; no se realizaron pruebas de carga.
- La serialización limita la capacidad de procesamiento concurrente.
- El timeout LLM sigue en el cliente existente de SC-04; no se agregó cancelación cooperativa al desconectarse HTTP.
- No hay identificadores de operación ni deduplicación: reenviar una captura puede agregar otra letra y reenviar una traducción puede llamar de nuevo a Gemini.
- La API agrupa las causas de fallo de SC-04 en un código.
- SC-05 conserva su política de fallo completo ante un recurso faltante.
- SC-05 y SC-07 mantienen índices separados; un cambio externo de catálogo puede requerir reinicio para mantener una vista consistente.
- El arranque requiere credenciales y artefactos locales; falta comprobar instalación limpia en otro entorno.
- Queda pendiente revisar la advertencia de Starlette al actualizar dependencias.

Se mantienen las limitaciones lingüísticas del prototipo: deletreo principalmente estático, vocabulario cerrado, dependencia LLM y ausencia de gramática normativa completa ASL/LSM.

## 37. Evolución del controlador

`TranslationController` no depende de `Request` ni de un sistema de trabajos. Recibe almacén, componentes y configuración.

Una futura capa de ejecución diferida podrá invocarlo conservando las responsabilidades y contratos de SC-03 a SC-06. Esa evolución requerirá definir seguimiento de trabajos, consulta de estado y cancelación, además de revisar bloqueos. No está implementada en esta fase.

## 38. Estado de cierre

```text
[x] FastAPI local y ocho endpoints mínimos.
[x] Dependencias compartidas cargadas por lifespan.
[x] Reconocedores ASL/LSM protegidos.
[x] Capturas deliberadas JPEG/PNG con validación.
[x] Conservación de confianza y Top-k.
[x] Comandos space, backspace y clear.
[x] Sesiones independientes en memoria bajo SC-07.
[x] Petición de traducción abierta hasta el resultado final.
[x] Operaciones bloqueantes fuera del event loop.
[x] Contratos existentes reutilizados.
[x] Plan visual sin ventanas OpenCV.
[x] URLs HTTP y soporte Range probado.
[x] Entrada conservada ante fallo SC-04.
[x] Resultado textual conservado ante fallo visual.
[x] Manejo de unresolved total y parcial.
[x] Suite completa: 103 pruebas aprobadas.
[x] Inicialización con modelos y recursos reales.
[x] Configuración y ejecución documentadas.
[x] Componentes y scripts existentes conservados.
[ ] Revisión del resultado por el responsable del proyecto.
[ ] Integración y validación con SC-01 en React + Vite.
[ ] Validación de cámara y reproducción web.
```

La implementación se entregó sin realizar commit. SC-02 constituye la primera capa HTTP del prototipo, validada automáticamente en el entorno local y preparada para integrar posteriormente SC-01.

---

**Fin del documento de fase SC-02 — Control del flujo de traducción mediante FastAPI.**
