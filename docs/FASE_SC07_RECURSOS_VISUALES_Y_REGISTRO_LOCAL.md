# Fase SC-07 — Recursos visuales y registro local

Continuación del documento:

`FASE_SC04_CORRECCION_TRADUCCION_E_INTEGRACION.md`

El documento anterior finalizó estableciendo como siguiente bloque la transformación de los conceptos canónicos producidos por SC-04 en recursos visuales correspondientes a la lengua de señas destino.

El contrato disponible al inicio de esta fase fue:

```text
TranslationResult
+
target_language
+
concepts
```

Ejemplo:

```text
target_language:
ASL

concepts:
FIRST_PERSON_SINGULAR
WANT
WATER
```

A partir de este punto se trabajó en la construcción y organización del repositorio visual requerido por SC-07 y que posteriormente será consumido por SC-05.

---

## 37. Objetivo de la fase

El objetivo principal de esta fase fue construir un repositorio visual local completo para los 71 conceptos canónicos definidos en:

```text
data/vocabulary/sign2sign_vocabulary.json
```

para ambos idiomas de salida:

```text
ASL
LSM
```

El repositorio debía cumplir las siguientes condiciones:

- existir un recurso visual local para cada `concept_id`;
- conservar separación entre ASL y LSM;
- mantener trazabilidad de la fuente de cada recurso;
- utilizar nombres compatibles directamente con los `concept_id`;
- permitir posteriormente que SC-05 resuelva conceptos a videos;
- permitir distinguir recursos locales y recursos online;
- detectar archivos faltantes o nombres no válidos;
- evitar registros duplicados.

---

## 38. Relación entre SC-04, SC-05, SC-06 y SC-07

La separación de responsabilidades adoptada es:

```text
SC-04
Corrección y traducción
        ↓
TranslationResult
+
target_language
+
concepts
        ↓
SC-05
Mapeo de recursos visuales
        ↓
recursos resueltos
        ↓
SC-06
Generación de salida visual
        ↓
secuencia ordenada de videos
```

SC-07 proporciona la infraestructura de recursos utilizada por SC-05:

```text
SC-07
        ↓
repositorio local
+
visual_resources.json
+
resource_sources.csv
```

Por lo tanto:

```text
SC-04 decide QUÉ conceptos deben mostrarse.

SC-05 decide QUÉ recursos corresponden
a esos conceptos.

SC-06 decide CÓMO reproducir la secuencia.

SC-07 mantiene DÓNDE están disponibles
los recursos y su información asociada.
```

---

## 39. Estructura física del repositorio

Se estableció la siguiente estructura:

```text
resources/
├── asl/
│   ├── HELLO.mp4
│   ├── GOODBYE.mp4
│   ├── WATER.mp4
│   └── ...
│
└── lsm/
    ├── HELLO.mp4
    ├── GOODBYE.mp4
    ├── WATER.mp4
    └── ...
```

La convención adoptada es:

```text
resources/{language}/{CONCEPT_ID}.mp4
```

Ejemplos:

```text
resources/asl/WATER.mp4
resources/lsm/WATER.mp4

resources/asl/FIRST_PERSON_SINGULAR.mp4
resources/lsm/FIRST_PERSON_SINGULAR.mp4
```

Esto permite resolver directamente un recurso a partir de:

```text
concept_id
+
language
```

sin depender del nombre textual natural en español o inglés.

---

## 40. Catálogo funcional de recursos

El catálogo funcional se almacena en:

```text
data/resources/visual_resources.json
```

Cada recurso contiene actualmente:

```json
{
  "concept_id": "WATER",
  "language": "LSM",
  "delivery": "local",
  "location": "resources/lsm/WATER.mp4",
  "type": "video",
  "available": true
}
```

Los idiomas soportados son:

```text
ASL
LSM
```

y se definieron dos modalidades posibles:

```text
local
online
```

Durante esta fase se cerró completamente la modalidad:

```text
local
```

La modalidad:

```text
online
```

queda preparada conceptualmente y será integrada posteriormente.

---

## 41. Archivo de trazabilidad

La procedencia de los recursos se mantiene separada del catálogo funcional mediante:

```text
data/resources/resource_sources.csv
```

Campos:

```text
concept_id
language
delivery
source_name
source_url
notes
```

Esta separación permite que:

```text
visual_resources.json
```

sea utilizado por el sistema durante ejecución, mientras que:

```text
resource_sources.csv
```

mantiene información documental y de trazabilidad.

---

## 42. Obtención de recursos ASL

Para ASL se utilizó como fuente principal:

```text
ASL Signbank
```

Los recursos disponibles fueron descargados y renombrados utilizando directamente el `concept_id` correspondiente.

Ejemplo:

```text
glosa Signbank
        ↓
concept_id Sign2Sign
        ↓
resources/asl/{CONCEPT_ID}.mp4
```

Después del barrido inicial se obtuvieron:

```text
63 / 71
```

recursos ASL locales.

Los ocho conceptos restantes fueron:

```text
HE
SHE
YOU_ALL
SUBWAY
HOW_ARE_YOU
NICE_TO_MEET_YOU
WHAT_IS_YOUR_NAME
HOW_DO_YOU_FEEL
```

Para estos recursos se decidió realizar grabaciones propias basadas en material externo de referencia.

---

## 43. Casos especiales de ASL

### 43.1 HE y SHE

ASL utiliza una realización deíctica de tercera persona que no codifica mediante esta forma una distinción de género equivalente a:

```text
HE
SHE
```

Por esta razón se realizó una única ejecución visual de tercera persona y se utilizaron dos archivos:

```text
HE.mp4
SHE.mp4
```

con la misma realización.

Esta decisión aplica únicamente a ASL.

En LSM:

```text
HE
SHE
```

corresponden a realizaciones distintas y cuentan con videos diferentes.

### 43.2 YOU_ALL

`YOU_ALL` se mantuvo separado de:

```text
SECOND_PERSON
THEY
```

La realización grabada corresponde a un señalamiento/barrido dirigido hacia un conjunto de interlocutores.

No se reutilizó directamente el recurso correspondiente a `THEY`.

### 43.3 SUBWAY

ASL Signbank contiene una entrada asociada a `SUBWAY`, pero durante la consulta realizada el recurso audiovisual no se encontraba disponible.

Por esta razón:

```text
SUBWAY
```

fue incorporado mediante grabación propia basada en una referencia externa.

---

## 44. Tratamiento de expresiones ASL

El vocabulario canónico contiene seis conceptos de tipo:

```text
expression
```

No se asumió que una expresión semántica correspondiera necesariamente a una única seña indivisible.

Sin embargo, después de revisar los recursos disponibles se decidió utilizar recursos directos cuando fuera posible.

Ya se contaba con recursos directos de ASL Signbank para:

```text
YOU_ARE_WELCOME
EXCUSE_ME
```

Las expresiones restantes fueron grabadas como realizaciones completas:

```text
HOW_ARE_YOU
NICE_TO_MEET_YOU
WHAT_IS_YOUR_NAME
HOW_DO_YOU_FEEL
```

Esto evita construir artificialmente las expresiones mediante cortes entre videos independientes y permite conservar mejor:

- continuidad del movimiento;
- dirección;
- ritmo;
- transiciones;
- componentes no manuales;
- expresión facial.

El vocabulario canónico no fue modificado por estas decisiones.

---

## 45. Recursos LSM

Para LSM no se encontró un repositorio equivalente a ASL Signbank que permitiera descargar de manera uniforme todos los videos necesarios.

Por esta razón se decidió producir un repositorio local propio para el prototipo.

Se grabaron recursos correspondientes a:

```text
71 / 71
```

conceptos canónicos.

Cada `concept_id` cuenta con un archivo independiente:

```text
resources/lsm/{CONCEPT_ID}.mp4
```

Las realizaciones fueron elaboradas utilizando material de referencia de LSM.

Debe conservarse como limitación metodológica que:

```text
las grabaciones propias no han sido
validadas por un especialista certificado en LSM.
```

Esta limitación es una de las razones principales por las que se planteó una modalidad de recursos externos online.

---

## 46. Modalidad online propuesta

La modalidad online no sustituye necesariamente al repositorio local.

Todos los conceptos cuentan con un recurso local.

El objetivo de la modalidad online es proporcionar una referencia externa adicional cuando el recurso local fue producido por el equipo y no proviene directamente de una fuente audiovisual externa validada.

La política adoptada es:

```text
ASL descargado de ASL Signbank
→ local
→ no requiere necesariamente alternativa online

ASL grabado por el equipo
→ local
→ candidato a recurso online de referencia

LSM grabado por el equipo
→ local
→ candidato a recurso online de referencia
```

Fuentes consideradas para una fase posterior:

```text
ASL:
- ASL University / Lifeprint
- Gallaudet / recursos institucionales equivalentes

LSM:
- WikiSigns
- otras fuentes externas confiables por validar
```

WikiSigns utiliza en varios casos videos alojados en YouTube.

Para estos recursos se plantea utilizar reproducción mediante el mecanismo online autorizado correspondiente, en lugar de descargar arbitrariamente los videos.

La modalidad online todavía NO se considera implementada.

---

## 47. Formato de las grabaciones propias

Para las grabaciones propias se utilizó aproximadamente:

```text
Contenedor:
MP4

Resolución:
1280 x 720

Relación:
16:9

Frame rate:
aprox. 30 FPS

Video:
H.264

Pixel format:
yuv420p
```

Se mantuvo una estructura visual consistente:

```text
posición neutral
↓
entrada de las manos al espacio de señado
↓
realización completa
↓
retorno a posición neutral
```

Las manos no necesitan permanecer visibles desde el primer frame, siguiendo un comportamiento similar al observado en recursos de ASL Signbank.

---

## 48. Eliminación del audio

Las grabaciones realizadas inicialmente contenían una pista de audio.

Debido a que:

```text
la salida de Sign2Sign es exclusivamente visual
```

se decidió eliminar el audio de los recursos propios.

Se utilizó FFmpeg sin volver a codificar el video:

```text
-map 0:v:0
-c:v copy
-an
```

De esta manera:

```text
video
→ se conserva sin pérdida por recodificación

audio
→ se elimina
```

No se realizó compresión adicional de los videos en esta fase.

---

## 49. Script de registro individual

Se dispone de:

```text
scripts/register_visual_resource.py
```

Responsabilidades principales:

- cargar el vocabulario canónico;
- validar `concept_id`;
- validar idioma;
- validar modalidad `local` u `online`;
- validar rutas locales;
- validar URLs online;
- evitar registros duplicados;
- actualizar `visual_resources.json`;
- actualizar `resource_sources.csv`;
- calcular progreso.

Para recursos locales construye rutas con el formato:

```text
resources/{language}/{CONCEPT_ID}.mp4
```

---

## 50. Registro masivo

Debido a que era necesario incorporar:

```text
8 recursos ASL propios
+
71 recursos LSM
```

se agregó:

```text
scripts/register_visual_resources_bulk.py
```

El script realiza antes del registro:

```text
validación del vocabulario
↓
validación de cantidad de conceptos
↓
auditoría de archivos físicos
↓
detección de archivos faltantes
↓
detección de nombres no canónicos
↓
detección de registros existentes
↓
detección de inconsistencias JSON / CSV
```

Antes de realizar cambios crea un respaldo de:

```text
visual_resources.json
resource_sources.csv
```

en:

```text
data/resources/backups/
```

Los registros ASL provenientes previamente de ASL Signbank no son reemplazados.

El script solamente agrega recursos faltantes.

---

## 51. Incidencia detectada durante la auditoría

Durante la primera ejecución de la validación masiva se detectó:

```text
Faltante:
SECOND_PERSON

Extra:
SECOND_PERSON_SINGLAR
```

El archivo LSM había sido nombrado incorrectamente como:

```text
SECOND_PERSON_SINGLAR.mp4
```

Se corrigió a:

```text
SECOND_PERSON.mp4
```

La segunda ejecución confirmó:

```text
ASL
Esperados:   71
Encontrados: 71
Faltantes:   ninguno
Extras:      ninguno

LSM
Esperados:   71
Encontrados: 71
Faltantes:   ninguno
Extras:      ninguno
```

Este caso confirmó la utilidad de validar los nombres físicamente contra el vocabulario canónico antes del registro.

---

## 52. Resultado final del registro local

Después de ejecutar el registro masivo se obtuvo:

```text
Recursos locales:

ASL:
71 / 71

LSM:
71 / 71

Pares locales:
71 / 71
```

Resultado:

```text
OK: repositorio visual local completo
(71/71 por idioma)
```

Por lo tanto actualmente existen:

```text
71 recursos ASL locales
+
71 recursos LSM locales
=
142 asociaciones concepto/idioma locales
```

Todos los conceptos del vocabulario cerrado disponen de un recurso local para ambos idiomas.

---

## 53. Estado actual de SC-07

El hito correspondiente a:

```text
SC-07 + estructura de recursos
```

se considera completado para el repositorio visual local.

Estado:

```text
[x] estructura resources/asl
[x] estructura resources/lsm
[x] 71/71 recursos ASL
[x] 71/71 recursos LSM
[x] nombres alineados con concept_id
[x] catálogo visual_resources.json
[x] trazabilidad resource_sources.csv
[x] registro individual
[x] registro masivo
[x] detección de duplicados
[x] validación de archivos físicos
[x] respaldo previo a registro masivo
[x] recursos propios sin audio
[x] repositorio local completo
[ ] recursos online de referencia
[ ] integración del manejo temporal con el orquestador final
```

La ausencia de recursos online no bloquea el flujo principal, ya que todos los conceptos cuentan actualmente con recurso local.

---

## 54. Contrato disponible para SC-05

A partir de este punto SC-05 puede recibir:

```text
TranslationResult
+
target_language
+
concepts
```

Ejemplo:

```text
target_language:
LSM

concepts:
FIRST_PERSON_SINGULAR
WANT
WATER
```

y resolver:

```text
FIRST_PERSON_SINGULAR
→ resources/lsm/FIRST_PERSON_SINGULAR.mp4

WANT
→ resources/lsm/WANT.mp4

WATER
→ resources/lsm/WATER.mp4
```

Resultado esperado:

```text
[
    FIRST_PERSON_SINGULAR.mp4,
    WANT.mp4,
    WATER.mp4
]
```

conservando el orden semántico recibido desde SC-04.

---

## 55. Siguiente bloque

El siguiente trabajo corresponde a:

```text
SC-05
Mapeo de recursos visuales
```

Se propone incorporar:

```text
src/
└── visual_mapping/
    ├── __init__.py
    ├── contracts.py
    └── resource_mapper.py
```

y:

```text
tests/
└── test_resource_mapper.py
```

SC-05 deberá resolver inicialmente recursos con:

```text
delivery = local
```

y validar al menos:

```text
- concept_id existente;
- idioma destino válido;
- recurso disponible;
- conservación del orden;
- resolución de varios conceptos;
- ASL y LSM independientes;
- error controlado ante recurso inexistente.
```

Una vez validado SC-05, el siguiente paso será:

```text
SC-06
Generación de salida visual
```

para reproducir de manera ordenada los recursos obtenidos.

Posteriormente se integrará:

```text
SC-03
↓
SC-04
↓
SC-05
↓
SC-06
```

antes de incorporar el orquestador general mediante FastAPI y la interfaz final en React.

---

## 56. Resumen ejecutivo de la fase

SC-07 cuenta actualmente con un repositorio visual local completo para las dos lenguas soportadas por Sign2Sign.

Se dispone de:

```text
ASL: 71/71
LSM: 71/71
Pares: 71/71
```

Los recursos están organizados por `concept_id`, registrados en un catálogo funcional y acompañados por información de trazabilidad.

ASL utiliza principalmente recursos procedentes de ASL Signbank, complementados con ocho conceptos producidos mediante grabaciones propias.

LSM dispone de un repositorio completo de grabaciones propias, cuya falta de validación por un especialista se conserva como limitación del prototipo y motiva la futura incorporación de recursos online externos de referencia.

Con el repositorio local completo, el proyecto puede avanzar a SC-05 sin depender todavía de la modalidad online.

---

**Fin del estado actual de SC-07.**
