\# Fase SC-04 — Corrección, traducción textual e integración con SC-03



\## 1. Objetivo de la fase



Esta fase tuvo como objetivo desarrollar el subsistema \*\*SC-04 Corrección y traducción textual\*\*, encargado de recibir la salida estructurada producida por SC-03, corregir posibles errores provenientes del reconocimiento de señas, traducir el contenido entre los idiomas puente español e inglés y transformar el significado obtenido en una secuencia de conceptos canónicos pertenecientes a un vocabulario cerrado de Sign2Sign.



El subsistema recibe información adicional a la cadena Top-1 reconocida, incluyendo las alternativas Top-k y sus probabilidades. Esta información permite utilizar el contexto lingüístico y el vocabulario permitido para resolver determinadas ambigüedades sin depender exclusivamente de la predicción principal del clasificador.



El alcance de esta fase incluye:



\- definición y normalización del vocabulario cerrado;

\- definición del contrato de entrada y salida de SC-04;

\- construcción del prompt restringido;

\- conexión con un modelo de lenguaje;

\- validación de las respuestas generadas;

\- manejo de contenido fuera del vocabulario;

\- pruebas unitarias;

\- pruebas funcionales con el LLM real;

\- integración funcional entre SC-03 y SC-04.



La asociación de los conceptos resultantes con videos o recursos visuales no forma parte de esta fase y corresponde al subsistema posterior.



\---



\## 2. Entrada recibida desde SC-03



SC-03 produce una instancia `BuiltText` que contiene:



\- lengua de señas de origen;

\- texto intermedio construido;

\- secuencia ordenada de tokens;

\- letra Top-1 por captura;

\- alternativas Top-k;

\- probabilidades asociadas;

\- separadores entre palabras.



Ejemplo conceptual:



```json

{

&#x20; "source\_language": "LSM",

&#x20; "text": "HCLA",

&#x20; "tokens": \[

&#x20;   {

&#x20;     "kind": "letter",

&#x20;     "value": "H",

&#x20;     "top\_k": \[

&#x20;       {"label": "H", "probability": 0.98},

&#x20;       {"label": "G", "probability": 0.01},

&#x20;       {"label": "N", "probability": 0.01}

&#x20;     ]

&#x20;   }

&#x20; ]

}

```



SC-04 utiliza tanto el texto completo como la evidencia probabilística asociada a cada letra.



\---



\## 3. Decisión sobre el vocabulario



\### 3.1 Vocabulario canónico



Se decidió no representar el vocabulario únicamente como una correspondencia directa entre palabras en español e inglés.



En su lugar, se definió un \*\*vocabulario canónico basado en conceptos\*\*, donde cada elemento posee un identificador independiente de la lengua utilizada.



Ejemplo:



```json

{

&#x20; "concept\_id": "WANT",

&#x20; "type": "atomic",

&#x20; "spanish": {

&#x20;   "canonical": "QUERER",

&#x20;   "aliases": \["QUIERES"]

&#x20; },

&#x20; "english": {

&#x20;   "canonical": "WANT",

&#x20;   "aliases": \["YOU WANT"]

&#x20; }

}

```



De esta manera, diferentes formas flexionadas pueden mapearse hacia un mismo concepto sin requerir un recurso diferente para cada conjugación.



Ejemplo:



```text

QUIERO

QUIERES

QUEREMOS

&#x20;   ↓

WANT

```



El significado gramatical adicional, como la persona expresada por la conjugación, puede recuperarse mediante otros conceptos disponibles en el vocabulario.



\---



\## 4. Vocabulario Sign2Sign V2



Se normalizó el vocabulario original del proyecto para producir la versión \*\*V2 Canónica\*\*.



La versión utilizada contiene:



```text

71 conceptos totales

├── 65 atomic

└── 6 expression

```



El archivo utilizado por el sistema es:



```text

data/vocabulary/sign2sign\_vocabulary.json

```



\### 4.1 Conceptos atomic



Los conceptos `atomic` representan unidades semánticas que pueden combinarse con otros conceptos.



Ejemplos:



```text

HELLO

WATER

WANT

NOT

UNDERSTAND

HOSPITAL

FIRST\_PERSON\_SINGULAR

SECOND\_PERSON

```



\### 4.2 Conceptos expression



Los conceptos `expression` representan expresiones cuyo significado completo puede tratarse como una unidad.



Se definieron las siguientes expresiones:



```text

HOW\_ARE\_YOU

YOU\_ARE\_WELCOME

EXCUSE\_ME

NICE\_TO\_MEET\_YOU

WHAT\_IS\_YOUR\_NAME

HOW\_DO\_YOU\_FEEL

```



Ejemplo:



```text

CÓMO ESTÁS

&#x20;   ↓

HOW\_ARE\_YOU

```



En este caso no es necesario devolver adicionalmente `HOW` y `SECOND\_PERSON`, ya que el concepto de expresión conserva por sí mismo el significado requerido.



\---



\## 5. Normalizaciones lingüísticas realizadas



Durante la construcción del vocabulario V2 se tomaron varias decisiones para reducir ambigüedades.



\### 5.1 Negación composicional



Las formas negativas dejaron de tratarse como conceptos independientes.



Ejemplo:



```text

NO\_QUERER

```



se representa como:



```text

NOT + WANT

```



De manera equivalente:



```text

NO\_PODER

→ NOT + CAN



NO\_SABER

→ NOT + KNOW\_INFORMATION



NO\_ENTENDER

→ NOT + UNDERSTAND



NO\_GUSTAR

→ NOT + LIKE

```



Esto permite aplicar la negación a otros conceptos sin definir una entrada adicional para cada combinación.



\---



\### 5.2 SABER y CONOCER



Se separaron los significados relacionados con conocimiento.



```text

SABER

→ KNOW\_INFORMATION

```



representa conocimiento de información o hechos.



```text

CONOCER

→ KNOW\_PERSON

```



representa conocer o estar familiarizado con una persona.



También se definió:



```text

MEET

```



para representar conocer a una persona por primera vez.



\---



\### 5.3 TENER y HAY



Se separaron los conceptos de posesión y existencia.



```text

TENER

→ HAVE

```



mientras que:



```text

HAY

→ EXIST

```



Esto evita representar como equivalentes situaciones semánticamente distintas.



\---



\### 5.4 TERMINAR y FIN



Se separaron:



```text

TERMINAR

→ FINISH

```



y:



```text

FIN

→ END

```



para distinguir la acción de finalizar de la noción de conclusión o término.



\---



\### 5.5 Conjugaciones



Las conjugaciones no se almacenan como conceptos separados.



Por ejemplo:



```text

QUIERO AGUA

```



puede interpretarse como:



```text

FIRST\_PERSON\_SINGULAR

WANT

WATER

```



mientras que:



```text

QUIERES AGUA

```



puede producir:



```text

SECOND\_PERSON

WANT

WATER

```



\---



\## 6. Componente Vocabulary



Se creó:



```text

src/text\_processing/

├── \_\_init\_\_.py

└── vocabulary.py

```



La clase `Vocabulary` es responsable de:



\- cargar el archivo JSON;

\- validar la estructura general;

\- comprobar tipos de conceptos;

\- detectar identificadores duplicados;

\- validar campos canónicos;

\- obtener conceptos por identificador;

\- filtrar conceptos por tipo;

\- generar un catálogo compacto para el prompt;

\- comprobar si un `concept\_id` pertenece al vocabulario permitido.



Entre los métodos disponibles se encuentran:



```text

has\_concept()

get\_concept()

get\_allowed\_ids()

get\_by\_type()

get\_prompt\_catalog()

```



La representación utilizada para el prompt excluye información de trazabilidad que no es necesaria durante la inferencia.



\---



\## 7. Pruebas unitarias del vocabulario



Se creó:



```text

tests/test\_vocabulary.py

```



Las pruebas verifican:



\- número total de conceptos;

\- número de conceptos `atomic`;

\- número de conceptos `expression`;

\- existencia de conceptos relevantes;

\- eliminación de formas negativas como identificadores independientes;

\- generación del catálogo utilizado por el prompt.



Resultado inicial:



```text

Ran 6 tests



OK

```



Se verificó:



```text

Total:       71

Atomic:      65

Expressions: 6

```



\---



\## 8. Contratos de SC-04



Se creó:



```text

src/text\_processing/contracts.py

```



Se definieron dos estructuras principales.



\### 8.1 CorrectionRequest



`CorrectionRequest` representa la entrada formal de SC-04.



Contiene:



```text

source\_language

target\_language

recognized\_text

tokens

```



Se genera directamente desde `BuiltText`:



```text

BuiltText

&#x20;   ↓

CorrectionRequest.from\_built\_text()

```



También determina automáticamente los idiomas puente.



```text

LSM → ASL

Spanish → English



ASL → LSM

English → Spanish

```



El sistema impide utilizar la misma lengua como origen y destino.



\---



\### 8.2 TranslationResult



`TranslationResult` representa la salida validada de SC-04.



Contiene:



```text

source\_language

target\_language

recognized\_text

corrected\_text

translated\_text

concepts

unresolved

```



Ejemplo:



```json

{

&#x20; "source\_language": "LSM",

&#x20; "target\_language": "ASL",

&#x20; "recognized\_text": "HCLA",

&#x20; "corrected\_text": "HOLA",

&#x20; "translated\_text": "HELLO",

&#x20; "concepts": \[

&#x20;   "HELLO"

&#x20; ],

&#x20; "unresolved": \[]

}

```



\---



\## 9. Construcción del prompt



Se creó:



```text

src/text\_processing/prompt\_builder.py

```



`PromptBuilder` genera la entrada utilizada por el LLM combinando:



\- lengua de origen;

\- lengua de destino;

\- idioma puente de origen;

\- idioma puente de destino;

\- texto reconocido;

\- alternativas Top-k;

\- probabilidades;

\- vocabulario completo permitido;

\- reglas lingüísticas;

\- contrato obligatorio de respuesta.



\---



\## 10. Uso de Top-k durante la corrección



Las probabilidades de SC-03 son tratadas como evidencia, no como umbrales de aceptación.



Ejemplo:



```text

R = 54 %

U = 44 %

V = 2 %

```



Si SC-03 produce:



```text

AGRA

```



SC-04 puede utilizar:



\- estructura de la palabra;

\- vocabulario permitido;

\- alternativa `U`;

\- contexto lingüístico;



para producir:



```text

AGUA

```



La alternativa Top-1 no es considerada una decisión irreversible.



\---



\## 11. Corrección fuera del Top-k



También se comprobó que una interpretación puede resultar válida aunque la letra necesaria no se encuentre dentro del Top-3.



Ejemplo evaluado:



```text

HCLA

```



donde para la segunda posición se suministró un Top-3 equivalente a:



```text

C

D

X

```



sin incluir la letra `O`.



El modelo produjo:



```text

HCLA

↓

HOLA

↓

HELLO

↓

HELLO

```



Este comportamiento confirma que Top-k funciona como evidencia adicional, mientras que el vocabulario cerrado y el contexto pueden apoyar una corrección cuando existe suficiente evidencia lingüística.



\---



\## 12. Reglas semánticas del prompt



El prompt establece, entre otras, las siguientes reglas:



\- sólo pueden utilizarse `concept\_id` existentes;

\- no pueden inventarse conceptos;

\- las conjugaciones deben normalizarse;

\- la negación es composicional;

\- deben distinguirse `KNOW\_INFORMATION`, `KNOW\_PERSON` y `MEET`;

\- deben distinguirse `HAVE` y `EXIST`;

\- deben distinguirse `FINISH` y `END`;

\- las expresiones pueden utilizarse como unidades completas;

\- Top-k debe utilizarse como evidencia;

\- una entrada no debe corregirse forzosamente;

\- los elementos fuera del vocabulario deben incluirse en `unresolved`;

\- el significado representable no debe perderse entre el texto traducido y la lista de conceptos.



\---



\## 13. Recuperación de información gramatical implícita



Durante las primeras pruebas se observó que el modelo podía traducir correctamente una frase pero omitir información representable dentro de la lista de conceptos.



Ejemplo inicial:



```text

NO ENTIENDO

```



producía:



```text

I DO NOT UNDERSTAND

```



pero:



```text

NOT

UNDERSTAND

```



La traducción incluía el sujeto en primera persona, pero éste no se conservaba en la representación canónica.



Se modificó el prompt para establecer que la lista `concepts` debe conservar toda la información semántica representable mediante el vocabulario.



Después de la modificación se obtuvo:



```text

NO ENTIENDO

↓

I DO NOT UNDERSTAND

↓

FIRST\_PERSON\_SINGULAR

NOT

UNDERSTAND

```



También se comprobaron:



```text

QUIERO AGUA

↓

FIRST\_PERSON\_SINGULAR

WANT

WATER

```



y:



```text

QUIERES AGUA

↓

SECOND\_PERSON

WANT

WATER

```



\---



\## 14. Tratamiento de expresiones completas



La recuperación de participantes implícitos no debe producir redundancia cuando una expresión ya representa todo el significado.



Ejemplo:



```text

CÓMO ESTÁS

```



produce:



```text

HOW\_ARE\_YOU

```



y no:



```text

HOW\_ARE\_YOU

HOW

SECOND\_PERSON

```



La regla adoptada es utilizar la representación mínima que preserve completamente el significado disponible.



\---



\## 15. Formato de respuesta del LLM



Se definió una respuesta JSON obligatoria:



```json

{

&#x20; "corrected\_text": "string",

&#x20; "translated\_text": "string",

&#x20; "concepts": \[

&#x20;   "CONCEPT\_ID"

&#x20; ],

&#x20; "unresolved": \[

&#x20;   "string"

&#x20; ]

}

```



No se admiten explicaciones adicionales ni conceptos ajenos al vocabulario.



\---



\## 16. ResponseValidator



Se creó:



```text

src/text\_processing/response\_validator.py

```



El componente comprueba:



\- que la respuesta sea JSON válido;

\- que la raíz sea un objeto;

\- que existan todos los campos requeridos;

\- que los textos tengan el tipo correcto;

\- que `concepts` sea una lista;

\- que cada concepto exista en el vocabulario;

\- que `unresolved` sea una lista de cadenas;

\- que la respuesta contenga al menos conceptos o contenido sin resolver.



Una respuesta no pasa directamente desde el LLM hacia el resto del sistema.



El flujo es:



```text

LLM

&#x20;↓

JSON

&#x20;↓

ResponseValidator

&#x20;↓

TranslationResult

```



\---



\## 17. Manejo de contenido fuera del vocabulario



Se definieron tres posibles situaciones.



\### 17.1 Entrada completamente resoluble



Ejemplo:



```text

AGUA

```



Resultado:



```json

{

&#x20; "corrected\_text": "AGUA",

&#x20; "translated\_text": "WATER",

&#x20; "concepts": \["WATER"],

&#x20; "unresolved": \[]

}

```



\---



\### 17.2 Entrada parcialmente resoluble



Ejemplo:



```text

HCLA CAFE

```



Resultado obtenido:



```json

{

&#x20; "corrected\_text": "HOLA CAFÉ",

&#x20; "translated\_text": "HELLO CAFE",

&#x20; "concepts": \[

&#x20;   "HELLO"

&#x20; ],

&#x20; "unresolved": \[

&#x20;   "CAFÉ"

&#x20; ]

}

```



El contenido válido se conserva mientras la parte que no puede representarse se reporta explícitamente.



\---



\### 17.3 Entrada completamente irresoluble



Durante una prueba se utilizó:



```text

ZXQW

```



La primera versión del `ResponseValidator` rechazó la respuesta porque `corrected\_text` estaba vacío.



Este comportamiento permitió identificar que una entrada completamente desconocida constituye un caso válido:



```json

{

&#x20; "corrected\_text": "",

&#x20; "translated\_text": "",

&#x20; "concepts": \[],

&#x20; "unresolved": \[

&#x20;   "ZXQW"

&#x20; ]

}

```



Se modificó el validador para permitir textos vacíos únicamente cuando:



```text

concepts = \[]

y

unresolved != \[]

```



Por el contrario, esta respuesta continúa siendo inválida:



```json

{

&#x20; "corrected\_text": "",

&#x20; "translated\_text": "",

&#x20; "concepts": \[],

&#x20; "unresolved": \[]

}

```



La política adoptada es:



```text

si puede resolverse

→ corregir y traducir



si sólo una parte puede resolverse

→ traducir la parte válida + unresolved



si nada puede resolverse

→ no inventar + unresolved

```



\---



\## 18. Pruebas unitarias del procesamiento textual



Se incorporaron:



```text

tests/

├── test\_prompt\_builder.py

└── test\_response\_validator.py

```



\### 18.1 PromptBuilder



Se implementaron inicialmente 10 pruebas para verificar:



\- lenguas de origen y destino;

\- idiomas puente;

\- rechazo de origen y destino iguales;

\- inclusión del texto reconocido;

\- inclusión de Top-k;

\- inclusión del vocabulario;

\- presencia de los 71 identificadores;

\- esquema de salida;

\- exclusión de trazabilidad innecesaria;

\- serialización de `TranslationResult`.



Resultado:



```text

Ran 10 tests



OK

```



\---



\### 18.2 ResponseValidator



Se verificaron, entre otros:



\- respuesta válida como diccionario;

\- respuesta válida como JSON;

\- rechazo de conceptos inexistentes;

\- rechazo de campos faltantes;

\- rechazo de JSON inválido;

\- validación de campos textuales;

\- validación del tipo de `concepts`;

\- contenido parcialmente unresolved;

\- contenido completamente unresolved;

\- expresiones canónicas;

\- normalización de identificadores;

\- rechazo de una respuesta sin conceptos ni unresolved.



Después de incorporar el caso completamente irresoluble, la suite total acumulada alcanzó:



```text

52 tests

```



sin introducir regresiones en las pruebas anteriores.



\---



\## 19. Integración simulada de SC-04



Antes de realizar llamadas externas se creó:



```text

scripts/test\_sc04\_pipeline.py

```



La prueba ejecuta:



```text

BuiltText

&#x20;↓

CorrectionRequest

&#x20;↓

PromptBuilder

&#x20;↓

respuesta simulada

&#x20;↓

ResponseValidator

&#x20;↓

TranslationResult

```



Se utilizó el caso:



```text

AGRA

```



con:



```text

R = 54 %

U = 44 %

V = 2 %

```



y una respuesta simulada:



```text

AGUA

WATER

WATER

```



La prueba finalizó correctamente.



Esto permitió comprobar la integración de todos los componentes locales antes de introducir una dependencia externa.



\---



\## 20. Selección del modelo de lenguaje



Para las pruebas reales de SC-04 se seleccionó:



```text

gemini-3.5-flash-lite

```



La elección se realizó priorizando:



\- baja latencia;

\- costo reducido;

\- capacidad suficiente para una tarea restringida;

\- disponibilidad de salida estructurada;

\- capacidad de procesar el vocabulario completo en cada solicitud.



El sistema no utiliza al LLM como traductor completamente libre.



El modelo recibe:



```text

texto reconocido

\+

Top-k

\+

probabilidades

\+

71 conceptos permitidos

\+

reglas de normalización

\+

esquema JSON

```



\---



\## 21. Configuración de la API



Se configuró un proyecto de Google AI Studio / Gemini API asociado al proyecto Sign2Sign.



La credencial se almacena únicamente de forma local mediante:



```text

.env

```



utilizando:



```text

GEMINI\_API\_KEY

```



El archivo `.env` se encuentra excluido del control de versiones mediante `.gitignore`.



También se conserva en dicho archivo la configuración externa utilizada por otras partes del proyecto, como la ruta de datos.



No se incluyen secretos dentro del código fuente ni en el repositorio remoto.



\---



\## 22. Prueba de conexión con Gemini



Se creó:



```text

scripts/test\_gemini\_connection.py

```



La prueba realizó una llamada mínima al modelo y obtuvo:



```text

Respuesta: OK

```



confirmando:



\- autenticación correcta;

\- disponibilidad del modelo;

\- conectividad;

\- configuración correcta del SDK.



\---



\## 23. GeminiClient



Se creó:



```text

src/text\_processing/gemini\_client.py

```



El componente implementa:



```text

CorrectionRequest

&#x20;↓

PromptBuilder

&#x20;↓

Gemini

&#x20;↓

respuesta estructurada

&#x20;↓

ResponseValidator

&#x20;↓

TranslationResult

```



También registra:



\- modelo utilizado;

\- respuesta cruda;

\- latencia de la llamada.



La estructura auxiliar `GeminiCallResult` conserva estos datos para evaluación.



\---



\## 24. Primera prueba real de SC-04



Se creó:



```text

scripts/test\_sc04\_gemini.py

```



Se utilizó nuevamente:



```text

AGRA

```



con la alternativa correcta `U` como Top-2.



Gemini produjo:



```json

{

&#x20; "source\_language": "LSM",

&#x20; "target\_language": "ASL",

&#x20; "recognized\_text": "AGRA",

&#x20; "corrected\_text": "AGUA",

&#x20; "translated\_text": "WATER",

&#x20; "concepts": \[

&#x20;   "WATER"

&#x20; ],

&#x20; "unresolved": \[]

}

```



La latencia registrada fue:



```text

3.076 s

```



La respuesta pasó correctamente por `ResponseValidator`.



\---



\## 25. Pruebas lingüísticas iniciales



Se creó una batería adicional para comprobar distintos comportamientos.



\### 25.1 Expresión predefinida



Entrada:



```text

COMO ESTAS

```



Resultado:



```text

CÓMO ESTÁS

HOW ARE YOU

HOW\_ARE\_YOU

```



Latencia observada:



```text

2.875 s

```



\---



\### 25.2 Negación composicional



Entrada:



```text

NO QUIERO AGUA

```



Resultado:



```text

NO QUIERO AGUA

I DON'T WANT WATER

NOT

WANT

WATER

```



Latencia observada:



```text

1.913 s

```



Posteriormente el prompt fue refinado para conservar también los participantes implícitos cuando éstos puedan representarse.



\---



\### 25.3 Contenido fuera del vocabulario



Entrada:



```text

QUIERO CAFE

```



Resultado:



```text

QUIERO CAFÉ

I WANT COFFEE

```



Conceptos:



```text

FIRST\_PERSON\_SINGULAR

WANT

```



Contenido sin resolver:



```text

CAFÉ

```



Latencia:



```text

1.425 s

```



El modelo no generó `COFFEE` como identificador canónico porque dicho concepto no existe en el vocabulario.



\---



\## 26. Pruebas de robustez



Se creó:



```text

scripts/test\_sc04\_gemini\_robustness.py

```



La batería incluye escenarios donde:



\- la letra correcta es Top-2;

\- la letra correcta no aparece en Top-3;

\- se utiliza una palabra larga;

\- se prueba la dirección inversa;

\- sólo parte de la entrada pertenece al vocabulario;

\- la entrada completa es arbitraria.



\### 26.1 Correcta presente en Top-2



```text

AGRA

→ AGUA

→ WATER

```



Resultado satisfactorio.



\---



\### 26.2 Correcta fuera de Top-3



Entrada:



```text

HCLA

```



La letra necesaria `O` no fue incluida dentro del Top-3 simulado.



Resultado:



```text

HOLA

HELLO

HELLO

```



La prueba confirmó que Top-k no constituye una frontera rígida.



\---



\### 26.3 Palabra larga con correcta fuera de Top-3



Entrada:



```text

HOSPITAK

```



Resultado:



```text

HOSPITAL

HOSPITAL

HOSPITAL

```



También resultó satisfactoria.



\---



\### 26.4 Dirección ASL → LSM



Entrada:



```text

HELLC

```



Resultado:



```text

HELLO

HOLA

HELLO

```



La prueba confirmó que el mismo flujo funciona en la dirección inversa.



\---



\### 26.5 Entrada parcialmente resoluble



Entrada:



```text

HCLA CAFE

```



Resultado:



```text

HOLA CAFÉ

HELLO CAFE

```



Conceptos:



```text

HELLO

```



Unresolved:



```text

CAFÉ

```



\---



\### 26.6 Entrada arbitraria



Entrada:



```text

ZXQW

```



Resultado final, después de ajustar el contrato:



```json

{

&#x20; "corrected\_text": "",

&#x20; "translated\_text": "",

&#x20; "concepts": \[],

&#x20; "unresolved": \[

&#x20;   "ZXQW"

&#x20; ]

}

```



No se forzó ningún concepto.



\---



\### 26.7 Resultado global de la batería



Después de corregir el tratamiento de entradas completamente irresolubles:



```text

Casos completados: 6/6

Latencia promedio: 1.791 s

Latencia total: 10.744 s

```



\---



\## 27. Integración real SC-03 → SC-04



Una vez verificados los componentes de forma aislada, se creó:



```text

scripts/run\_sc03\_sc04\_webcam.py

```



Este script integra directamente:



```text

Webcam

&#x20;↓

SignRecognizer

&#x20;↓

TextBuilder

&#x20;↓

BuiltText

&#x20;↓

CorrectionRequest

&#x20;↓

PromptBuilder

&#x20;↓

Gemini 3.5 Flash-Lite

&#x20;↓

ResponseValidator

&#x20;↓

TranslationResult

```



No se utilizan entradas simuladas.



La cadena reconocida por SC-03 se envía directamente hacia SC-04 al presionar:



```text

T

```



\---



\## 28. Controles del flujo integrado



Durante la captura:



| Tecla | Acción |

|---|---|

| `ENTER` | Capturar y reconocer una letra |

| `SPACE` | Separar palabras |

| `BACKSPACE` | Eliminar la última letra o espacio |

| `ESC` | Limpiar la cadena |

| `T` | Finalizar SC-03 y enviar el resultado a SC-04 |

| `Q` | Salir |



Después de obtener el resultado de SC-04:



| Tecla | Acción |

|---|---|

| `ESC` | Iniciar una nueva cadena |

| `Q` | Salir |



Si la llamada a SC-04 presenta un error, el contenido de `TextBuilder` no se elimina, permitiendo reintentar sin volver a capturar todas las señas.



\---



\## 29. Pruebas reales mediante webcam



\### 29.1 HOLA reconocido correctamente



SC-03 produjo:



```text

HOLA

```



SC-04 produjo:



```text

Reconocido: HOLA

Corregido: HOLA

Traducido: HELLO

Conceptos: HELLO

Unresolved: ninguno

```



Latencia observada:



```text

4.368 s

```



Esta fue una de las primeras llamadas del flujo integrado.



\---



\### 29.2 Error real de reconocimiento corregido



SC-03 produjo:



```text

HCLA

```



SC-04 produjo:



```text

Corregido: HOLA

Traducido: HELLO

Conceptos: HELLO

```



Latencia:



```text

1.832 s

```



Este caso resulta especialmente relevante porque el error provino del reconocimiento real mediante webcam y no de una entrada artificial.



\---



\### 29.3 AGUA



SC-03 produjo:



```text

AGUA

```



SC-04 produjo:



```text

AGUA

WATER

WATER

```



Latencia:



```text

1.734 s

```



\---



\### 29.4 NO ENTIENDO



En una primera ejecución se obtuvo:



```text

Reconocido: NO ENTIENDO

Corregido: NO ENTIENDO

Traducido: I DO NOT UNDERSTAND

Conceptos:

NOT

UNDERSTAND

```



La traducción era correcta, pero la representación conceptual omitía el sujeto implícito.



Después de refinar el prompt se obtuvo:



```text

Reconocido: NO ENTIENDO

Corregido: NO ENTIENDO

Traducido: I DO NOT UNDERSTAND

Conceptos:

FIRST\_PERSON\_SINGULAR

NOT

UNDERSTAND

```



Latencia observada:



```text

2.833 s

```



\---



\### 29.5 QUIERO AGUA



Resultado:



```text

Reconocido: QUIERO AGUA

Corregido: QUIERO AGUA

Traducido: I WANT WATER

Conceptos:

FIRST\_PERSON\_SINGULAR

WANT

WATER

```



Latencia:



```text

2.721 s

```



\---



\### 29.6 QUIERES AGUA



Resultado:



```text

Reconocido: QUIERES AGUA

Corregido: QUIERES AGUA

Traducido: DO YOU WANT WATER

Conceptos:

SECOND\_PERSON

WANT

WATER

```



Latencia:



```text

1.854 s

```



\---



\### 29.7 CÓMO ESTÁS



Resultado:



```text

Reconocido: COMO ESTAS

Corregido: CÓMO ESTÁS

Traducido: HOW ARE YOU

Conceptos:

HOW\_ARE\_YOU

```



Latencia:



```text

1.793 s

```



La expresión no fue descompuesta innecesariamente.



\---



\## 30. Observaciones sobre latencia



Durante las pruebas realizadas se observaron respuestas aproximadamente entre:



```text

1.4 s

y

4.4 s

```



La mayoría de las llamadas posteriores a la inicial se ubicaron aproximadamente alrededor de:



```text

1.7 – 2.8 s

```



Las primeras llamadas de algunas ejecuciones presentaron mayor latencia.



No se establece todavía una latencia promedio definitiva del subsistema, ya que será necesario ejecutar una evaluación con mayor número de casos.



\---



\## 31. Observaciones sobre costo



Durante las primeras pruebas reales realizadas con Gemini API se observó un consumo económico bajo.



Como referencia experimental, el conjunto de pruebas iniciales realizado durante una sesión produjo un costo aproximado reportado de:



```text

0.36 MXN

```



Este valor corresponde únicamente a las pruebas ejecutadas en ese periodo y no se utiliza todavía como estimación formal del costo por solicitud.



Para una evaluación posterior se propone registrar:



```text

latencia

tokens de entrada

tokens de salida

costo estimado

```



por ejecución.



\---



\## 32. Decisiones vigentes de SC-04



Después de las implementaciones y pruebas realizadas se mantienen las siguientes decisiones:



\### Vocabulario



```text

71 conceptos canónicos

65 atomic

6 expression

```



\### Modelo utilizado



```text

gemini-3.5-flash-lite

```



\### Política Top-k



```text

Top-k = evidencia

Top-k ≠ restricción rígida

```



\### Política de corrección



```text

usar contexto + vocabulario + probabilidades

sin forzar una interpretación cuando la evidencia sea insuficiente

```



\### Política de salida



```text

todo concept\_id debe pertenecer al vocabulario

```



\### Política unresolved



```text

contenido no representable

→ unresolved

```



\### Política de consistencia semántica



```text

translated\_text

y

concepts



deben representar el mismo significado disponible

```



\### Política de expresiones



```text

si un expression conserva todo el significado

→ utilizar la expresión sin descomposición redundante

```



\---



\## 33. Archivos incorporados



Durante esta fase se incorporaron principalmente:



```text

data/

└── vocabulary/

&#x20;   └── sign2sign\_vocabulary.json



src/

└── text\_processing/

&#x20;   ├── \_\_init\_\_.py

&#x20;   ├── contracts.py

&#x20;   ├── gemini\_client.py

&#x20;   ├── prompt\_builder.py

&#x20;   ├── response\_validator.py

&#x20;   └── vocabulary.py



tests/

├── test\_vocabulary.py

├── test\_prompt\_builder.py

└── test\_response\_validator.py



scripts/

├── test\_sc04\_pipeline.py

├── test\_gemini\_connection.py

├── test\_sc04\_gemini.py

├── test\_sc04\_gemini\_cases.py

├── test\_sc04\_gemini\_robustness.py

└── run\_sc03\_sc04\_webcam.py

```



\---



\## 34. Estado de SC-04



Con los componentes y verificaciones realizadas, SC-04 cuenta con:



\- vocabulario cerrado normalizado;

\- conceptos canónicos independientes del idioma;

\- conceptos `atomic`;

\- conceptos `expression`;

\- normalización de conjugaciones;

\- negación composicional;

\- contrato formal de entrada;

\- contrato formal de salida;

\- construcción automática del prompt;

\- conservación de Top-k;

\- reglas lingüísticas restringidas;

\- conexión real con Gemini;

\- salida JSON estructurada;

\- validación posterior a la respuesta;

\- rechazo de conceptos inventados;

\- manejo de contenido parcialmente fuera del vocabulario;

\- manejo de contenido completamente irresoluble;

\- recuperación de información gramatical implícita;

\- traducción LSM → ASL;

\- traducción ASL → LSM;

\- pruebas unitarias;

\- pruebas funcionales con el LLM;

\- pruebas de robustez;

\- integración funcional real con SC-03 mediante webcam.



\---



\## 35. Flujo funcional alcanzado



El sistema cuenta actualmente con el siguiente flujo:



```text

Usuario

&#x20; ↓

Webcam

&#x20; ↓

SC-03

&#x20; ↓

detección de mano

&#x20; ↓

landmarks

&#x20; ↓

clasificador ASL / LSM

&#x20; ↓

Top-1 + Top-k

&#x20; ↓

TextBuilder

&#x20; ↓

BuiltText

&#x20; ↓

SC-04

&#x20; ↓

CorrectionRequest

&#x20; ↓

PromptBuilder

&#x20; ↓

Gemini 3.5 Flash-Lite

&#x20; ↓

ResponseValidator

&#x20; ↓

TranslationResult

&#x20; ↓

texto corregido

\+

texto traducido

\+

conceptos canónicos

\+

unresolved

```



De esta manera, una secuencia capturada directamente mediante señas puede producir una representación textual y conceptual validada sin intervención manual entre SC-03 y SC-04.



\---



\## 36. Siguiente bloque



El siguiente trabajo corresponde principalmente al subsistema encargado de transformar los conceptos canónicos resultantes en recursos visuales de la lengua de señas destino.



El contrato disponible para dicho subsistema es:



```text

TranslationResult

\+

target\_language

\+

concepts

```



Ejemplo:



```text

target\_language:

ASL



concepts:

FIRST\_PERSON\_SINGULAR

WANT

WATER

```



El siguiente bloque deberá resolver:



\- estructura del repositorio de videos;

\- correspondencia entre `concept\_id` y recurso visual;

\- disponibilidad de recursos LSM y ASL;

\- tratamiento de expresiones;

\- tratamiento de conceptos sin recurso;

\- secuenciación de videos;

\- reproducción de la salida visual.



Antes de considerar SC-04 completamente congelado se podrán realizar pruebas adicionales con entradas reales para caracterizar con mayor precisión su calidad, latencia y costo, sin modificar el contrato funcional ya establecido.

