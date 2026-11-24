# Fase SC-01 — Interfaz web de usuario mediante React + Vite

**Proyecto:** Sign2Sign  
**Fecha de documentación:** 3 de noviembre de 2026  
**Estado:** Implementación local completada, pruebas automáticas aprobadas y flujo principal validado manualmente  
**Subsistema principal:** SC-01 Interfaz de usuario  
**Integración alcanzada:** React + Vite → SC-02 → SC-03 → SC-04 → SC-05 → SC-06 → SC-07  
**Backend:** FastAPI local implementado previamente en SC-02  
**Alcance de esta fase:** interacción web, captura deliberada, construcción textual, presentación de resultados y reproducción visual  

Continuación del trabajo descrito en:

- `FASE_SC02_CONTROL_FLUJO_TRADUCCION_FASTAPI.md`.
- `FASE_SC04_CORRECCION_TRADUCCION_E_INTEGRACION.md`.
- `FASE_SC05_SC06_MAPEO_Y_SALIDA_VISUAL.md`.
- `FASE_SC07_RECURSOS_VISUALES_Y_REGISTRO_LOCAL.md`.

Los resultados consignados en este documento corresponden al desarrollo incremental y a las validaciones realizadas durante la construcción de SC-01. La evaluación formal del sistema completo corresponde posteriormente a las pruebas del sistema definidas para CSI 5.

---

## 1. Objetivo de la fase

El objetivo de SC-01 fue implementar la interfaz web mediante la cual el usuario puede operar el flujo de traducción de Sign2Sign sin interactuar directamente con scripts, ventanas OpenCV o la documentación HTTP del backend.

Antes de esta fase, SC-02 ya exponía mediante FastAPI los mecanismos necesarios para crear sesiones, procesar capturas, construir texto, solicitar traducciones y entregar los recursos visuales. SC-01 utiliza dichos contratos para presentar una experiencia de interacción unificada en navegador.

```text
Usuario
    ↓
SC-01: React + TypeScript + Vite
    ↓
HTTP / JSON / imagen binaria
    ↓
SC-02: FastAPI y control del flujo
    ↓
SC-03: reconocimiento y construcción textual
    ↓
SC-04: corrección y traducción
    ↓
SC-05: mapeo de recursos
    ↓
SC-06: plan de salida visual
    ↓
SC-07: recursos y estado temporal
    ↓
SC-01: presentación textual y reproducción de videos
```

SC-01 no incorpora lógica de clasificación, corrección lingüística, traducción, selección semántica de recursos ni reordenamiento de conceptos. Su responsabilidad se limita a la interacción, presentación y control de la experiencia de usuario.

---

## 2. Arquitectura disponible al inicio

Al comenzar SC-01 se disponía de:

| Elemento | Estado previo |
|---|---|
| SC-02 | API FastAPI local con sesiones, captura, comandos, traducción y recursos |
| SC-03 | Reconocedores ASL/LSM y constructor textual |
| SC-04 | Corrección y traducción mediante Gemini |
| SC-05 | Mapeo de conceptos a recursos visuales |
| SC-06 | Construcción del plan de reproducción |
| SC-07 | Sesiones temporales y repositorio de recursos |
| Recursos visuales | 71 asociaciones ASL y 71 LSM |
| Contrato HTTP | Disponible bajo `/api/v1` |

La interfaz se desarrolló como cliente del backend existente. No fue necesario modificar SC-02–SC-07 para implementar el flujo web.

---

## 3. Decisiones de alcance

Durante la implementación se conservaron las siguientes decisiones:

1. Utilizar React con TypeScript y Vite.
2. Mantener SC-01 como cliente del backend, sin duplicar lógica del núcleo.
3. Utilizar `fetch` para la comunicación HTTP.
4. Configurar Vite con proxy `/api` hacia FastAPI durante el desarrollo local.
5. Mantener captura deliberada: una acción de usuario equivale a una imagen enviada al backend.
6. Mantener activación y desactivación manual de la cámara.
7. Utilizar botones visibles y atajos de teclado como métodos complementarios.
8. Mantener el backend como fuente autoritativa del texto de la sesión.
9. Reproducir los videos en el navegador sin llamar a `VisualSequencePlayer.play()`.
10. Mantener la secuencia recibida localmente para permitir repetirla sin otra llamada a Gemini.
11. Incorporar español e inglés como idiomas de interfaz, independientes de la lengua de señas.
12. Mantener TTS como función complementaria y no como canal principal de salida.
13. Priorizar feedback visual para las operaciones relevantes.
14. Incorporar ayuda de uso y contenido informativo sin convertir la interfaz en una aplicación multipágina.
15. Añadir timeline y modo enfoque sin duplicar la lógica del reproductor.
16. Conservar la paleta y estructura general una vez alcanzada una versión funcional.
17. No incorporar reconocimiento continuo, autenticación, comunicación remota entre usuarios ni almacenamiento persistente.

---

# Parte I — Arquitectura y organización del frontend

## 4. Estructura general

La implementación quedó contenida en `frontend/`.

Entre los archivos y componentes principales se encuentran:

```text
frontend/
├── src/
│   ├── api/
│   │   └── sign2signApi.ts
│   ├── components/
│   │   ├── CameraPanel.tsx
│   │   ├── RecognitionPanel.tsx
│   │   ├── TextBuilderPanel.tsx
│   │   ├── TranslationResult.tsx
│   │   ├── SequencePlayer.tsx
│   │   ├── SequenceTimeline.tsx
│   │   ├── ShortcutHelp.tsx
│   │   ├── UserGuide.tsx
│   │   └── AboutSection.tsx
│   ├── hooks/
│   │   ├── useCamera.ts
│   │   ├── useSign2SignSession.ts
│   │   ├── useShortcuts.ts
│   │   └── usePlayerFocus.ts
│   ├── i18n/
│   │   ├── messages.ts
│   │   └── UiLanguage.tsx
│   ├── presentation/
│   │   └── conceptLabels.ts
│   ├── types/
│   │   └── api.ts
│   ├── App.tsx
│   ├── main.tsx
│   └── styles.css
├── vite.config.ts
├── package.json
└── package-lock.json
```

La estructura separa acceso HTTP, cámara, estado de sesión, atajos, presentación, internacionalización y reproducción visual.

## 5. Responsabilidades principales

| Componente | Responsabilidad |
|---|---|
| `App.tsx` | Organiza el flujo principal de la interfaz |
| `sign2signApi.ts` | Centraliza las llamadas HTTP a SC-02 |
| `useCamera.ts` | Gestiona permisos, activación, vista previa y captura |
| `useSign2SignSession.ts` | Mantiene el ciclo de sesión y las operaciones principales |
| `useShortcuts.ts` | Implementa los atajos sin interferir con controles editables |
| `CameraPanel.tsx` | Presenta webcam y captura deliberada |
| `TextBuilderPanel.tsx` | Presenta la cadena y comandos de edición |
| `RecognitionPanel.tsx` | Presenta letra, confianza y alternativas |
| `TranslationResult.tsx` | Presenta texto reconocido, corregido y traducido |
| `SequencePlayer.tsx` | Reproduce los recursos en orden |
| `SequenceTimeline.tsx` | Representa y permite navegar la secuencia visual |
| `usePlayerFocus.ts` | Controla el modo enfoque y su manejo de foco/scroll |
| `UserGuide.tsx` | Explica el uso operativo en cinco pasos |
| `AboutSection.tsx` | Presenta descripción, alcance y cierre académico |
| `messages.ts` | Centraliza textos ES/EN |
| `conceptLabels.ts` | Convierte conceptos a etiquetas amigables reutilizando el vocabulario canónico |

---

# Parte II — Comunicación con SC-02

## 6. Proxy local mediante Vite

Durante el desarrollo, Vite redirige:

```text
/api → http://127.0.0.1:8000
```

La interfaz utiliza rutas relativas como:

```text
/api/v1/sessions
/api/v1/sessions/{id}/captures
/api/v1/sessions/{id}/translations
/api/v1/resources/{language}/{concept_id}
```

Esta configuración evita incorporar CORS al backend local y mantiene desacoplada la dirección física del servicio respecto a los componentes de React.

## 7. Endpoints consumidos

SC-01 utiliza los contratos existentes de SC-02:

| Método | Ruta | Uso en SC-01 |
|---|---|---|
| GET | `/api/v1/health` | Indicador de disponibilidad |
| POST | `/api/v1/sessions` | Crear sesión |
| GET | `/api/v1/sessions/{id}` | Sincronizar estado cuando es necesario |
| POST | `/api/v1/sessions/{id}/captures` | Enviar una captura |
| POST | `/api/v1/sessions/{id}/commands` | Espacio, borrado y limpieza |
| POST | `/api/v1/sessions/{id}/translations` | Ejecutar traducción |
| DELETE | `/api/v1/sessions/{id}` | Cerrar sesión |
| GET | `/api/v1/resources/{language}/{concept_id}` | Obtener video |

No se implementaron polling, trabajos diferidos ni WebSockets. La petición de traducción permanece abierta hasta recibir el resultado final de SC-02.

## 8. Ciclo de sesión

La interfaz conserva únicamente el identificador de sesión en memoria.

```text
Selección de lengua
    ↓
Creación de sesión
    ↓
Capturas y comandos
    ↓
Traducción
    ↓
Presentación del resultado
    ↓
Nueva sesión / cambio de lengua / cierre
    ↓
Eliminación de sesión
```

El cambio del idioma visual ES/EN no crea una nueva sesión ni modifica la lengua de señas seleccionada.

---

# Parte III — Cámara y construcción textual

## 9. Activación manual de cámara

La cámara se gestiona mediante:

```javascript
navigator.mediaDevices.getUserMedia()
```

Su activación y desactivación son acciones explícitas del usuario.

Cuando la cámara está apagada:

- la vista previa no se utiliza;
- `Capturar seña` permanece deshabilitado;
- el atajo `Enter` no ejecuta una captura.

No se agregó reconocimiento continuo.

## 10. Vista previa y orientación

La vista previa puede mostrarse espejada para facilitar la interacción frente a la webcam.

El espejo se aplica únicamente a la presentación mediante CSS. La imagen enviada al backend no se invierte por este efecto.

La captura se realiza tomando el frame actual del elemento `<video>`, dibujándolo sobre un canvas y generando un `Blob` JPEG que se envía directamente como cuerpo binario.

No se utiliza Base64.

## 11. Captura deliberada

El flujo principal es:

```text
Usuario realiza una seña
    ↓
Capturar seña / Enter
    ↓
Frame actual
    ↓
Blob JPEG
    ↓
POST /captures
    ↓
SC-03
    ↓
Letra + confianza + Top-k
    ↓
Actualización de la interfaz
```

Una captura representa una letra. La interfaz no agrega directamente la predicción a una cadena local autoritativa; utiliza el estado devuelto por el backend.

## 12. Construcción del mensaje

Los comandos disponibles son:

| Acción | Control visible | Atajo |
|---|---|---|
| Capturar | `Capturar seña` | `Enter` |
| Separar palabras | `Espacio` | `Space` |
| Borrar último | `Borrar último` | `Backspace` |
| Limpiar | `Limpiar` | `Esc` |
| Traducir | `Traducir mensaje` | `T` |
| Repetir secuencia | `Repetir secuencia` | `P` |

Los botones muestran una representación tipo `kbd` de su tecla correspondiente para facilitar el descubrimiento de los atajos.

`useShortcuts` evita interceptar teclas cuando el foco se encuentra en elementos interactivos o editables, durante composición de texto, al utilizar modificadores o cuando la tecla permanece sostenida.

---

# Parte IV — Idioma, ayuda y accesibilidad visual

## 13. Idioma de interfaz

Se incorporó un sistema ligero ES/EN.

Comportamiento automático:

```text
LSM como lengua de origen → interfaz ES
ASL como lengua de origen → interfaz EN
```

El usuario puede sobrescribir esta asociación mediante:

```text
ES | EN | Automático
```

El idioma visual es independiente de:

- `source_language`;
- `target_language`;
- idioma del TTS;
- etiquetas semánticas de la salida.

## 14. Feedback visual

Las acciones relevantes producen retroalimentación visual.

Entre los estados contemplados se encuentran:

- cámara activa/inactiva;
- captura válida;
- ausencia de seña válida;
- operación en curso;
- traducción finalizada;
- errores de conexión;
- advertencias por contenido no resuelto.

Una captura correcta resalta la última predicción y muestra una confirmación visual. Los mensajes no dependen exclusivamente de sonido.

## 15. Estado del backend

La cabecera conserva un indicador circular:

```text
verde → backend disponible
rojo  → backend no disponible
```

El texto `Backend conectado/desconectado` fue retirado de la presentación visible para reducir ruido visual.

La información permanece disponible mediante atributos accesibles y tooltip. La lógica del health check no fue modificada.

## 16. Guía de uso

Se agregó `Cómo usar / How to use` como ayuda operativa accesible desde la cabecera.

La guía contiene cinco pasos:

1. Seleccionar la lengua.
2. Activar la cámara.
3. Construir el mensaje.
4. Traducir.
5. Consultar el resultado.

La guía incluye los atajos principales, pero no explica detalles internos como sesiones, subsistemas o endpoints.

Abrir o cerrar la guía:

- no crea solicitudes HTTP;
- no elimina la sesión;
- no apaga la cámara;
- no borra el texto;
- no reinicia la reproducción.

## 17. Sección Acerca de

El contenido institucional se separó de la guía operativa y se ubicó al final real de la página.

Incluye:

- descripción breve de Sign2Sign;
- alcance del prototipo;
- principales limitaciones;
- cierre académico de ESCOM-IPN.

El cierre utilizado establece que Sign2Sign fue desarrollado en la Escuela Superior de Cómputo del Instituto Politécnico Nacional con fines académicos como parte de un Trabajo Terminal.

Esta información utiliza menor jerarquía visual que las funciones de traducción.

## 18. Consideraciones de inclusividad

La interfaz se refinó bajo un enfoque visual-first:

- ninguna operación principal requiere audio;
- iconos relevantes se acompañan con texto;
- los atajos se presentan también como botones;
- los estados se expresan mediante texto/símbolos además de color;
- se mantiene foco visible;
- las instrucciones son cortas;
- se respeta `prefers-reduced-motion`;
- TTS es complementario y nunca automático.

No se afirma conformidad formal con WCAG ni validación por parte de la comunidad Sorda, ya que no se realizó una evaluación formal de ese tipo dentro del alcance actual.

Se consideró incorporar videos de instrucciones en LSM y ASL. Por razones de tiempo, esta mejora quedó identificada como trabajo futuro.

---

# Parte V — Presentación de resultados

## 19. Resultado textual

SC-01 presenta:

- texto reconocido;
- texto corregido;
- traducción.

Los conceptos, elementos `unresolved` y datos técnicos pueden mostrarse de forma secundaria cuando corresponde.

No se modifica el contenido semántico recibido desde el backend.

## 20. Herramientas complementarias

Los textos disponen de:

- copiar;
- escuchar texto;
- imprimir.

La copia utiliza las funciones del navegador.

La impresión utiliza el mecanismo de impresión del navegador y estilos orientados a una salida legible.

## 21. Text-to-Speech

TTS utiliza Web Speech API:

```javascript
SpeechSynthesisUtterance
```

Idiomas:

```text
texto español → es-MX
texto inglés  → en-US
```

El idioma del TTS se deriva del contenido y no del idioma visual de la interfaz.

Antes de iniciar una nueva lectura se cancela la anterior para evitar voces superpuestas.

TTS se conserva como función complementaria para escenarios donde una persona signante desea comunicar el resultado a una persona oyente.

No se incorporaron servicios externos ni configuración avanzada de voz.

---

# Parte VI — Reproducción visual

## 22. Reproductor web

SC-01 reproduce la secuencia de recursos mediante `<video>`.

El backend entrega URLs HTTP ordenadas. La interfaz conserva dicha secuencia localmente.

Controles disponibles:

- reproducir/pausar;
- anterior;
- siguiente;
- repetir secuencia;
- posición actual.

Cuando un video termina, el reproductor avanza al siguiente recurso hasta completar la secuencia.

## 23. Repetición local

`Repetir secuencia` reinicia la navegación de los recursos ya recibidos.

```text
resultado disponible
    ↓
P / Repetir secuencia
    ↓
índice = 0
    ↓
reproducción local
```

No se realiza otra llamada a:

```text
POST /translations
```

Por tanto, repetir no vuelve a invocar Gemini ni los subsistemas de corrección y mapeo.

## 24. Timeline visual

Se incorporó una timeline para sustituir la presentación directa de identificadores técnicos como:

```text
HOW_ARE_YOU
FIRST_PERSON_SINGULAR
```

La timeline distingue:

- elemento completado;
- elemento actual;
- elemento pendiente.

También permite seleccionar un elemento y navegar directamente a dicho recurso. La navegación es local y deja el video seleccionado pausado.

## 25. Etiquetas amigables de conceptos

Las etiquetas no se obtienen mediante una segunda lógica de traducción en frontend.

`conceptLabels.ts` reutiliza:

```text
data/vocabulary/sign2sign_vocabulary.json
```

La etiqueta seleccionada depende de la lengua de salida:

```text
salida LSM → etiqueta canónica en español
salida ASL → etiqueta canónica en inglés
```

Esto permite, por ejemplo, presentar una etiqueta comprensible al usuario en lugar de un identificador técnico interno.

El `concept_id` original permanece sin modificaciones y puede conservarse para detalles técnicos.

## 26. Modo enfoque

Se incorporó un modo destinado a priorizar la consulta de la salida visual.

El modo enfoque utiliza el mismo:

- `SequencePlayer`;
- elemento `<video>`;
- índice;
- timeline;
- estado;
- secuencia;
- controles.

No existe un reproductor alternativo con lógica duplicada.

```text
Vista normal: 2 / 3
    ↓
Modo enfoque
    ↓
Continúa: 2 / 3
    ↓
Cerrar
    ↓
Vista normal: 2 / 3
```

La vista cubre temporalmente el contenido principal mediante estilos y prioriza:

1. lengua de salida;
2. timeline;
3. video;
4. controles;
5. cierre.

Al abrir se controla el foco, se evita scroll accidental de la página subyacente y al cerrar se devuelve el foco al control original.

El atajo `Esc` se protege para evitar que el cierre del modo enfoque provoque simultáneamente el comando `Limpiar`.

---

# Parte VII — Manejo de errores y estados

## 27. Errores de conexión

Cuando FastAPI no está disponible, la interfaz traduce el fallo a un mensaje entendible y evita mostrar errores internos de `fetch` o stack traces.

El indicador de backend cambia al estado no disponible.

## 28. Contenido unresolved

Cuando la traducción termina correctamente pero existen elementos sin recurso visual:

- el resultado textual se conserva;
- se presenta una advertencia;
- el caso no se trata como un fallo fatal.

Si existen recursos válidos junto con elementos no resueltos, la interfaz puede reproducir la parte disponible y mantener la advertencia correspondiente.

## 29. Protección ante acciones duplicadas

La interfaz deshabilita o bloquea las acciones correspondientes cuando existen operaciones en curso.

Entre otros casos:

- no envía capturas duplicadas;
- no envía dos traducciones simultáneas;
- no ejecuta captura si la cámara está inactiva;
- respeta los estados de sesión devueltos por SC-02.

---

# Parte VIII — Evolución incremental

## 30. Primera versión funcional

La primera versión estableció:

- React + TypeScript + Vite;
- cliente HTTP;
- ciclo de sesión;
- cámara;
- construcción textual;
- resultados;
- reproducción secuencial;
- TTS;
- copia;
- impresión.

La primera suite frontend quedó en:

```text
36 pruebas aprobadas
```

## 31. Refinamiento de idioma, teclado y feedback

Posteriormente se incorporaron:

- ES/EN automático;
- override manual;
- atajos;
- ayuda de atajos;
- feedback visual;
- mejora de estados;
- auto-scroll al resultado;
- ajustes de TTS y accesibilidad.

La suite aumentó a:

```text
68 pruebas aprobadas
```

## 32. Guía y refinamientos de interfaz

Se añadieron progresivamente:

- `Cómo usar / How to use`;
- indicador de backend simplificado;
- representación `kbd` en botones;
- mejoras de contraste;
- contenido informativo.

Las distintas iteraciones registraron:

```text
71 pruebas
78 pruebas
```

## 33. Timeline y modo enfoque

La salida visual se refinó mediante:

- timeline;
- etiquetas amigables;
- navegación local;
- modo enfoque;
- gestión de foco;
- conservación del estado del reproductor.

La suite alcanzó:

```text
87 pruebas aprobadas
```

## 34. Reorganización final

`About`, alcance y cierre académico se retiraron de la guía operativa y se trasladaron al final de la página principal.

Resultado final registrado:

```text
88 pruebas frontend aprobadas
```

---

# Parte IX — Validación

## 35. Pruebas automáticas

La suite frontend verifica, entre otros aspectos:

- cliente HTTP;
- render inicial;
- sesiones;
- selección de lengua;
- captura;
- comandos;
- traducción;
- errores;
- `unresolved`;
- ES/EN;
- override manual;
- atajos;
- protección de controles;
- TTS;
- reproductor;
- repetición local;
- guía;
- indicador del backend;
- timeline;
- etiquetas amigables;
- navegación de timeline;
- modo enfoque;
- foco y scroll;
- conservación del estado.

Resultado final:

```text
npm test
88 pruebas aprobadas
```

## 36. Build y dependencias

Las verificaciones finales registradas fueron:

| Comprobación | Resultado |
|---|---|
| `npm test` | 88 pruebas aprobadas |
| `npm run build` | Correcto |
| `npm audit` | 0 vulnerabilidades |
| `git diff --check` | Sin errores |

## 37. Validación manual

Durante el desarrollo se realizaron pruebas manuales con navegador, cámara y backend reales.

Se comprobó:

- conexión React ↔ FastAPI;
- selección ASL/LSM;
- activación/desactivación manual de cámara;
- captura de señas reales;
- confianza y Top-k;
- construcción mediante botones y teclado;
- corrección y traducción real con Gemini;
- salida visual ASL y LSM;
- reproducción secuencial;
- repetición sin nueva traducción;
- TTS;
- copia;
- impresión;
- comportamiento `unresolved`;
- cambio de idioma ES/EN;
- preservación de sesión.

Entre los casos observados durante la validación se incluyeron secuencias con errores de reconocimiento recuperadas posteriormente por SC-04, comprobando el flujo completo desde la webcam hasta la salida visual.

Las pruebas completas del sistema y su evaluación formal se documentarán en CSI 5.

---

# Parte X — Ejecución local

## 38. Inicio del backend

Desde la raíz del proyecto:

```powershell
.venv\Scripts\activate
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --workers 1
```

## 39. Inicio del frontend

En otra terminal:

```powershell
cd frontend
npm ci
npm run dev
```

Dirección local:

```text
http://127.0.0.1:5173
```

La interfaz utiliza el proxy configurado por Vite para comunicarse con FastAPI.

---

# Parte XI — Limitaciones y trabajo futuro

## 40. Limitaciones actuales

- El prototipo se ejecuta en entorno local.
- No existe autenticación ni persistencia de usuarios.
- No se implementa comunicación remota persona a persona.
- No existe reconocimiento continuo de lengua de señas.
- La entrada se basa principalmente en deletreo manual mediante configuraciones estáticas.
- La salida utiliza un vocabulario visual previamente delimitado.
- El sistema no representa toda la gramática ni complejidad lingüística de ASL o LSM.
- TTS depende de las capacidades y voces disponibles en el navegador/sistema operativo.
- La interfaz aplica criterios de accesibilidad visual, pero no cuenta con certificación WCAG.
- No se realizó una evaluación formal de usabilidad con participantes de la comunidad Sorda.

## 41. Mejoras identificadas para trabajo futuro

Se identificaron como posibles extensiones:

- guía de uso mediante video signado en LSM;
- guía de uso mediante video signado en ASL;
- evaluación formal con usuarios signantes;
- revisión de accesibilidad con metodología específica;
- despliegue remoto/multiusuario;
- ampliación del vocabulario;
- reconocimiento dinámico o continuo.

El video explicativo signado se consideró útil, pero se dejó explícitamente como trabajo futuro por las restricciones de tiempo y alcance del Trabajo Terminal.

---

# Parte XII — Estado de cierre

## 42. Estado final de SC-01

```text
[x] React + TypeScript + Vite.
[x] Comunicación mediante proxy con SC-02.
[x] Creación y cierre de sesiones.
[x] Selección ASL ↔ LSM.
[x] Cámara manual.
[x] Captura deliberada.
[x] Texto autoritativo desde backend.
[x] Confianza y Top-k.
[x] Comandos de edición.
[x] Botones + atajos de teclado.
[x] ES / EN / Automático.
[x] Traducción real mediante SC-02.
[x] Resultado reconocido, corregido y traducido.
[x] Manejo de unresolved.
[x] Copiar.
[x] Imprimir.
[x] TTS en es-MX y en-US.
[x] Reproductor secuencial.
[x] Repetición local sin nueva llamada a Gemini.
[x] Timeline visual.
[x] Etiquetas amigables desde vocabulario canónico.
[x] Modo enfoque.
[x] Guía Cómo usar / How to use.
[x] Acerca de Sign2Sign y alcance al final de página.
[x] Indicador visual de disponibilidad del backend.
[x] Feedback visual y foco de teclado.
[x] prefers-reduced-motion.
[x] Suite frontend: 88 pruebas aprobadas.
[x] Build de producción correcto.
[x] npm audit: 0 vulnerabilidades.
[x] git diff --check sin errores.
[x] Flujo principal validado manualmente.
[ ] Evaluación formal con usuarios de la comunidad Sorda.
[ ] Video de instrucciones signado en LSM/ASL.
[ ] Pruebas finales del sistema documentadas en CSI 5.
```

Con el alcance actual, SC-01 se considera funcional y congelada para la etapa de cierre del prototipo. Las siguientes actividades deben concentrarse principalmente en pruebas del sistema, documentación final, presentación y defensa de las decisiones técnicas adoptadas.

---

**Fin del documento de fase SC-01 — Interfaz web de usuario mediante React + Vite.**
