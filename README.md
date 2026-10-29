Activar el entorno virtual 
    .venv\Scripts\activate

Probar SC-03
    python scripts\run_sc03_webcam.py --language LSM

Probar SC-03 y SC-04
    python scripts\run_sc03_sc04_webcam.py --language LSM

Probar nucleo SC-03 + SC-04 + SC-05 + SC-06 + SC-07
    python scripts\run_sc03_sc04_sc05_sc06_webcam.py --language LSM
## SC-02: backend FastAPI local

Desde la raiz del proyecto, con Python 3.12 y el entorno activado:

```powershell
python -m pip install -r requirements.txt
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --workers 1
```

Configura `GEMINI_API_KEY` en `.env` usando `.env.example` como referencia.
`GEMINI_MODEL` permite elegir otro modelo compatible con SC-04. Los modelos
ASL/LSM y sus escaladores deben existir en las rutas usadas por los scripts
actuales; no se descargan al iniciar. El ciclo de vida carga los componentes
una vez y falla al arrancar si falta una dependencia necesaria.

Documentacion HTTP: http://127.0.0.1:8000/docs

| Metodo | Ruta bajo `/api/v1` | Uso |
|---|---|---|
| GET | `/health` | Disponibilidad local; no llama al proveedor LLM |
| POST | `/sessions` | JSON `{"source_language":"LSM"}` o `ASL` |
| GET | `/sessions/{id}` | Texto, estado y ultimo resultado |
| POST | `/sessions/{id}/captures` | Imagen binaria JPEG/PNG en el cuerpo HTTP |
| POST | `/sessions/{id}/commands` | JSON `{"command":"space"}`, `backspace` o `clear` |
| POST | `/sessions/{id}/translations` | Espera y devuelve el resultado final |
| DELETE | `/sessions/{id}` | Elimina el contenido temporal |
| GET | `/resources/{language}/{concept_id}` | Video MP4 registrado, con soporte Range |

La captura usa `Content-Type: image/jpeg` o `image/png`, sin multipart.
Por ejemplo, desde un futuro cliente: `fetch(url, {method: 'POST',
headers: {'Content-Type': blob.type}, body: blob})`. El servidor valida
firma, decodificacion y dimensiones: maximo 5 MiB y 4096 x 4096 pixeles
como area maxima. Decodifica a BGR, ignora orientacion EXIF y no refleja ni
rota la imagen. Un espejo CSS de la vista previa no debe alterar los pixeles
enviados. Cada peticion representa una captura deliberada.

Cada sesion mantiene su propio TextBuilder. Los estados son `capturing`,
`processing`, `ready`, `failed` y `closed` (este ultimo solo durante el borrado).
La captura o edicion invalida el resultado anterior. Hay un maximo de 100
sesiones y 500 letras por cadena. No hay persistencia ni expiracion automatica:
se libera memoria con DELETE o al cerrar el proceso. Un reinicio pierde las
sesiones. Ejecutar exclusivamente con un worker.

Las rutas sincronas de FastAPI se ejecutan en su threadpool. La captura recibe
el cuerpo con un limite y delega decodificacion e inferencia al threadpool.
Un bloqueo por sesion rechaza solicitudes simultaneas con 409, incluida su
consulta o borrado mientras esta ocupada. Un bloqueo global serializa SC-03;
otro serializa el flujo SC-04 a SC-06, sin bloquear el event loop.

La traduccion devuelve 200 y campos `state`, `error`, `resources` y
`latency_seconds`, junto al contrato textual de SC-04. Cada recurso incluye
una URL relativa HTTP bajo `/api/v1/resources/`, posicion y total. Un error
visual devuelve 200 con `state: failed`, la etapa fallida y el resultado
textual preservado. Si solo hay unresolved, no se crea un plan vacio.
Un fallo SC-04 devuelve 502 con etapa SC-04 y conserva la entrada para
reintento. No se publican mensajes internos de excepciones, rutas fisicas ni
la respuesta cruda de Gemini. No se abre ninguna ventana OpenCV.

El controlador es sincrono e independiente de las rutas: una futura capa de
trabajos podra invocarlo sin cambiar los contratos de SC-03 a SC-06.
Las sesiones temporales pertenecen a SC-07 en
`src/resources/temporary_session_store.py`.

No hay CORS habilitado: el futuro Vite puede usar un proxy `/api` al backend.
Antes de usar origenes distintos habra que configurar CORS para ese origen.
Este servicio local no incluye autenticacion. Debe permanecer en loopback.
La compatibilidad de los codecs MP4 con el navegador requiere validacion al
integrar SC-01; OpenCV y un navegador pueden soportar codecs diferentes.

Pruebas automaticas, sin camara ni llamadas reales a Gemini:

```powershell
python -m unittest discover -s tests -v
```

Los scripts interactivos existentes siguen disponibles para validar el nucleo
local. FastAPI no los importa ni ejecuta. En el entorno actual Starlette emite
una advertencia de deprecacion de su adaptador TestClient para httpx; las
pruebas funcionan con las versiones declaradas.

## SC-01: interfaz React + TypeScript + Vite

El frontend está en `frontend/`. Requiere Node.js 20.18 o posterior y npm.
Se usa Vite 6 para mantener compatibilidad con el Node instalado en este
proyecto. Las versiones exactas de dependencias están en `package-lock.json`.

Inicia primero FastAPI desde la raíz, con su configuración y artefactos locales:

```powershell
.venv\Scripts\activate
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --workers 1
```

En otra terminal:

```powershell
cd frontend
npm ci
npm run dev
```

Abre `http://127.0.0.1:5173`. Vite redirige `/api` a
`http://127.0.0.1:8000`, incluyendo videos y solicitudes Range. No requiere
CORS ni cambios en FastAPI. La vista previa de `npm run preview` también tiene
el proxy local; el directorio `dist` por sí solo necesita un servidor que
redirija `/api` al backend. No se implementó despliegue en esta fase.

La interfaz permite iniciar, cambiar y cerrar sesiones, capturar una imagen
deliberadamente, editar el texto devuelto por el backend, traducir y reproducir
la secuencia recibida. También ofrece copia, lectura en voz alta e impresión.
No se guardan sesiones en localStorage ni se ejecutan modelos en el navegador.
La vista previa se refleja solo mediante CSS; el canvas no invierte los
píxeles. La captura JPEG se limita a un encuadre de hasta 1280 × 720.

Las traducciones esperan una respuesta final, sin polling ni trabajos. Tras
completarlas se consulta una vez la sesión para actualizar el estado del
backend. No se reenvían capturas ni traducciones automáticamente ante errores.
La repetición de videos usa exclusivamente la secuencia ya recibida.

La cámara requiere localhost o un contexto seguro y permiso del usuario.
Los clips de señas se reproducen sin audio para permitir inicio automático;
si el navegador lo bloquea, utiliza Reproducir. La voz utiliza Web Speech API,
con `en-US` para inglés y `es-MX` para español; su disponibilidad depende de
las voces del sistema/navegador. Copiar requiere permiso de portapapeles.

Cerrar sesión elimina explícitamente su estado en FastAPI. Al desmontar o
salir de la página se intenta un DELETE con `keepalive`; esa limpieza no está
garantizada si la red falla o el backend mantiene la sesión ocupada. El backend
no tiene expiración automática, por lo que conviene cerrar la sesión desde
la interfaz antes de abandonar la página. Las sesiones se pierden al reiniciar
FastAPI. No uses varios workers para esta versión.

Pruebas automáticas y compilación:

```powershell
cd frontend
npm test
npm run build
npm audit
```

Vitest y Testing Library simulan FastAPI, cámara y APIs multimedia. Las
pruebas no llaman a Gemini ni usan una cámara real. No se creó todavía el
documento de cierre de fase SC-01: depende de la validación manual siguiente.

### Validación manual pendiente de SC-01

La segunda iteración de UX conserva la cámara y el flujo existentes. El idioma
de interfaz es inglés para origen ASL y español para origen LSM. El selector
ES / EN permite sobrescribir esa elección sin cambiar la lengua de señas ni
crear otra sesión; Automático restaura el comportamiento por lengua de origen.
La elección manual se conserva durante el uso de la página y se restablece
al recargar. El idioma hablado por TTS depende del texto, no de la interfaz.

Además de los botones, están disponibles estos atajos:

| Tecla | Acción |
|---|---|
| Enter | Capturar seña, solo con sesión y cámara activa |
| Space | Agregar espacio |
| Backspace | Borrar último elemento |
| Escape | Limpiar |
| T | Traducir mensaje |
| P | Repetir la secuencia disponible, sin volver a traducir |

La ayuda Atajos / Shortcuts muestra este mapa. Los atajos se ignoran dentro
de botones, campos, selectores, contenido editable y otros controles. También
se ignoran teclas mantenidas, combinaciones con modificadores y composición
de texto. Conservan las restricciones de sus botones correspondientes.

Una captura válida muestra una confirmación visual; la no detección aparece
como advertencia para reintentar. La confirmación de traducción deja de ser
vigente al iniciar otra operación. Un resultado nuevo correcto desplaza la
vista hacia su sección; se respeta la preferencia de movimiento reducido.

La V2 requiere una nueva validación manual: idioma automático y manual,
atajos fuera y dentro de controles, cámara activa/inactiva, frase y traducción
reales, auto-scroll, repetición con P, TTS en ambas lenguas, copia, impresión,
unresolved, cambio de sesión y desconexión del backend. Las pruebas
automatizadas no cierran esta validación ni sustituyen evaluación con usuarios.

1. Iniciar FastAPI con un worker y verificar su arranque completo.
2. Iniciar Vite con `npm run dev`.
3. Abrir el frontend y comprobar que indique Backend conectado.
4. Seleccionar ASL e iniciar sesión; comprobar ASL → LSM.
5. Activar webcam y aceptar el permiso. Probar también la denegación y recuperación del permiso.
6. Capturar una letra real, comprobando que el espejo visual no altere su reconocimiento.
7. Revisar la confianza y desplegar las alternativas Top-k.
8. Construir una frase en inglés con capturas y espacios; probar Borrar último y Limpiar.
9. Pulsar Traducir y verificar que los controles se bloqueen mientras espera la llamada real a Gemini.
10. Comprobar texto reconocido, corregido y traducción al español.
11. Verificar reproducción automática de todos los videos en el orden recibido y el indicador de posición.
12. Probar Pausar y Reproducir dentro de un clip.
13. Probar Anterior y Siguiente, incluidos los límites de la secuencia.
14. Repetir secuencia; en la pestaña Network comprobar que no se envíe otro POST a `/translations`.
15. Escuchar el texto corregido en inglés (`en-US`).
16. Escuchar la traducción en español (`es-MX`).
17. Copiar ambos textos y probar Imprimir con la vista previa de impresión.
18. Probar una entrada fuera del vocabulario: verificar la respuesta real `unresolved`, total o parcial, sin tratarla como fallo fatal.
19. Cambiar a LSM, comprobar que se elimine la sesión anterior y repetir una frase en español con salida ASL. Probar también error SC-04 y reintento cuando sea posible.
20. Cerrar sesión y desactivar cámara. Comprobar liberación de cámara, eliminación de sesión y comportamiento al cerrar la pestaña.

Validar además la interfaz en laptop y tablet, navegación por teclado, foco,
compatibilidad de los códecs MP4 y funcionamiento de voz/portapapeles en el
navegador utilizado durante la defensa. Las pruebas simuladas no sustituyen
estas comprobaciones.
