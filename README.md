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
