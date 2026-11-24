# Sign2Sign

**Sign2Sign** es un prototipo académico de traducción bidireccional entre **American Sign Language (ASL)** y **Lengua de Señas Mexicana (LSM)** desarrollado mediante técnicas de visión por computadora, aprendizaje automático y modelos de lenguaje.

El sistema reconoce señas estáticas correspondientes al alfabeto manual mediante una webcam, construye una representación textual intermedia, corrige posibles errores de reconocimiento, realiza la traducción entre inglés y español y genera una salida visual mediante recursos de la lengua de señas destino.

El prototipo permite realizar los siguientes flujos:

- **ASL → LSM**
- **LSM → ASL**

---

## Funcionamiento general

El flujo de Sign2Sign se divide en las siguientes etapas:

```text
Webcam
   ↓
Reconocimiento de la seña
   ↓
Construcción del texto
   ↓
Corrección y traducción mediante LLM
   ↓
Obtención de conceptos canónicos
   ↓
Mapeo de recursos visuales
   ↓
Reproducción de la traducción
```

### 1. Captura

El usuario selecciona la lengua de señas de origen y activa la webcam desde la interfaz web.

Cada captura corresponde deliberadamente a una letra del alfabeto manual.

### 2. Reconocimiento de señas

La imagen capturada se procesa mediante **MediaPipe Hand Landmarker**, que obtiene 21 landmarks de la mano.

Cada landmark contiene coordenadas `x`, `y` y `z`, formando un vector de **63 características**.

Sign2Sign utiliza dos clasificadores independientes:

- **ASL:** MLP con características `img_norm`.
- **LSM:** MLP con características `world_norm`.

Además de la predicción principal, el sistema conserva las tres alternativas con mayor probabilidad (**Top-3**).

### 3. Construcción textual

Las letras reconocidas se incorporan progresivamente a una cadena.

El usuario puede:

- capturar una letra;
- insertar espacios;
- borrar el último elemento;
- limpiar el texto;
- enviar el mensaje para su traducción.

### 4. Corrección y traducción

El texto reconocido es enviado al módulo de procesamiento lingüístico.

Este módulo utiliza **Google Gemini API** para:

1. corregir posibles errores producidos durante el reconocimiento;
2. traducir entre los idiomas puente inglés y español;
3. obtener una representación semántica mediante conceptos controlados.

El procesamiento utiliza tanto el texto construido como las alternativas Top-3 generadas durante el reconocimiento.

### 5. Vocabulario canónico

Sign2Sign utiliza un vocabulario cerrado de **71 conceptos canónicos**:

- 65 conceptos atómicos;
- 6 expresiones.

El modelo de lenguaje únicamente puede producir identificadores pertenecientes a este vocabulario para las siguientes etapas del sistema.

Los elementos que no pueden ser representados se registran como `unresolved`.

### 6. Salida visual

Los conceptos obtenidos son asociados con recursos visuales previamente registrados.

El repositorio contiene:

- **71 recursos ASL**
- **71 recursos LSM**
- **142 videos en total**

Los videos correspondientes son enviados a la interfaz y reproducidos consecutivamente respetando el orden de los conceptos generados.

---

# Arquitectura

El proyecto está dividido en siete subsistemas:

| Subsistema | Función |
|---|---|
| **SC-01** | Interfaz web |
| **SC-02** | Control del flujo de traducción |
| **SC-03** | Reconocimiento de señas y construcción textual |
| **SC-04** | Corrección y traducción mediante LLM |
| **SC-05** | Mapeo de conceptos a recursos visuales |
| **SC-06** | Preparación de la secuencia visual |
| **SC-07** | Gestión de recursos y datos temporales |

La interfaz únicamente se comunica con **SC-02**, implementado mediante FastAPI. Este subsistema coordina el resto del flujo de procesamiento.

```text
Usuario
   ↕
SC-01
Interfaz React
   ↕
SC-02
FastAPI
   ↓
SC-03
Reconocimiento
   ↓
SC-04
Corrección y traducción
   ↓
SC-05
Mapeo de recursos
   ↓
SC-06
Salida visual
```

SC-07 participa como soporte para la administración temporal de las sesiones y los recursos utilizados por el sistema.

---

# Tecnologías principales

### Backend

- Python 3.12
- FastAPI
- TensorFlow / Keras
- MediaPipe
- OpenCV
- scikit-learn
- Google Gemini API

### Frontend

- React
- TypeScript
- Vite

---

# Requisitos

Para ejecutar Sign2Sign se requiere:

- **Python 3.12**
- **Node.js 20.18 o posterior**
- npm
- Webcam
- Navegador web moderno
- Conexión a Internet
- Clave válida de Google Gemini API

---

# Archivos de los modelos

Antes de ejecutar el sistema deben encontrarse los siguientes archivos:

```text
models/
│
├── mediapipe/
│   └── hand_landmarker.task
│
├── asl/
│   ├── A3_mlp_img_norm.keras
│   └── scaler_img_norm.joblib
│
└── lsm/
    ├── V2_M2_mlp_world_norm.keras
    └── scaler_world_norm.joblib
```

Estos archivos contienen los clasificadores y escaladores utilizados durante el reconocimiento.

---

# Configuración inicial

## 1. Obtener el proyecto

Si se utiliza Git:

```bash
git clone https://github.com/Yael23211/Sign2Sign.git
cd Sign2Sign
```

También es posible utilizar directamente la carpeta entregada con el proyecto.

---

## 2. Crear el entorno virtual

Desde la raíz de Sign2Sign:

```bash
python -m venv .venv
```

En Windows:

```powershell
.venv\Scripts\activate
```

En Linux o macOS:

```bash
source .venv/bin/activate
```

---

## 3. Instalar las dependencias del backend

Con el entorno virtual activado:

```bash
python -m pip install -r requirements.txt
```

---

## 4. Configurar Gemini API

El proyecto incluye el archivo:

```text
.env.example
```

Crear una copia llamada:

```text
.env
```

En Windows puede utilizarse:

```powershell
copy .env.example .env
```

Dentro de `.env`, configurar:

```text
GEMINI_API_KEY=TU_CLAVE_API
```

También puede especificarse el modelo de Gemini:

```text
GEMINI_MODEL=gemini-3.5-flash-lite
```

**No se debe compartir ni subir al repositorio una clave API real.**

---

# Ejecución de Sign2Sign

Sign2Sign requiere ejecutar el backend y el frontend simultáneamente.

## 1. Iniciar el backend

Desde la carpeta raíz del proyecto y con el entorno virtual activado:

```bash
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --workers 1
```

El backend estará disponible en:

```text
http://127.0.0.1:8000
```

La documentación interactiva de la API puede consultarse en:

```text
http://127.0.0.1:8000/docs
```

---

## 2. Iniciar el frontend

Abrir una **segunda terminal**.

Desde la raíz del proyecto:

```bash
cd frontend
npm ci
npm run dev
```

La interfaz estará disponible en:

```text
http://127.0.0.1:5173
```

Abrir esta dirección en el navegador.

---

# Uso del sistema

Una vez iniciado Sign2Sign:

1. Seleccionar la lengua de señas de origen.
2. Iniciar una sesión.
3. Activar la webcam.
4. Permitir el acceso a la cámara desde el navegador.
5. Realizar una seña del alfabeto manual frente a la cámara.
6. Capturar la seña.
7. Repetir el proceso para formar palabras o frases.
8. Utilizar espacios para separar palabras.
9. Revisar el texto reconocido.
10. Seleccionar **Traducir**.
11. Esperar el procesamiento del texto.
12. Consultar:
   - texto reconocido;
   - texto corregido;
   - traducción;
   - secuencia visual en la lengua destino.
13. Los videos correspondientes se reproducirán desde la propia interfaz.

---

# Atajos de teclado

La interfaz incluye los siguientes atajos:

| Tecla | Acción |
|---|---|
| `Enter` | Capturar una seña |
| `Space` | Insertar espacio |
| `Backspace` | Borrar último elemento |
| `Escape` | Limpiar texto |
| `T` | Traducir |
| `P` | Repetir la secuencia visual |

Los atajos únicamente se ejecutan cuando las condiciones necesarias para cada operación se encuentran disponibles.

---

# Pruebas

## Backend

Desde la raíz del proyecto:

```bash
python -m unittest discover -s tests -v
```

## Frontend

```bash
cd frontend
npm test
```

Para verificar la compilación:

```bash
npm run build
```

---

# Estructura general del proyecto

```text
Sign2Sign/
│
├── data/
│   ├── resources/
│   └── vocabulary/
│
├── docs/
│
├── frontend/
│   ├── public/
│   ├── src/
│   ├── package.json
│   └── vite.config.ts
│
├── models/
│   ├── asl/
│   ├── lsm/
│   └── mediapipe/
│
├── resources/
│   ├── asl/
│   └── lsm/
│
├── scripts/
├── src/
├── tests/
│
├── .env.example
├── requirements.txt
└── README.md
```

---

# Consideraciones y limitaciones

Sign2Sign es un **prototipo académico y demostrativo**.

Su alcance actual contempla principalmente:

- reconocimiento de señas estáticas del alfabeto manual;
- construcción deliberada de texto letra por letra;
- traducción mediante idiomas puente;
- vocabulario visual cerrado;
- ejecución en un entorno local y controlado.

El sistema no pretende representar toda la complejidad gramatical de ASL o LSM ni sustituir la labor de intérpretes profesionales.

Algunas señas con componentes dinámicos son tratadas mediante una representación estática dentro del alcance definido para el prototipo.

La etapa de corrección y traducción requiere conexión a Internet debido al uso de Google Gemini API.

El desempeño del reconocimiento mediante webcam puede variar dependiendo de factores como iluminación, orientación de la mano, fondo y ejecución de la seña.

---

# Trabajo futuro

Entre las posibles extensiones de Sign2Sign se encuentran:

- reconocimiento de señas dinámicas;
- procesamiento de secuencias temporales;
- ampliación del vocabulario visual;
- mejora de la robustez ante diferentes condiciones de captura;
- evaluación con usuarios de las comunidades Sordas;
- reducción de la dependencia de servicios externos;
- despliegue del sistema fuera del entorno local.

---

## Trabajo Terminal

Proyecto desarrollado en la **Escuela Superior de Cómputo del Instituto Politécnico Nacional (ESCOM-IPN)**.

**Sign2Sign: Diseño e implementación de un sistema bilingüe de traducción automática para lenguas de señas (ASL y LSM) mediante técnicas de inteligencia artificial.**