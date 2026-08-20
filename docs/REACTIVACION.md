# **Cierre de Reactivación del Proyecto Sign2Sign**



**Periodo planificado:** 15–17 de agosto de 2026

**Fecha de cierre:** 18 de agosto de 2026

**Proyecto:** Sign2Sign – TT 2026-B171



\## 1. Objetivo de la reactivación



La etapa de reactivación tuvo como propósito establecer nuevamente las condiciones necesarias para continuar con la construcción de Sign2Sign, definiendo una estructura reproducible para el código, los datos y los experimentos del proyecto.



Debido a que no existía previamente un repositorio formal del sistema, se creó una nueva estructura de proyecto y se incorporaron los experimentos anteriores únicamente como antecedentes de desarrollo.



\## 2. Estructura del proyecto



El código fuente y los archivos versionables del proyecto se almacenan en el repositorio local:



`C:\Users\yaelb\Sign2Sign`



Los datasets y productos derivados de gran tamaño se almacenan externamente en:



`D:\\Sign2SignData`



La separación entre código y datos evita incorporar los datasets al repositorio Git y permite mantener los archivos originales sin modificaciones.



La estructura externa de datos definida es:



```text

D:\\Sign2SignData\\

├── raw\\

│   ├── asl\\

│   └── lsm\\

├── processed\\

│   └── landmarks\\

│       ├── asl\\

│       └── lsm\\

└── inventory\\

```



\## 3. Control de versiones



Se inicializó un repositorio Git para Sign2Sign.



La rama principal del proyecto es:



`main`



Se configuró un archivo `.gitignore` para excluir, entre otros elementos:



\* datasets;

\* datos procesados;

\* modelos entrenados;

\* entornos virtuales;

\* archivos temporales;

\* variables de entorno;

\* dependencias de frontend.



\## 4. Entorno local



Se utiliza:



\*\*Python 3.12.9\*\*



Se creó un entorno virtual local:



`.venv`



Las dependencias iniciales se encuentran registradas en:



`requirements.txt`



El entorno local está orientado principalmente a:



\* procesamiento de imágenes;

\* extracción de landmarks;

\* procesamiento y análisis de datos;

\* evaluación;

\* inferencia;

\* pruebas con webcam.



Por el momento no se instalarán TensorFlow ni PyTorch localmente.



En caso de requerirse entrenamiento mediante redes neuronales, se contempla utilizar Google Colab para aprovechar recursos de GPU. La instalación local de frameworks de Deep Learning se evaluará posteriormente dependiendo del modelo finalmente seleccionado y de los requisitos de inferencia.



\## 5. Dataset ASL



\*\*Fuente:\*\* Synthetic ASL Alphabet de Lexset, obtenido mediante Kaggle.



Archivo original:



`D:\\Sign2SignData\\raw\\asl\\ASL\_original.zip`



El archivo se conserva sin modificaciones.



\### Inventario



El dataset contiene:



\* 26 letras del alfabeto ASL;

\* 900 imágenes de entrenamiento por letra;

\* 100 imágenes de prueba por letra;

\* 26,000 imágenes correspondientes a letras;

\* 1,000 imágenes adicionales pertenecientes a la clase `Blank`;

\* 27,000 imágenes en total;

\* un archivo auxiliar `alphabet.jpg`.



Distribución:



\* Train: 23,400 imágenes de letras.

\* Test: 2,600 imágenes de letras.

\* Blank Train: 900 imágenes.

\* Blank Test: 100 imágenes.



La clase `Blank` no será utilizada como una clase del clasificador alfabético. Se conservará como conjunto negativo potencial para evaluar el comportamiento del módulo de detección de mano.



\### Auditoría visual preliminar



Se revisaron muestras de las clases A, M y Z en los conjuntos Train y Test.



Se observó una diversidad visual adecuada respecto a:



\* fondos;

\* iluminación;

\* tonos de piel;

\* distancia respecto a la cámara;

\* orientación;

\* ubicación de la mano en la imagen.



No se identificó visualmente una redundancia temporal sistemática comparable con la encontrada en el dataset LSM.



Se conservará inicialmente la separación Train/Test proporcionada por la fuente.



\## 6. Dataset LSM



Archivo original:



`D:\\Sign2SignData\\raw\\lsm\\LSM\_original.zip`



El archivo original se conserva comprimido y sin modificaciones.



\### Inventario



El conjunto contiene:



\* 89,710 archivos;

\* 89,683 imágenes;

\* aproximadamente 22.14 GB descomprimidos;

\* 23 clases disponibles.



Las clases faltantes son:



\* E

\* L

\* Ñ

\* Z



Las clases disponibles contienen aproximadamente 3,900 imágenes cada una, por lo que existe un buen balance cuantitativo entre ellas.



\### Redundancia detectada



La inspección visual de la clase N permitió identificar dos características:



1\. Existe diversidad entre distintas sesiones de captura, incluyendo variaciones de mano, iluminación, fondo, posición y condiciones de adquisición.

2\. Existe una elevada similitud entre imágenes pertenecientes a una misma sesión o secuencia.



Se identificaron nombres de archivos que parecen agrupar distintas sesiones, por ejemplo:



`N1`, `N3`, `N5`, `N6`.



Por este motivo no se utilizará una división aleatoria simple por imagen para generar los conjuntos Train, Validation y Test.



Se estudiará la posibilidad de realizar la separación considerando sesiones o grupos de captura para evitar que imágenes prácticamente idénticas aparezcan simultáneamente en entrenamiento y evaluación.



\## 7. Data augmentation



Se contempla utilizar aumento de datos para mejorar la robustez de los modelos ante condiciones reales de captura.



Las posibles transformaciones a evaluar incluyen:



\* pequeñas modificaciones de brillo;

\* contraste;

\* pequeñas rotaciones;

\* ligeras variaciones de escala;

\* pequeñas traslaciones.



Los parámetros definitivos todavía no se establecen.



El aumento de datos se aplicará únicamente a las muestras destinadas al conjunto de entrenamiento.



Validation y Test conservarán muestras sin aumento artificial.



Antes de aplicar las transformaciones se verificará que estas no modifiquen de forma significativa la configuración manual correspondiente a cada letra.



\## 8. Tratamiento de letras dinámicas



Las letras que requieren movimiento serán representadas mediante una postura estática representativa.



Esta decisión constituye una limitación explícita del prototipo.



Los posibles errores provocados por la similitud entre posturas estáticas serán analizados posteriormente mediante las métricas del clasificador y su matriz de confusión.



\## 9. Modelos de reconocimiento



Se mantienen las siguientes decisiones:



\* habrá un modelo independiente para ASL;

\* habrá un modelo independiente para LSM;

\* las imágenes no serán utilizadas directamente como entrada final del clasificador;

\* MediaPipe será utilizado para obtener landmarks de la mano;

\* los landmarks serán normalizados antes de utilizarse como características;

\* se establecerá experimentalmente un umbral de confianza;

\* una predicción que no supere dicho umbral será rechazada.



\## 10. Entrenamientos anteriores



Dentro del material LSM se localizaron experimentos previos utilizando:



\* MobileNetV2;

\* ResNet50;

\* EfficientNetV2-M;

\* TensorFlow;

\* PyTorch.



Estos experimentos se consideran únicamente antecedentes.



Se identificó en algunos scripts una división aleatoria por imagen. Debido a la redundancia observada entre frames del conjunto LSM, dicha estrategia podría producir fuga de información entre entrenamiento y evaluación.



También se identificó un posible problema en uno de los scripts de PyTorch relacionado con la asignación de transformaciones a subconjuntos que comparten una misma instancia de `ImageFolder`.



Estos experimentos no serán reutilizados directamente como solución final.



\## 11. Recursos visuales ASL y LSM



Actualmente no se dispone localmente del repositorio final de videos utilizado para representar las señas de salida.



Existen fuentes previamente sugeridas por los directores del Trabajo Terminal.



La obtención y procesamiento de estos recursos se realizará posteriormente y no se considera actualmente un bloqueo para el desarrollo de los módulos de reconocimiento.



\## 12. Backlog inmediato



Las siguientes actividades corresponden a la etapa de limpieza y preparación de datos:



1\. Analizar los grupos o sesiones existentes en cada clase del dataset LSM.

2\. Cuantificar la redundancia intrasesión.

3\. Definir una estrategia de selección o submuestreo para LSM.

4\. Definir una estrategia de particionado Train/Validation/Test evitando fuga entre sesiones.

5\. Documentar el tratamiento de las clases LSM faltantes E, L, Ñ y Z.

6\. Verificar archivos corruptos o inválidos en ambos datasets.

7\. Evaluar posteriormente la tasa de detección de mano mediante MediaPipe.

8\. Definir las transformaciones de data augmentation utilizadas únicamente en entrenamiento.

9\. Preparar los datasets para la posterior conversión de imágenes a landmarks.



\## 13. Estado de la reactivación



Con la creación del repositorio, organización del almacenamiento, definición del entorno local, inventario inicial de los datasets ASL y LSM y registro de las decisiones técnicas, se considera concluida la etapa de Reactivación.



El proyecto continúa con la etapa:



\*\*Limpieza y preparación de datos — 18 al 24 de agosto de 2026.\*\*



