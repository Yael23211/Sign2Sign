\# Limpieza y preparación de datasets



\*\*Proyecto:\*\* Sign2Sign  

\*\*Periodo de referencia:\*\* 18–24 de agosto de 2026  

\*\*Fecha de cierre:\*\* 20 de agosto de 2026



\## 1. Objetivo



Auditar, depurar y documentar los datasets de imágenes utilizados para

el reconocimiento de ASL y LSM, manteniendo trazabilidad sobre cada

muestra antes de realizar la conversión de imágenes a landmarks.



\## 2. Dataset ASL



Fuente original:



`D:\\Sign2SignData\\raw\\asl\\ASL\_original.zip`



El conjunto contiene:



\- 26 letras.

\- 900 imágenes Train por letra.

\- 100 imágenes Test por letra.

\- 26,000 imágenes correspondientes a letras.

\- 1,000 imágenes de la clase Blank.

\- 27,000 imágenes en total.



\### Integridad



Se validaron las 27,000 imágenes.



\- Válidas: 27,000.

\- Inválidas: 0.

\- Integridad: 100 %.



\### Duplicados



No se detectaron duplicados exactos.



\### Clase Blank



Blank no será utilizada como clase del clasificador alfabético.



Se conserva como conjunto negativo para evaluar posteriormente el

comportamiento de MediaPipe cuando no existe una mano en la imagen.



\### Diversidad



Se realizó una auditoría visual sobre diferentes clases y sobre los

conjuntos Train y Test.



Se observó diversidad respecto a:



\- fondo;

\- iluminación;

\- tono de piel;

\- perspectiva;

\- posición;

\- distancia respecto de la cámara.



No se observó redundancia temporal sistemática similar a la presente

en el dataset LSM.



La separación Train/Test original se conserva mediante el campo

`split\_original`.



El split definitivo será establecido posteriormente durante la

generación del dataset de landmarks.



\## 3. Dataset LSM



Fuente original:



`D:\\Sign2SignData\\raw\\lsm\\LSM\_original.zip`



El conjunto contiene inicialmente:



\- 89,683 imágenes.

\- 23 clases disponibles.

\- aproximadamente 3,900 imágenes por clase.



Las clases faltantes originalmente son:



\- E

\- L

\- Ñ

\- Z



\### Integridad



Se validaron las 89,683 imágenes.



\- Válidas: 89,683.

\- Inválidas: 0.

\- Integridad: 100 %.



\### Duplicados exactos



Se identificaron:



\- 183 grupos de duplicados.

\- 366 archivos involucrados.

\- 183 imágenes redundantes.



Todos los duplicados pertenecen a la misma clase.



No se encontraron imágenes idénticas etiquetadas como clases

diferentes.



Las muestras redundantes no fueron eliminadas físicamente del ZIP.



Se registraron como:



`status = rejected`



`reject\_reason = exact\_duplicate`



Como resultado:



\- Imágenes originales: 89,683.

\- Imágenes aceptadas: 89,500.

\- Imágenes rechazadas: 183.



\## 4. Redundancia y sesiones LSM



La inspección visual mostró alta similitud entre imágenes pertenecientes

a una misma sesión o secuencia de captura.



Al mismo tiempo, existen diferencias importantes entre sesiones,

incluyendo variaciones de:



\- persona;

\- fondo;

\- iluminación;

\- orientación;

\- posición de la mano.



Los nombres de archivo permiten identificar parcialmente bloques o

sesiones de adquisición.



Por esta razón no se realizará un split aleatorio por imagen.



Cuando se generen Train, Validation y Test se utilizará la información

de grupos/fuentes para reducir el riesgo de fuga de información entre

conjuntos.



\## 5. Letras LSM faltantes



\### E



Se utilizarán provisionalmente las 1,000 imágenes ASL-E.



La configuración manual es similar pero no completamente idéntica a

LSM-E.



Estas muestras quedan marcadas como:



`source\_language = ASL`



`provisional\_source = 1`



La clase deberá ser evaluada especialmente con capturas LSM reales.



\### L



Se utilizarán provisionalmente las 1,000 imágenes ASL-L debido a la

alta similitud de configuración manual.



\### Z



Se utilizarán provisionalmente las 1,000 imágenes ASL-Z.



Debido a que Z involucra movimiento, el prototipo utilizará una postura

estática representativa, manteniendo esta condición como limitación.



\### Ñ



No se creará artificialmente una clase Ñ mediante duplicación de las

imágenes de N.



Dado que la diferencia N/Ñ depende del movimiento y el prototipo utiliza

una captura estática, entrenar ambas clases con geometría prácticamente

idéntica introduciría etiquetas contradictorias.



La primera versión del clasificador no incluirá Ñ como clase entrenada.



La ambigüedad N/Ñ será manejada posteriormente mediante el vocabulario

cerrado y el módulo de corrección contextual.



\## 6. Dataset LSM provisional



Se creó:



`lsm\_manifest\_provisional.csv`



Este contiene:



\- 89,683 registros del dataset LSM original para mantener trazabilidad.

\- 183 muestras LSM rechazadas por duplicación.

\- 89,500 muestras LSM aceptadas.

\- 1,000 muestras provisionales ASL-E.

\- 1,000 muestras provisionales ASL-L.

\- 1,000 muestras provisionales ASL-Z.



Total de filas:



92,683.



Total de muestras candidatas aceptadas:



92,500.



\## 7. Balance



Actualmente existe desbalance entre las clases LSM originales

(\~3,900 imágenes) y E, L y Z (1,000 imágenes provisionales).



No se realizará balanceo en esta etapa.



Primero se realizará la extracción de landmarks y se determinará

cuántas muestras válidas quedan realmente disponibles por clase.



Posteriormente se evaluarán estrategias como:



\- submuestreo;

\- data augmentation únicamente sobre Train;

\- pesos de clase;

\- incorporación de capturas LSM reales.



\## 8. Capturas mediante webcam



Las capturas reales mediante webcam podrán utilizarse posteriormente

para evaluar la generalización del modelo.



Se plantea mantener inicialmente un conjunto independiente

`webcam\_benchmark` que no participe en el entrenamiento.



Si se detecta diferencia importante entre el rendimiento del dataset y

el rendimiento real, podrá crearse posteriormente un conjunto

`webcam\_adaptation`.



Las capturas propias serán especialmente útiles para clases

provisionales como E.



\## 9. Manifests generados



Los datasets originales permanecen intactos.



La selección y estado de las muestras se controla mediante manifests:



\- `asl\_manifest.csv`

\- `lsm\_manifest.csv`

\- `lsm\_manifest\_clean.csv`

\- `lsm\_manifest\_provisional.csv`



Estos archivos permitirán mantener trazabilidad durante la generación

de landmarks, los splits y el entrenamiento.



\## 10. Resultado



Los datasets de imágenes han sido inventariados, validados y depurados.



ASL no presenta archivos corruptos ni duplicados exactos.



LSM no presenta archivos corruptos y sus 183 duplicados exactos han

sido identificados y excluidos lógicamente mediante el manifest.



Las clases LSM E, L y Z disponen de muestras ASL provisionales

identificadas explícitamente por procedencia.



Ñ permanece como limitación del reconocimiento estático y será tratada

mediante corrección contextual.



Con estos resultados, los datasets se consideran preparados para

iniciar la etapa de conversión:



\*\*Imagen → landmarks.\*\*

