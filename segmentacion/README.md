# Segmentacion

Este directorio contiene código para ayudaros a segmentar una imagen.

## Archivos

- select_pixels.py    Funciones para pintar encima de una imagen

- pinta_colores.py    Ejemplo para mostrar la distribución de color de los píxeles marcados

- etiqueta_imagenes   Esqueleto de programa para sacar los datos de entrenamiento de el algoritmo de segmentación.

- pract_run	    Esqueleto de programa para ejecutar el algoritmo de segmentación.

Este programa primero entrena el clasificador con los datos de entrenamiento y luego segmenta el vídeo.

## Ejecucion
Para ejecutar usar:
./prac_run.py imagen_original imagen_marcada video_a_clasificar
Por ejemplo:
./prac_run.py ~/roboticaUPM/rsc/imgs/ln1.png ~/roboticaUPM/rsc/imgs/lnMark1.png ~/roboticaUPM/rsc/line0.avi
