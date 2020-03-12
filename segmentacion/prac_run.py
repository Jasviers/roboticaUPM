####################################################
# Esqueleto de programa para ejecutar el algoritmo de segmentacion.
# Este programa primero entrena el clasificador con los datos de
#  entrenamiento y luego segmenta el video (este entrenamiento podria
#  hacerse en "prac_ent.py" y aqui recuperar los parametros del clasificador
###################################################


import cv, cv2
from scipy.misc import imread, imsave
from matplotlib import pyplot as plt
import numpy as np

#import clasif as cl


# Leo las imagenes de entrenamiento
imNp = imread('linea.png')
markImg = imread('lineaMarcada.png')

# Preparo los datos de entrenamiento
# saco todos los puntos marcados en rojo/verde/azul
data_marca=imNp[np.all(markImg == [255,0,0],2)]
data_fondo=imNp[np.all(markImg == [0,255,0],2)]
data_linea=imNp[np.all(markImg == [0,0,255],2)]


# Creo y entreno los segmentadores euclideos
segmEuc = seg.segEuclid([data_fondo, data_linea, data_marca])
segmMano = seg.segMano2()



# Inicio la captura de imagenes
capture = cv2.VideoCapture(...)

# Ahora clasifico el video
while ():
    # voy a segmentar solo una de cada 25 imagenes y la muestro
    ........
    cv2.imshow("Imagen",img)

    # La pongo en formato numpy

    # Segmento la imagen.
    # Compute rgb normalization
    imrgbn=np.rollaxis((np.rollaxis(imNp,2)+0.0)/np.sum(imNp,2),0,3)[:,:,:2]
    
    labelsEu=segmEuc.segmenta(imNp)
    labelsMa=segmMano.segmenta(imNp)


    # Vuelvo a pintar la imagen
    # genero la paleta de colores
    paleta = np.array([[0,0,0],[0,0,255],[255,0,0],[0,255,0]],dtype=np.uint8)
    # ahora pinto la imagen
    cv2.imshow("Segmentacion Euclid",cv2.cvtColor(paleta[labelsEu],cv2.COLOR_RGB2BGR))
    cv2.imshow("Segmentacion Mano",cv2.cvtColor(paleta[labelsMa],cv2.COLOR_RGB2BGR))

    # Para pintar texto en una imagen
    cv2.putText(imDraw,'Lineas: {0}'.format(len(convDefsLarge)),(15,20),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0))
    # Para pintar un circulo en el centro de la imagen
    cv2.circle(imDraw, (imDraw.shape[1]/2,imDraw.shape[0]/2), 2, (0,255,0), -1)

    # Guardo esta imagen para luego con todas ellas generar un video
    cv2.imwrite

