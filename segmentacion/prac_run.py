####################################################
# Esqueleto de programa para ejecutar el algoritmo de segmentacion.
# Este programa primero entrena el clasificador con los datos de
#  entrenamiento y luego segmenta el video (este entrenamiento podria
#  hacerse en "prac_ent.py" y aqui recuperar los parametros del clasificador
###################################################


import cv2
from scipy.misc import imread, imsave
from matplotlib import pyplot as plt
import numpy as np
from sklearn.neighbors import NearestCentroid
from time import time

# Medimos el tiempo
start = time()

# Creamos el clasificador
clf = NearestCentroid()

# Leo las imagenes de entrenamiento
imNp_aux = imread('../rsc/imgs/ln1.png', mode='RGB')
imNp = cv2.cvtColor(imNp_aux, cv2.COLOR_BGR2RGB)
markImg_aux = imread('../rsc/imgs/lnMark1.png', mode='RGB')
markImg = cv2.cvtColor(markImg_aux, cv2.COLOR_BGR2RGB)


# Preparo los datos de entrenamiento
# saco todos los puntos marcados en rojo/verde/azul
data_marca = imNp[np.all(markImg == [255,0,0], 2)]
data_fondo = imNp[np.all(markImg == [0,255,0], 2)]
data_linea = imNp[np.all(markImg == [0,0,255], 2)]

# Preparamos las etiquetas
lbl_fondo = np.zeros(data_fondo.shape[0], dtype=np.uint8)
lbl_linea = np.ones(data_linea.shape[0], dtype=np.uint8)
lbl_marca = np.zeros(data_marca.shape[0], dtype=np.uint8) + 2

# Entrenemos el modelo
clf.fit(np.concatenate([data_fondo, data_linea, data_marca]), np.concatenate([lbl_fondo, lbl_linea, lbl_marca]))

print("Tiempo entrenamiento:") # Ojala f-strings
print(time() - start)

# Capturamos el video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

# Inicio la captura de imagenes
capture = cv2.VideoCapture("line0.mp4")
count = 0
ret, frame = capture.read()
filename = 0

# Clasificamos el video
while ret:

    if count%25 == 0:
       count += 1
       imNp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
       imrgbn = np.rollaxis((np.rollaxis(imNp, 2) + 0.0)/np.sum(imNp, 2), 0, 3)[:,:,:2]
       predicted_image = clf.predict(np.reshape(imNp, (imNp.shape[0]*imNp.shape[1], imNp.shape[2]))) # Creamos la prediccion y redimensionamos

       predicted_image = np.reshape(predicted_image, (imNp.shape[0], imNp.shape[1])) # Recuperamos las dimensiones
       paleta = np.array([[0,0,255], [255,0,0], [0,255,0]], dtype=np.uint8)

       cv2.imshow("Segmentacion Euclid", cv2.cvtColor(paleta[predicted_image], cv2.COLOR_RGB2BGR))

       cv2.imwrite('images/image%03d.png' % filename ,cv2.cvtColor(paleta[predicted_image], cv2.COLOR_RGB2BGR))
       filename += 1

       out.write(cv2.cvtColor(paleta[palvideo], cv2.COLOR_RGB2BGR))
    else:
       count += 1

    ret, frame = capture.read()

capture.release()
out.release()
cv2.destroyAllWindows()

print("Tiempo total:")
print(time()-start)
