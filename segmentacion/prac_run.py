#!/usr/bin/env python2.7

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
from time import time, sleep
import sys


class Segmentador():

    def __init__(self):
        pass


    def clf_create(self, img, mkImg):
        # Creamos el clasificador
        self.clf = NearestCentroid()

        # Leo las imagenes de entrenamiento
        imNp_aux = imread(img, mode='RGB')
        self.imNp = cv2.cvtColor(imNp_aux, cv2.COLOR_BGR2RGB)
        markImg_aux = imread(mkImg, mode='RGB')
        self.markImg = cv2.cvtColor(markImg_aux, cv2.COLOR_BGR2RGB)


        # Preparo los datos de entrenamiento
        # saco todos los puntos marcados en rojo/verde/azul
        data_marca = self.imNp[np.all(self.markImg == [255,0,0], 2)]
        data_fondo = self.imNp[np.all(self.markImg == [0,255,0], 2)]
        data_linea = self.imNp[np.all(self.markImg == [0,0,255], 2)]

        # Preparamos las etiquetas
        lbl_fondo = np.zeros(data_fondo.shape[0], dtype=np.uint8)
        lbl_linea = np.ones(data_linea.shape[0], dtype=np.uint8)
        lbl_marca = np.zeros(data_marca.shape[0], dtype=np.uint8) + 2

        # Entrenemos el modelo
        self.clf.fit(np.concatenate([data_fondo, data_linea, data_marca]), np.concatenate([lbl_fondo, lbl_linea, lbl_marca]))


    def video_create(self, video):
        # Capturamos el video
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

        # Inicio la captura de imagenes
        capture = cv2.VideoCapture(video)
        count = 0
        ret, self.frame = capture.read()
        filename = 0

        # Clasificamos el video
        while ret:

            if count%25 == 0:
                count += 1
                imNp = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                imrgbn = np.rollaxis((np.rollaxis(imNp, 2) + 0.0)/np.sum(imNp, 2), 0, 3)[:,:,:2]
                predicted_image = self.clf.predict(np.reshape(imNp, (imNp.shape[0]*imNp.shape[1], imNp.shape[2]))) # Creamos la prediccion y redimensionamos

                self.predImg = np.reshape(predicted_image, (imNp.shape[0], imNp.shape[1])) # Recuperamos las dimensiones
                paleta = np.array([[0,0,255], [255,0,0], [0,255,0]], dtype=np.uint8)

                self.__line_identification()

                cv2.imshow("Segmentacion Euclid", cv2.cvtColor(paleta[self.predImg], cv2.COLOR_RGB2BGR))

                cv2.imwrite('images/image%03d.png' % filename, cv2.cvtColor(paleta[self.predImg], cv2.COLOR_RGB2BGR))
                filename += 1

                out.write(cv2.cvtColor(paleta[self.predImg], cv2.COLOR_RGB2BGR))
            else:
                count += 1

            ret, self.frame = capture.read()

        capture.release()
        out.release()
        cv2.destroyAllWindows()


    def __line_identification(self):
        cruce = []
        linImg = (self.predImg==1).astype(np.uint8)*255
        contours, _ = cv2.findContours(linImg, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(self.frame, contours, -1, (0, 255, 0), 3)
        for cont in contours:
            if len(cont) > 100: cruce.append(cont)
            for point in cont:
                height, width = self.frame.shape[:2]
                if (point[0][0]==0) or (point[0][0]==width-1) or (point[0][1]==0) or (point[0][1]==height-1):
                    cv2.circle(self.frame, tuple(point[0]), 3, [0,0,255], -1)
        if len(cruce) > 2:
            cv2.putText(self.frame,'Cruece de {0} salidas'.format(len(cruce)-1), (15,20), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0))
        else:
            moments = cv2.moments(max(contours, key=cv2.contourArea))
            x = int(moments['m10']/moments['m00'])
            if x >= 120:
                cv2.putText(self.frame,'Curva hacia la derecha', (15,20), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0))
                return -1.0
            elif 120 > x and x > 50:
                cv2.putText(self.frame,'Recta', (15,20), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0))
                return 0
            elif 50 >= x:
                cv2.putText(self.frame,'Curva hacia la izquierda', (15,20), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0))
                return 1.0

        cv2.waitKey(177) # Comentarlo para mejorar tiempos
        cv2.imshow("contorno", self.frame)


    def __arrow_direction(self): # Esta por terminar
        linImg = (self.predImg==2).astype(np.uint8)*255
        contours, _ = cv2.findContours(linImg, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(self.frame, contours, -1, (0, 255, 0), 3)
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 20)
        left, right = [0, 0], [0, 0]
        for i in contours:
            th, rh = i[0][1], i[0][0]
            if (np.round(th, 2) >= 1.0 and np.round(th, 2) <= 1.1) or (np.round(th, 2) >= 2.0 and np.round(th, 2) <= 2.1):
                if rh >= 20 and rh <= 30:
                    left[0] += 1
                elif rh >= 60 and rh <= 65:
                    left[1] += 1
                elif rh >= -73 and rh <= -57:
                    right[0] += 1
                elif rh >= 148 and rh <= 176:
                    right[1] += 1
        if left[0] >= 1 and left[1] >= 1:
            cv2.putText(self.frame, 'Flecha izquierda', (15,20), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0))
            return 1.0
        elif right[1] >= 1 and right[0] >= 1:
            cv2.putText(self.frame, "Flecha derecha", (15,20), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0))
            return -1.0
        return 0


if __name__ == "__main__":
    start = time()
    seg = Segmentador()
    print("Tiempo al crear el segmentador: {}".format(time() - start))
    seg.clf_create(sys.argv[1], sys.argv[2])
    print("Tiempo del clasificador: {}".format(time() - start))
    seg.video_create(sys.argv[3])
    print("Tiempo total: {}".format(time() - start))

