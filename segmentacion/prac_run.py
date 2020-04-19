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
from sklearn.externals import joblib
from time import time, sleep
import sys, os, pickle
import identificar_bifurcacion as bif
import math


class Segmentador(object):


    def __init__(self):
        pass


    def clf_create(self, imgPath, mkImgPath):
        # Creamos el clasificador
        self.clf = NearestCentroid()
        data, lbls = None, None

        for i, j in zip(sorted(os.listdir(imgPath)), sorted(os.listdir(mkImgPath))):
            imNp_aux = imread(i)
            markImg_aux = imread(j)
            self.imNp = cv2.cvtColor(imNp_aux, cv2.COLOR_BGR2RGB)
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

            if data is None or lbls is None:
                data = np.concatenate([data_fondo, data_linea, data_marca])
                lbls = np.concatenate([lbl_fondo, lbl_linea, lbl_marca])
            else:
                data_aux = np.concatenate([data_fondo, data_linea, data_marca])
                data = np.concatenate([data, data_aux])
                lbls_aux = np.concatenate([lbl_fondo, lbl_linea, lbl_marca])
                lbls = np.concatenate([lbls, lbls_aux])

       # Entrenemos el modelo
        self.clf.fit(data, lbls)
        joblib.dump(self.clf, 'clasificador.pkl')


    def clf_load(self):
        self.clf = joblib.load('clasificador.pkl')


    def video_create(self, video):
        # Capturamos el video
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

        # Inicio la captura de imagenes
        capture = cv2.VideoCapture(video)
        count = 0
        ret, self.frame = capture.read()
        filename = 0
        self.centro=()


        # Clasificamos el video
        while ret:

            if count%25 == 0:
                height, width = self.frame.shape[:2]
                self.frame = self.frame[70:height, 0:width]
                cv2.imwrite('images/image%03d.png' % filename, cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR))
                imNp = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                imrgbn = np.rollaxis((np.rollaxis(imNp, 2) + 0.0)/np.sum(imNp, 2), 0, 3)[:,:,:2]
                predicted_image = self.clf.predict(np.reshape(imNp, (imNp.shape[0]*imNp.shape[1], imNp.shape[2]))) # Creamos la prediccion y redimensionamos

                self.predImg = np.reshape(predicted_image, (imNp.shape[0], imNp.shape[1])) # Recuperamos las dimensiones
                paleta = np.array([[0,0,255], [255,0,0], [0,255,0]], dtype=np.uint8)

                self.__line_identification()
                self.__arrow_direction()

                #sleep(1)
                cv2.waitKey(1)
                cv2.imshow("Segmentacion Euclid", cv2.cvtColor(paleta[self.predImg], cv2.COLOR_RGB2BGR))

                cv2.imwrite('images/image%03d.png' % filename, cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR))
                filename += 1

                out.write(cv2.cvtColor(paleta[self.predImg], cv2.COLOR_RGB2BGR))

            count += 1
            ret, self.frame = capture.read()

        capture.release()
        out.release()
        cv2.destroyAllWindows()


    def __line_identification(self):
        camino = []
        salidas, self.centro = bif.existen_bifurcaciones(self.frame, self.predImg, self.centro)
        cv2.circle(self.frame, self.centro, 3, [0,0,255], -1)

        linImg = (self.predImg==1).astype(np.uint8)*255
        ret, thresh = cv2.threshold(linImg,50,255,0)
        _, contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for cont in contours:#identificamos el contorno que determina nuestro camino
           if cv2.pointPolygonTest(cont, self.centro, False)== 0.0:
              camino=cont


        for point in salidas:
              cv2.circle(self.frame, point, 3, [255,0,0], -1)
        defects=[]
        if len(camino)>0:
           cv2.drawContours(self.frame, camino, -1, (0, 255, 0), 1)
           #hull = cv2.convexHull(camino)
           #cv2.drawContours(self.frame, [hull], -1, (255, 0, 0), 2)
           hull = cv2.convexHull(camino,returnPoints=False)
           defects = cv2.convexityDefects(camino,hull)
        if len(salidas)>1:
            cv2.putText(self.frame,'Cruce de {0} salidas'.format(len(salidas)), (15,20), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0))
            return -1
        elif len(salidas)==1:
           for i in range(defects.shape[0]):
              start,end,f,d = defects[i,0]
              start = tuple(camino[start][0])
              end = tuple(camino[end][0])
              if i==0 or d>dis:
                 far = tuple(camino[f][0])
                 dis = d
                 sdef = start
                 edef=end
           #cv2.line(self.frame,sdef,edef,[0,0,255],2)
           #far = tuple(camino[far][0])
           #cv2.circle(self.frame, tuple(far), 3, [255,0,0], -1)
           if dis>2000 and ((salidas[0][0]-self.centro[0])*(far[1]-self.centro[1])-(salidas[0][1]-self.centro[1])*(far[0] -self.centro[0]))>0:
              cv2.putText(self.frame,'Curva hacia la izquierda', (15,20), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0))
              vel = (170-salidas[0][1])*100
              cv2.putText(self.frame,'Velocidad = {0} %'.format(vel/170), (15,40), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0))
              giro = (abs(160-salidas[0][0]))*100
              cv2.putText(self.frame,'Velocidad de giro = {0} %'.format(giro/160), (15,60), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0))
           elif dis>2000 and ((salidas[0][0]-self.centro[0])*(far[1]-self.centro[1])-(salidas[0][1]-self.centro[1])*(far[0] -self.centro[0]))<0:
              cv2.putText(self.frame,'Curva hacia la derecha', (15,20), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0))
              vel = (170-salidas[0][1])*100
              cv2.putText(self.frame,'Velocidad = {0} %'.format(vel/170), (15,40), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0))
              giro = (abs(160-salidas[0][0]))*100
              cv2.putText(self.frame,'Velocidad de giro = {0} %'.format(giro/160), (15,60), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0))
           else:
              cv2.putText(self.frame,'Recta', (15,20), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0))
              vel = (170-salidas[0][1])*100
              cv2.putText(self.frame,'Velocidad = {0} %'.format(vel/170), (15,40), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0))
              giro = (abs(160-salidas[0][0]))*100
              cv2.putText(self.frame,'Velocidad de giro = {0} %'.format(giro/160), (15,60), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0))
        #elif (len(contours) > 0):
            #moments = cv2.moments(max(contours, key=cv2.contourArea))
            #x = int(moments['m10']/moments['m00'])
            #if x >= 120:
                #cv2.putText(self.frame,'Curva hacia la derecha', (15,20), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0))
            #    return -1.0
            #elif 120 > x and x > 50:
                #cv2.putText(self.frame,'Recta', (15,20), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0))
            #    return 0
            #elif 50 >= x:
                #cv2.putText(self.frame,'Curva hacia la izquierda', (15,20), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0))
            #    return 1.0
        #cv2.waitKey(177) # Comentarlo para mejorar tiempos
        #cv2.imshow("contorno", self.frame)
        return 0


    def __arrow_direction(self): # Esta por terminar
        linImg = (self.predImg==2).astype(np.uint8)*255
        _, contours, _ = cv2.findContours(linImg, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        if len(contours) > 0:
            contours = max(contours, key=lambda x : len(x)) # Si encuentra mas de un contorno
            elipse = cv2.fitEllipse(contours)
            #cv2.ellipse(self.frame, elipse, (0,255,0))
            caja = np.int0(cv2.boxPoints(elipse))
            #cv2.drawContours(self.frame, [caja], -1, (255,0,0), 3)
            # Calculamos los puntos medios del rectangulo
            pm1 = (caja[3] + caja[0])/2
            pm2 = (caja[2] + caja[1])/2
            cv2.circle(self.frame, tuple(pm1), 2, (0,255,0), -1)
            #cv2.circle(self.frame, tuple(pm2), 2, (0,255,0), -1)
            height, width = self.frame.shape[:2]
            vec = np.array(pm2-pm1)
            p1 = (0, pm1[1] + ((0-pm1[0])*vec[1])/(vec[0]))
            p2 = (width-1, pm2[1] + ((width-1-pm2[0])*vec[1])/(vec[0]))
            #cv2.line(self.frame, p1, p2, (0,0,255), 2)
            hull = cv2.convexHull(contours)
            cv2.drawContours(self.frame, hull, -1, (0,0,255))


if __name__ == "__main__":
    start = time()
    seg = Segmentador()
    print("Tiempo al crear el segmentador: {}".format(time() - start))
    if os.path.isfile('clasificador.pkl'):
        seg.clf_load()
    else:
        seg.clf_create(sys.argv[1], sys.argv[2])
    print("Tiempo del clasificador: {}".format(time() - start))
    seg.video_create(sys.argv[3 if len(sys.argv) > 2 else 1])
    print("Tiempo total: {}".format(time() - start))


