#!/usr/bin/env python2.7

import cv2
from scipy.misc import imread
import numpy as np
from sklearn.neighbors import NearestCentroid
from sklearn.externals import joblib
from time import time
import sys, os
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
            self.imNp = cv2.cvtColor(imread(imgPath+i), cv2.COLOR_BGR2RGB)
            self.markImg = cv2.cvtColor(imread(mkImgPath+j), cv2.COLOR_BGR2RGB)

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
        joblib.dump(self.clf, '../clasificadores/segmentacion.pkl')


    def clf_load(self):
        self.clf = joblib.load('../clasificadores/segmentacion.pkl')


    def video_create(self, video):
        # Capturamos el video
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480))

        # Inicio la captura de imagenes
        capture = cv2.VideoCapture(video)
        count, filename = 0, 0
        ret, self.frame = capture.read()
        self.centro=()
        self.auxiliar=()


        # Clasificamos el video
        while ret:

            if count%3 == 0:
                self.height, self.width = self.frame.shape[:2]
                self.frame = self.frame[70:self.height, 0:self.width]
                cv2.imwrite('../rsc/generado/image%03d.png' % filename, cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR))
                imNp = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                predicted_image = self.clf.predict(np.reshape(imNp, (imNp.shape[0]*imNp.shape[1], imNp.shape[2]))) # Creamos la prediccion y redimensionamos

                self.predImg = np.reshape(predicted_image, (imNp.shape[0], imNp.shape[1])) # Recuperamos las dimensiones
                paleta = np.array([(0,0,255), (255,0,0), (0,255,0)], dtype=np.uint8)

                self.__line_identification()

                cv2.waitKey(1)
                cv2.imshow("Segmentacion Euclid", cv2.cvtColor(paleta[self.predImg], cv2.COLOR_RGB2BGR))

                cv2.imwrite('../rsc/generado/image%03d.png' % filename, cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR))
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
           hull = cv2.convexHull(camino,returnPoints=False)
           defects = cv2.convexityDefects(camino,hull)
        if len(salidas)>1:
            ret = self.__arrow_direction()
            if not ret:
               cv2.putText(self.frame,'Cruce de {0} salidas'.format(len(salidas)), (15,20), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0))
               tv = round((160-self.centro[0])/160.0,2)
               fv = 0.5
               return tv,fv
            else:
               salida=salidas[0]
               dist=self.distancia(salida,self.auxiliar)
               for i in range(0,len(salidas)):
                  if self.distancia(salidas[i],self.auxiliar)<dist:
                     salida=salidas[i]
                     dist=self.distancia(salidas[i],self.auxiliar)
               cv2.circle(self.frame, salida, 3, [0,255,0], -1)
               izq = True
               der = True
               for caminos in salidas:
                  if salida[0]>caminos[0]:  izq=False
                  if salida[0]<caminos[0]:  der=False
               if izq:  cv2.putText(self.frame,'Tomamos la salida izquierda', (15,20), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0))
               elif der:  cv2.putText(self.frame,'Tomamos la salida derecha', (15,20), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0))
               else:  cv2.putText(self.frame,'Tomamos la salida central', (15,20), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0))
               tv = round((160.0-salida[1])/160.0,2)
               fv = round((170.0-salida[0])/170.0,2)
               return tv,fv
               
              
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
          
           if dis>2000 and ((salidas[0][0]-self.centro[0])*(far[1]-self.centro[1])-(salidas[0][1]-self.centro[1])*(far[0] -self.centro[0]))>0:
              cv2.putText(self.frame,'Curva hacia la izquierda', (15,20), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0))
              vel = 170-salidas[0][1]
              cv2.putText(self.frame,'Velocidad = {0} '.format(round(vel/170.0,2)), (15,40), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0))
              giro = 160-salidas[0][0]
              cv2.putText(self.frame,'Velocidad de giro = {0} '.format(round(giro/160.0,2)), (15,60), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0))
              tv = round(giro/160.0,2)
              fv = round(vel/170.0,2)
              return tv,fv
           elif dis>2000 and ((salidas[0][0]-self.centro[0])*(far[1]-self.centro[1])-(salidas[0][1]-self.centro[1])*(far[0] -self.centro[0]))<0:
              cv2.putText(self.frame,'Curva hacia la derecha', (15,20), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0))
              vel = 170-salidas[0][1]
              cv2.putText(self.frame,'Velocidad = {0} '.format(round(vel/170.0,2)), (15,40), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0))
              giro = 160-salidas[0][0]
              cv2.putText(self.frame,'Velocidad de giro = {0} '.format(round(giro/160.0,2)), (15,60), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0))
              tv = round(giro/160.0,2)
              fv = round(vel/170.0,2)
              return tv,fv
           else:
              cv2.putText(self.frame,'Recta', (15,20), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0))
              vel = 170-salidas[0][1]
              cv2.putText(self.frame,'Velocidad = {0} '.format(round(vel/170.0,2)), (15,40), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0))
              giro = 160-salidas[0][0]
              cv2.putText(self.frame,'Velocidad de giro = {0} '.format(round(giro/160.0,2)), (15,60), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0))
              tv = round(giro/160.0,2)
              fv = round(vel/170.0,2)
              return tv,fv
       
        return 0


    def __arrow_direction(self): # Esta por terminar
        linImg = (self.predImg==2).astype(np.uint8)*255
        _, contours, _ = cv2.findContours(linImg, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        if len(contours) > 0:
            contours = max(contours, key=lambda x: len(x)) # Si encuentra mas de un contorno
            if len(contours) < 100:
                return False

            caja = np.int0(cv2.boxPoints(cv2.fitEllipse(contours)))
            #cv2.drawContours(self.frame, [caja], -1, (255,0,0), 3)

            # Calculamos los puntos medios del rectangulo
            pm1 = (caja[3] + caja[0])/2
            pm2 = (caja[2] + caja[1])/2

            moments = cv2.moments(contours)
            x = int(moments['m10']/moments['m00'])
            y = int(moments['m01']/moments['m00'])
            cv2.circle(self.frame, (x, y), 2, (0,255,0), -1)
            punta = pm1 if self.distancia(pm1, (x,y)) < self.distancia(pm2, (x,y)) else pm2
            cv2.circle(self.frame, tuple(punta), 2, (0,0,255), -1)

            p,q = self.full_line(punta, (x,y))
            #cv2.line(self.frame, p, q, (0,0,255), 2)
            if self.distancia(p, punta) < self.distancia(p, (x,y)):
                self.auxiliar = p
            else:
                self.auxiliar = q

            return True


    def slope(self, x0, y0, x1, y1):
       return float((y1-y0)/(x1-x0))


    def full_line(self, a, b):
       slope = self.slope(a[0], a[1], b[0], b[1])
       p = [0,0]
       q = [self.width, self.height]

       p[1] = int(-(a[0] - p[0]) * slope + a[1])
       q[1] = int(-(b[0] - q[0]) * slope + b[1])

       #cv2.line(self.frame, tuple(p), tuple(q), (0,0,255))
       return p, q


    def distancia(self, a, b):
        return math.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)


if __name__ == "__main__":
    start = time()
    seg = Segmentador()
    print("Tiempo al crear el segmentador: {}".format(time() - start))
    if os.path.isfile('../clasificadores/segmentacion.plk'):
        seg.clf_load()
    else:
        seg.clf_create(sys.argv[1], sys.argv[2])
    print("Tiempo del clasificador: {}".format(time() - start))
    seg.video_create(sys.argv[3 if len(sys.argv) > 2 else 1])
    print("Tiempo total: {}".format(time() - start))
