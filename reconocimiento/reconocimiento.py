#!/usr/bin/env python2.7

import cv2
import os
import sys
from time import time

import numpy as np
import sklearn.neighbors as neig
from sklearn.externals import joblib
from sklearn.model_selection import LeaveOneOut

import segmentacion.identificar_bifurcacion as bif


class Reconocimiento(object):


    def __init__(self):
        self.size = 0
        self.height = 0
        self.width = 0
        self.clf = None
        self.frame = None
        self.predImg = None
        self.etiquetas = {0: "Man", 1: "Stairs", 2: "Telephone", 3: "Woman"}
        if os.path.exists('../clasificadores/segmentacion.pkl'):
            self.segClf = joblib.load('../clasificadores/segmentacion.pkl')
        else:
            print("Necesario el clasificador de segmentacion.")
            sys.exit(-1)


    def __orb(self):
        pass


    def __hammingDist(self, d1, d2):
        assert d1.dtype == np.uint8 and d2.dtype == np.uint8
        d1_bits = np.unpackbits(d1)
        d2_bits = np.unpackbits(d2)
        return np.bitwise_xor(d1_bits, d2_bits).sum()


    def __hum(self, trainPath):
        etiq = 0
        self.size = len(os.listdir(trainPath))
        data, lbls = np.empty((self.size, 7)), np.empty(self.size)
        for i, img in enumerate(sorted(os.listdir(trainPath))):
            imNp = cv2.cvtColor(cv2.imread(trainPath+img), cv2.COLOR_BGR2RGB)
            predImg = self.segClf.predict(np.reshape(imNp, (imNp.shape[0]*imNp.shape[1], imNp.shape[2])))
            predImg = np.reshape(predImg, (imNp.shape[0], imNp.shape[1]))
            linImg = (predImg == 2).astype(np.uint8) * 255
            _, contours, _ = cv2.findContours(linImg, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            contours = max(contours, key=lambda x: len(x))
            data[i] = cv2.HuMoments(cv2.moments(contours, True)).T
            lbls[i] = etiq
            etiq += (((i+1) % 7) == 0)
        return data, lbls


    def clf_create(self, trainPath):
        # Creamos el clasificador
        self.clf = neig.KNeighborsClassifier(1, metric="euclidean")
        self.clf.fit(*self.__hum(trainPath))
        joblib.dump(self.clf, '../clasificadores/reconocimientoHu.pkl')


    def clf_load(self):
        self.clf = joblib.load('../clasificadores/reconocimientoHu.pkl')


    def kfolds(self, trainPath):
        data, lbls = self.__hum(trainPath)
        acc = 0
        for train, test in LeaveOneOut().split(data):
            clf = neig.KNeighborsClassifier(1, metric="euclidean")
            clf.fit(data[train], lbls[train])
            acc += (int(clf.predict(data[test])[0]) == int(lbls[test][0]))
        print("Aciertos con momentos de hu {} de {}".format(acc, self.size))


    def video_gererate(self, video):
        # Capturamos el video
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

        # Inicio la captura de imagenes
        capture = cv2.VideoCapture(video)
        count, filename = 0, 0
        centro = ()
        ret, self.frame = capture.read()

        while ret:

            if count % 3 == 0:
                self.height, self.width = self.frame.shape[:2]
                imNp = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)

                # Creamos la prediccion y redimensionamos
                predicted_image = self.segClf.predict(np.reshape(imNp, (imNp.shape[0] * imNp.shape[1], imNp.shape[2])))

                # Recuperamos las dimensiones
                self.predImg = np.reshape(predicted_image, (imNp.shape[0], imNp.shape[1]))

                #salidas, centro = bif.existen_bifurcaciones(self.frame, self.predImg, centro)
                #if not len(salidas) > 1:
                linImg = (self.predImg == 2).astype(np.uint8) * 255
                _, contours, _ = cv2.findContours(linImg, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                if len(contours) > 0:
                    contours = max(contours, key=lambda x: len(x))
                    if not len(contours) < 100:
                        fig = self.clf.predict(cv2.HuMoments(cv2.moments(contours, True)).T)
                        cv2.putText(self.frame, 'Identificado {} '.format(self.etiquetas[fig[0]]), (15, 60),
                                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0))

                paleta = np.array([(0, 0, 255), (255, 0, 0), (0, 255, 0)], dtype=np.uint8)

                cv2.waitKey(1)
                cv2.imshow("Pantalla de muestra", cv2.cvtColor(paleta[self.predImg], cv2.COLOR_RGB2BGR))

                cv2.imwrite('../rsc/generado/image%03d.png' % filename, cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR))
                filename += 1

                out.write(cv2.cvtColor(paleta[self.predImg], cv2.COLOR_RGB2BGR))

            count += 1
            ret, self.frame = capture.read()

        capture.release()
        out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    start = time()
    print("Comienzo de programa. Tiempo: {}".format(start))
    rec = Reconocimiento()
    print("Creacion de reconocedor de marcas. Tiempo: {}".format(time() - start))
    if os.path.exists('../clasificadores/reconocimientoHu.pkl'):
        rec.clf_load()
        print("Cargar el clasificador. Tiempo: {}".format(time() - start))
    else:
        rec.clf_create(sys.argv[1])
        print("Crear el clasificador. Tiempo: {}".format(time() - start))
    rec.video_gererate(sys.argv[2])
    print("Generar el video. Tiempo: {}".format(time() - start))
    rec.kfolds(sys.argv[1])
    print("Prueba de kfolds. Tiempo: {}".format(time() - start))
