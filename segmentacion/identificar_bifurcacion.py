import cv2
from scipy.misc import imread, imsave
from matplotlib import pyplot as plt
import numpy as np
import math

def existen_bifurcaciones(img,imgMrk, centro):
   imNp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

   imgMrk = np.reshape(imgMrk, (img.shape[0], img.shape[1]))
   paleta = np.array([[0,0,255], [255,0,0], [0,255,0]], dtype=np.uint8)
   imgDest =  cv2.medianBlur(cv2.cvtColor(paleta[imgMrk], cv2.COLOR_RGB2BGR), 5)
   imgDest[np.where((imgDest==[0,0,255]).all(axis=2))] = [0,255,0]
   gray = cv2.cvtColor(imgDest, cv2.COLOR_BGR2GRAY) 
   edged = cv2.Canny(gray, 30, 200) 
   img2, contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
   
   camino = []
   result = 0
   salidas = []
   auxiliar = []
   nuevo_centro =()

   if not any(centro):
      for i in range(len(contours)):
         cont = contours[i]
         for e in range(len(cont)):
            point = cont[e]
            height, width = img.shape[:2]
            if (point[0][1]==height-1):
               auxiliar.append(point[0])
               camino.append(cont)
            elif (point[0][0]==0) or (point[0][0]==width-1) or (point[0][1]==0) or (point[0][1]==height-1):
               salidas.append(tuple(point[0]))
      x = abs(auxiliar[0][0]-auxiliar[1][0])/2
      nuevo_centro = min(auxiliar[0][0],auxiliar[1][0])+x,auxiliar[0][1]

   else:
      for i in range(len(contours)):
         cont = contours[i]
         for e in range(len(cont)):
            point = cont[e]
            height, width = img.shape[:2]
            if (point[0][1]==height-1) and ((math.sqrt(pow((point[0][0]-centro[0]),2)+pow((point[0][1]-centro[1]),2)))<30):
               auxiliar.append(point[0])
               camino.append(cont)
            elif (point[0][0]==0) or (point[0][0]==width-1) or (point[0][1]==0) or (point[0][1]==height-1):
               salidas.append(tuple(point[0]))
      x = abs(auxiliar[0][0]-auxiliar[1][0])/2
      nuevo_centro = min(auxiliar[0][0],auxiliar[1][0])+x,auxiliar[0][1]

   if len(camino)>1:
      for i in range(len(camino[0])):
         point = camino[0][i]
         if (abs(cv2.pointPolygonTest(camino[1], tuple(point[0]), measureDist = True))>40):
            result = 1
      for i in range(len(camino[1])):
         point = camino[1][i]
         if (abs(cv2.pointPolygonTest(camino[0], tuple(point[0]), measureDist = True))>40):
            result = 1
   return result, salidas, nuevo_centro

