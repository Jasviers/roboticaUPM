import cv2
from scipy.misc import imread, imsave
from matplotlib import pyplot as plt
import numpy as np

def existen_bifurcaciones(img,imgMrk):
   imNp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

   imgMrk = np.reshape(imgMrk, (img.shape[0], img.shape[1]))
   paleta = np.array([[0,0,255], [255,0,0], [0,255,0]], dtype=np.uint8)
   imgDest =  cv2.medianBlur(cv2.cvtColor(paleta[imgMrk], cv2.COLOR_RGB2BGR), 5)
   imgDest[np.where((imgDest==[0,0,255]).all(axis=2))] = [0,255,0]
   gray = cv2.cvtColor(imgDest, cv2.COLOR_BGR2GRAY) 
   edged = cv2.Canny(gray, 30, 200) 
   img2, contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
   
   camino = []
   result = False

   for i in range(len(contours)):
      cont = contours[i]
      for e in range(len(cont)):
         point = cont[e]
         height, width = img.shape[:2]
         if (point[0][1]==height-1):
            camino.append(cont)

   for i in range(len(camino[0])):
      point = camino[0][i]
      if (abs(cv2.pointPolygonTest(camino[1], tuple(point[0]), measureDist = True))>50):
         result = True
   return result

