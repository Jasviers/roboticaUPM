import math

import cv2
import numpy as np


def existen_bifurcaciones(img, imgMrk, centro, ultSalida):
 
   linImg = (imgMrk==1).astype(np.uint8)*255
   _, contours, _ = cv2.findContours(linImg, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE) #sacamos los contornos de la linea
   bordes = []
   bordeN = []
   bordeS = []
   bordeW = []
   bordeE = []
   for cont in contours: #aqui vamos a guardar las distintas salidas y el borde de la imagen por el que salen
      for point in cont:
         height, width = img.shape[:2]
         if (point[0][0]==0):  bordeW.append(tuple(point[0]))
         elif (point[0][0]==width-1):   bordeE.append(tuple(point[0]))
         elif (point[0][1]==0):   bordeN.append(tuple(point[0]))
         elif (point[0][1]==height-1):   bordeS.append(tuple(point[0]))
   bordeN.sort()
   bordeS.sort()
   bordeW.sort()
   bordeE.sort()

   #el siguiente codigo se encarga de poder separar los puntos de cada borde en distintos bordes
   if len(bordeW)>0:
      definitiva=[]
      auxiliar=[]
      for point in bordeW:
         if len(auxiliar)<1:
            auxiliar.append(point)
         elif math.sqrt(pow((point[0]-auxiliar[-1][0]),2)+pow((point[1]-auxiliar[-1][1]),2))<5:
            auxiliar.append(point)
         else:
            definitiva.append(auxiliar)
            auxiliar=[]
            auxiliar.append(point)
      definitiva.append(auxiliar)
      for lista in definitiva:
         bordes.append(lista)

   if len(bordeE)>0:
      definitiva=[]
      auxiliar=[]
      for point in bordeE:
         if len(auxiliar)<1:
            auxiliar.append(point)
         elif math.sqrt(pow((point[0]-auxiliar[-1][0]),2)+pow((point[1]-auxiliar[-1][1]),2))<5:
            auxiliar.append(point)
         else:
            definitiva.append(auxiliar)
            auxiliar=[]
            auxiliar.append(point)
      definitiva.append(auxiliar)
      for lista in definitiva:
         bordes.append(lista)

   if len(bordeN)>0:
      definitiva=[]
      auxiliar=[]
      for point in bordeN:
         if len(auxiliar)<1:
            auxiliar.append(point)
         elif math.sqrt(pow((point[0]-auxiliar[-1][0]),2)+pow((point[1]-auxiliar[-1][1]),2))<5:
            auxiliar.append(point)
         else:
            definitiva.append(auxiliar)
            auxiliar=[]
            auxiliar.append(point)
      definitiva.append(auxiliar)
      for lista in definitiva:
         bordes.append(lista)

   if len(bordeS)>0:
      definitiva=[]
      auxiliar=[]
      for point in bordeS:
         if len(auxiliar)<1:
            auxiliar.append(point)
         elif math.sqrt(pow((point[0]-auxiliar[-1][0]),2)+pow((point[1]-auxiliar[-1][1]),2))<5:
            auxiliar.append(point)
         else:
            definitiva.append(auxiliar)
            auxiliar=[]
            auxiliar.append(point)
      definitiva.append(auxiliar)
      for lista in definitiva:
         bordes.append(lista)

   bordes.sort()
   definitiva=[]
   auxiliar=[]
   for i, borde in enumerate(bordes):
      if all(elem in auxiliar for elem in borde): continue
      auxiliar=[]
      flag = True
      if i == len(bordes)-1:  flag=False
      for point in borde:
         auxiliar.append(point)
         if flag:
            for point2 in bordes[i+1]:
               if math.sqrt(pow((point[0]-point2[0]),2)+pow((point[1]-point2[1]),2))<3:
                  for point2 in bordes[i+1]:  auxiliar.append(point2)
                  flag=False
                  break
      auxiliar.sort()
      definitiva.append(auxiliar)
   definitiva.sort()
   bordes=definitiva
         

   salidas = []
   auxiliar = []
   nuevo_centro =()
   puntos_salida=[]
   contornolinea=[]

   if not any(centro): # Si no existe un centro previo, buscamos uno que interseccione con el borde inferior de la imagen
      for cont in bordes:
         for point in cont:
            height, width = img.shape[:2]
            if cont not in auxiliar and (point[1]==height-1):
               auxiliar.append(cont)
            elif cont not in salidas:
               salidas.append(cont)
      if len(auxiliar) > 0:
         x = auxiliar[0]
         x = x[len(x)/2] # El centro estara en la posicion intermedia del borde de entrada
         nuevo_centro = x[0],x[1]
      if len(salidas) > 0:
         for salida in salidas:
            x = salida[len(salida)/2]
            puntos_salida.append(x)

   elif bordes:
      dist = math.sqrt(pow((bordes[0][0][0]-centro[0]), 2)+pow((bordes[0][0][1]-centro[1]), 2))
      caminoaprox = bordes[0]
      for i in range(len(bordes)): # Si existe un centro previo
         cont = bordes[i]
         for e in range(len(cont)):
            point = cont[e]
            if math.sqrt(pow((point[0]-centro[0]),2)+pow((point[1]-centro[1]),2)) < dist:
               dist = math.sqrt(pow((point[0]-centro[0]),2)+pow((point[1]-centro[1]),2))
               caminoaprox = cont
            if cont not in salidas:
               salidas.append(cont)
      x = caminoaprox[len(caminoaprox)/2]
      nuevo_centro = x[0],x[1]
      if len(salidas)>0:
         for salida in salidas:
            x = salida[len(salida)/2]
            puntos_salida.append(x)#cada salida es el punto medio de los bordes de salida

     

   if nuevo_centro:
      for cont in contours:#identificamos el contorno que determina nuestro camino
         if cv2.pointPolygonTest(cont, nuevo_centro, False)== 0.0:
            contornolinea=cont

      for x in puntos_salida:#sacamos el punto de entrada de la lista de salidas
         if type(x)==tuple:
            if x==nuevo_centro:  puntos_salida.remove(x)

      #elimino las salidas que no forman parte del camino
      puntos_salida[:] = [x for x in puntos_salida if len(contornolinea)>0 and abs(cv2.pointPolygonTest(contornolinea, x, True))<5]


   #si existe un corte en la imagen del camino hacemos lo siguiente
   if len(puntos_salida)==1 and len(contours)>1 and math.sqrt(pow((puntos_salida[0][0]-ultSalida[0]),2)+pow((puntos_salida[0][1]-ultSalida[1]),2)) > 100:
      print 'hola'
      dist = math.sqrt(pow((bordes[0][0][0]-centro[0]), 2)+pow((bordes[0][0][1]-centro[1]), 2))
      caminoaprox = bordes[0]
      salidas = []
      puntos_salida = []
      for i in range(len(bordes)): 
         cont = bordes[i]
         for e in range(len(cont)):
            point = cont[e]
            if math.sqrt(pow((point[0]-ultSalida[0]),2)+pow((point[1]-ultSalida[1]),2)) < dist:
               dist = math.sqrt(pow((point[0]-ultSalida[0]),2)+pow((point[1]-ultSalida[1]),2))
               caminoaprox = cont
            if cont not in salidas:
               salidas.append(cont)
      x = caminoaprox[len(caminoaprox)/2]
      nueva_salida = x[0],x[1]
      for cont in contours:#identificamos el contorno que determina nuestro camino basandonos en la salida
         if cv2.pointPolygonTest(cont, nueva_salida, False)== 0.0:
            contornolinea=cont
      for borde in bordes:
         print borde
         for e in range(len(borde)):
            point = borde[e]
            if cv2.pointPolygonTest(contornolinea, point, False)== 0.0 and math.sqrt(pow((nueva_salida[0]-point[0]),2)+pow((nueva_salida[1]-point[1]),2)) > 100:
               x = borde[len(borde)/2]
               nuevo_centro = x[0],x[1]
      if len(salidas)>0:
         for salida in salidas:
            x = salida[len(salida)/2]
            puntos_salida.append(x)#cada salida es el punto medio de los bordes de salida
         

   if nuevo_centro:
      for cont in contours:#identificamos el contorno que determina nuestro camino
         if cv2.pointPolygonTest(cont, nuevo_centro, False)== 0.0:
            contornolinea=cont

      for x in puntos_salida:#sacamos el punto de entrada de la lista de salidas
         if type(x)==tuple:
            if x==nuevo_centro:  puntos_salida.remove(x)

      #elimino las salidas que no forman parte del camino
      puntos_salida[:] = [x for x in puntos_salida if len(contornolinea)>0 and abs(cv2.pointPolygonTest(contornolinea, x, True))<5]

      #devuelvo contorno del camino, salidas y centro
      cv2.drawContours(img, contornolinea, -1, (0, 255, 0), 2)
      cv2.circle(img, nuevo_centro, 2, (255,0,0))
   return puntos_salida, nuevo_centro
