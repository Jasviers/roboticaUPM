import cv2
from matplotlib import pyplot as plt
from scipy.misc import imsave

import select_pixels as sel

# Abres el video / camara con (Recordar cambiar de video si se quiere usar otro)
capture = cv2.VideoCapture("../rsc/line0.mp4")

# Lees las imagenes y las muestras para elegir la(s) de entrenamiento
# posibles funciones a usar

cv2.waitKey()
_, img = capture.read()
cv2.imshow("Captura", img)

capture.release()
cv2.destroyWindow("Captura")

# Si deseas mostrar la imagen con funciones de matplotlib posiblemente haya que cambiar
# el formato, con BGR a RGB
imNp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Esta funcion del paquete "select_pixels" pinta los pixeles en la imagen 
# Puede ser util para el entrenamiento

markImg = sel.select_fg_bg(imNp)

# Tambien puedes mostrar imagenes con las funciones de matplotlib
plt.imshow(markImg)
plt.show()

# Hay que cambiar el nombre del fichero para no sobreescribir 
imsave("../rsc/imgs/ln1.png", img)
imsave("../rsc/imgs/lnMark1.png", markImg)

