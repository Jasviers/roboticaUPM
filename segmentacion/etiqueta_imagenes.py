import cv2
from scipy.misc import imread, imsave
from matplotlib import pyplot as plt
import select_pixels as sel


# Abres el video / camara con

capture = cv2.VideoCapture()

# Lees las imagenes y las muestras para elegir la(s) de entrenamiento
# posibles funciones a usar

cv2.waitKey()
capture.read()
cv2.imshow()

capture.release()
cv2.destroyWindow("Captura")

# Si deseas mostrar la imagen con funciones de matplotlib posiblemente haya que cambiar
# el formato, con
cv2.cvtColor(<img>, ...)

# Esta funcion del paquete "select_pixels" pinta los pixeles en la imagen 
# Puede ser util para el entrenamiento

markImg = sel.select_fg_bg(imNp)

# Tambien puedes mostrar imagenes con las funciones de matplotlib
plt.imshow(markImg)
plt.show()

# Si deseas guardar alguna imagen ....

imsave('lineaMarcada.png',markImg)

