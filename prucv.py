import numpy as np
import cv2 as cv

# Imagenes
img = cv.imread('control/line-images/lineBackground-1.png', 0)

cv.imshow('image', img)
cv.waitKey(0)
cv.destroyAllWindows()

# Video: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html#display-video
#

