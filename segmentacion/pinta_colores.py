
from scipy.misc import imread, imsave
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D as ax3d


# Leo las imagenes de entrenamiento
# imNp = imread('linea2017-2.png')
# imNpb = imread('linea2017-2b.png')
# markImg = imread('lineaMarcada2017-2.png')
# markImgb = imread('lineaMarcada2017-2b.png')

imNp = imread('linea2017-3.png')
markImg = imread('lineaMarcada2017-3.png')

# leo los datos
data_marca=imNp[np.where(np.all(np.equal(markImg,(255,0,0)),2))]
data_fondo=imNp[np.where(np.all(np.equal(markImg,(0,255,0)),2))]
data_linea=imNp[np.where(np.all(np.equal(markImg,(0,0,255)),2))]

# Pinto los datos RGB
fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot(data_marca[:,0],data_marca[:,1],data_marca[:,2],'r.',label='marca')
ax.plot(data_fondo[:,0],data_fondo[:,1],data_fondo[:,2],'g.',label='fondo')
ax.plot(data_linea[:,0],data_linea[:,1],data_linea[:,2],'b.',label='linea')
ax.set_xlabel('R Label')
ax.set_ylabel('G Label')
ax.set_zlabel('B Label')
ax.legend()
plt.title('Espacio RGB')


# Normalizo, vuelvo a leerlos y a pintarlos
imrgbn=np.rollaxis((np.rollaxis(imNp,2)+0.0)/np.sum(imNp,2),0,3)[:,:,:2]
imNp=imrgbn

data_marca=imNp[np.where(np.all(np.equal(markImg,(255,0,0)),2))]
data_fondo=imNp[np.where(np.all(np.equal(markImg,(0,255,0)),2))]
data_linea=imNp[np.where(np.all(np.equal(markImg,(0,0,255)),2))]

plt.figure()
plt.plot(data_marca[:,0],data_marca[:,1],'r.',label='marca')
plt.plot(data_fondo[:,0],data_fondo[:,1],'g.',label='fondo')
plt.plot(data_linea[:,0],data_linea[:,1],'b.',label='linea')
plt.title('Espacio RGB normalizado')

plt.show()

# Primera linea
p12 = plt.ginput(2, timeout=-1, show_clicks=True, mouse_pop=2, mouse_stop=3)
p1 = np.array(p12[0]+(1,))
p2 = np.array(p12[1]+(1,))
p = np.array(p12)
l1 = np.cross(p1,p2)
plt.plot(p[:,0],p[:,1],'r-')

# segunda linea
p12 = plt.ginput(2, timeout=-1, show_clicks=True, mouse_pop=2, mouse_stop=3)
p1 = np.array(p12[0]+(1,))
p2 = np.array(p12[1]+(1,))
l2 = np.cross(p1,p2)
p = np.array(p12)
plt.plot(p[:,0],p[:,1],'g-')


