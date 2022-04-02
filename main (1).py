from os import listdir
import os, random, copy
from PIL import Image
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots




ta = np.array([[1,2,3,4,5],[1,2,3,4,5]])

y = np.array([1,2,3])

coltot = y.sum(axis=0)
print(coltot)

wv = np.load('visualization.npy')
wv = wv[1:,]

eig = np.load('eigenvectors.npy')

imgmat = np.matmul(eig,wv)


imgmat += 5

img1 = imgmat[:,0]
img1 = np.reshape(img1, (200,300))
plt.imshow(img1)
plt.imsave('visualimg1',img1)

img2 = imgmat[:,1]
img2 = np.reshape(img2, (200,300))
plt.imshow(img2)
plt.imsave('visualimg2',img2)

img3 = imgmat[:,2]
img3 = np.reshape(img3, (200,300))
plt.imshow(img3)
plt.imsave('visualimg3',img3)

img4 = imgmat[:,3]
img4 = np.reshape(img4, (200,300))
plt.imshow(img4)
plt.imsave('visualimg4',img4)