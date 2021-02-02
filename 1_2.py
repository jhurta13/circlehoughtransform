import math
import cv2 as cv
import scipy
from PIL import Image
from numpy import asarray
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import skimage.feature
import numpy as np
import scipy.ndimage as ndimage

# read in image
from scipy.signal import argrelmax
from skimage.feature import peak_local_max

img = cv.imread("C:/Users/jhsandoval/Desktop/image/assignment2/coins.png", 0)

# gaussian blurring
blur = cv.GaussianBlur(img, (5, 5), 0)

# edge finding
edges = cv.Canny(blur, 100, 200)

# ranges of a,b,rmax params
a = int(edges.shape[0])
b = int(edges.shape[1])
rMax = 30

acc = np.zeros((a, b, rMax))

for x in range(0, edges.shape[0]):
    for y in range(0, edges.shape[1]):
        if edges[x, y] != 0:
            edge_points = [x, y]
            r = int(math.sqrt((x - edge_points[0]) ** 2 + (y - edge_points[1]) ** 2))
            acc[y, x, r] += 1

img = np.random.normal(size=(100, 256, 256))

img2 = ndimage.maximum_filter(acc, size=(1, 1, 1))

img_thresh = img2.mean() + img2.std() * 6

labels, num_labels = ndimage.label(img2 > img_thresh)

coords = ndimage.measurements.center_of_mass(acc, labels=labels, index=np.arange(1, num_labels + 1))

values = ndimage.measurements.maximum(acc, labels=labels, index=np.arange(1, num_labels + 1))

for i in coords:
    cv.circle(edges, (int(i[0]),int(i[1])), 40, (255,255,255), 1)
plt.imshow(edges)
plt.show()
print(len(coords))