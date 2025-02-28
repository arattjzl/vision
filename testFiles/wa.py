import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

img = cv.imread('dataset/open/578.jpg', cv.IMREAD_GRAYSCALE)
img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)

print(img.shape)
plt.imshow(img, cmap='gray')
plt.show()
