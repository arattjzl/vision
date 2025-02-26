import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

img = cv.imread('testFiles/eye.jpg', cv.IMREAD_GRAYSCALE)
img = cv.resize(img, (224, 224))
img = img / 255.0
img_array = np.array([img])

model = tf.keras.models.load_model('models/open-closed-eyes-v8.h5', compile=False)

predict = model.predict(img_array)
print(predict)
plt.imshow(img, cmap='gray')
plt.show()


