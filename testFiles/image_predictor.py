import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv


model = tf.keras.models.load_model('models/open-closed-eyes-v12.h5', compile=False)
model.summary()

img = cv.imread('testFiles/eye.png', cv.IMREAD_GRAYSCALE)
img = cv.resize(img, (224, 224))
img = img / 255.0
img_array = np.array([img])

predict = model.predict(img_array)
classid = np.argmax(predict, axis=1)[0]
print(classid)
plt.imshow(img, cmap='gray')
plt.show()
