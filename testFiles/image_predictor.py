import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv


path_model = "https://www.kaggle.com/models/google/mobilenet-v2/TensorFlow2/100-224-feature-vector/2"
def tfLearning_model():
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Lambda(lambda x: hub.KerasLayer(path_model,trainable=False)(x)))
    model.add(tf.keras.layers.Dense(30,activation = 'relu'))
    model.add(tf.keras.layers.Dense(2, activation = 'sigmoid'))
    model.compile(optimizer = 'SGD' , loss = 'categorical_crossentropy' , metrics = ['accuracy',tf.keras.metrics.AUC()])
    model.build([None,224,224,3])

    return model

# model = tfLearning_model()
model = tf.keras.models.load_model('models/open-closed-eyes-v11.h5', compile=False)
model.summary()

img = cv.imread('testFiles/eye.jpg', cv.IMREAD_GRAYSCALE)
img = cv.resize(img, (224, 224))
img = img / 255.0
img_array = np.array([img])

predict = model.predict(img_array)
classid = np.argmax(predict, axis=1)[0]
print(classid)
plt.imshow(img, cmap='gray')
plt.show()
