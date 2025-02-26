
import cv2 as cv
import numpy as np
import dlib
import time
import tensorflow as tf
import time
from morse3 import Morse
from functions import getEyes,summary
import matplotlib.pyplot as plt

model = tf.keras.models.load_model('models/open-closed-eyes-v8.h5', compile=False)

@tf.function
def fast_predict(x):
    return model(x)


inputs = []
open = []
close = []
block = []

try:
    test = 'testFiles/hello.mp4'
    cap = cv.VideoCapture(test)
    # cap.set(cv.CAP_PROP_FPS, 30)
    fps = cap.get(cv.CAP_PROP_FPS)
    detector = dlib.get_frontal_face_detector()   
    labels = ['closed', 'open']
    frame_count = 0
    openFrameCount = closeFrameCount = 0
    frame_ratio = 1
    isClose = isOpen = False

    while cap.isOpened():
        ret, frame = cap.read()
        width = int(frame.shape[1] * 0.25)
        height = int(frame.shape[0] * 0.25)
        frame = cv.resize(frame,(width, height))
        
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = detector(frame)
        # # if frame_count % frame_ratio == 0:
        
        topx, topy, botx, boty = getEyes(frame, faces[0])

        # # Extraer la región de interés (ROI)
        img = frame[topx:botx, topy:boty]
        img_gray_resized = cv.resize(img, (224, 224))  # Ajustar tamaño
        
        # plt.imshow(img_gray_resized, cmap='gray')
        
        img_array = np.expand_dims(img_gray_resized, axis=0)  # Añadir una dimensión extra para el batch
        img_array = img_array / 255.0  # Normaliza

        
        if frame_count % frame_ratio == 0:
            pred = model.predict(img_array)
            class_id = np.argmax(pred, axis=1)[0]
            # print("prediccion", class_id)

            # CLOSED EYE 
            if class_id == 0:
                inputs.append(0)
                block.append(0)     

            # OPEN EYE
            if class_id == 1:
                inputs.append(1)
                block.append(1)

            if len(block) == 5:
                if block[0] == block[-1] and block[0] != block[2]:
                    inputs[-3] = inputs[-1]
                elif block[0] != block[-1]
                elif block[-1] == block[-2] and block[1] == block[-1]:
                    inputs[-3] = inputs[-1]
                block.pop(0)
                    
        # # print(inputs)
        frame_count += 1

        if cv.waitKey(5) & 0xFF == ord('q'):
            break
    

except Exception as e:
    print(e)
finally:
    print(len(inputs), inputs)
    text = summary(inputs)
    textEncoded = "".join(text)
    morse = Morse(textEncoded)
    textDecoded = morse.morseToString()

    print(textEncoded)
    print(textDecoded)

cap.release()
cv.destroyAllWindows()

