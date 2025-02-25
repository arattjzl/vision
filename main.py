import cv2 as cv
import numpy as np
import dlib
import time
import tensorflow as tf
import time
from morse3 import Morse


def getEyes(frame, face):
    landmarks = predictor(frame, face)

    # xtext = landmarks.part(28).x
    # ytext = landmarks.part(28).y

    x_left = landmarks.part(42).x
    y_left = landmarks.part(42).y

    # x_top = round((landmarks.part(43).x + landmarks.part(44).x) / 2)
    y_top = round((landmarks.part(43).y + landmarks.part(44).y) /2)

    x_right = landmarks.part(45).x
    # y_right = landmarks.part(45).y

    # x_bot = round((landmarks.part(46).x + landmarks.part(47).x) /2)
    y_bot = round((landmarks.part(46).y + landmarks.part(47).y) /2)

    

    xc = round((x_left + x_right) / 2)
    yc = round((y_top + y_bot) / 2)

    p1 = np.array([xc, yc])
    p2 = np.array([x_left, y_left])

    r = round((np.linalg.norm(p1 - p2))*2.5)

    topx = round(yc - r)  # Coordenada superior (Y)
    topy = round(xc - r)  # Coordenada superior (X)
    botx = round(yc + r)  # Coordenada inferior (Y)
    boty = round(xc + r)  # Coordenada inferior (X)

    return topx, topy, botx, boty

    
inputs = []
try:
    cap = cv.VideoCapture('testFiles/hola.mp4')
    # cap.set(cv.CAP_PROP_FPS, 30)
    fps = cap.get(cv.CAP_PROP_FPS)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')   
    model = tf.keras.models.load_model('models/open-closed-eyes-v3.h5', compile=False)
    labels = ['closed', 'open']
    frame_count = 0
    last = None
    startTimerClose = startTimerOpen = ''
    totalTimeClosed = totalTimeOpen = 0
    # TODO: change the frame ratio to use less resources
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
        
        cv.imshow("img", img_gray_resized)
        
        img_array = np.expand_dims(img_gray_resized, axis=0)  # Añadir una dimensión extra para el batch
        img_array = img_array / 255.0  # Normalizar

        
        # cv.imwrite("testFiles/eye.jpg", img)
        
        if frame_count % frame_ratio == 0:
            pred = model.predict(img_array)
            class_id = np.argmax(pred, axis=1)[0]
            # print("prediccion", class_id)

        #     # CLOSED EYE 
        #     if class_id == 0:

        #         if not isClose:
        #             startTimerClose = time.perf_counter()
        #             isClose = True

        #         if isOpen:
        #             endTimerOpen = time.perf_counter()
        #             totalTimeOpen = endTimerOpen - startTimerOpen
        #             print("OPEN", totalTimeOpen)

        #             if totalTimeOpen > 0.7:
        #                 inputs.append(' ')
        #             # if totalTimeOpen > 1.5:
        #             #     inputs.append('  ')
                    
        #             isOpen = False
        #             totalTimeOpen = 0

        #     # OPEN EYE
        #     if class_id == 1:

        #         if isClose:
        #             endTimerClose = time.perf_counter()
        #             totalTimeClosed = endTimerClose - startTimerClose
        #             print("CLOSE", totalTimeClosed)

        #             if 0 < totalTimeClosed < 0.4:
        #                 inputs.append('.')
        #             elif 0.4 < totalTimeClosed:
        #                 inputs.append('-')
                    
        #             isClose = False
        #             totalTimeClosed = 0

        #         if not isOpen:
        #             startTimerOpen = time.perf_counter()
        #             isOpen = True

        # print(inputs)
        # frame_count += 1

        # cv.imshow('frame', frame)

        if cv.waitKey(30) & 0xFF == ord('q'):
            break
except Exception as e:
    print(e)
finally:
    textEncoded = "".join(inputs)
    morse = Morse(textEncoded)
    textDecoded = morse.morseToString()
    print(textEncoded)
    print(textDecoded)

cap.release()
cv.destroyAllWindows()

