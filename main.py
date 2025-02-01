import cv2 as cv
import numpy as np
import dlib
import matplotlib.pyplot as plt
import time
import tensorflow as tf

cap = cv.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')   

model = tf.keras.models.load_model('models/open-closed-eyes-v3.h5')

labels = ['closed', 'open']

frame_count = 0

while True:
    _, frame = cap.read()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        xtext = landmarks.part(28).x
        ytext = landmarks.part(28).y

        x_left = landmarks.part(42).x
        y_left = landmarks.part(42).y

        x_top = round((landmarks.part(43).x + landmarks.part(44).x) / 2)
        y_top = round((landmarks.part(43).y + landmarks.part(44).y) /2)

        x_right = landmarks.part(45).x
        y_right = landmarks.part(45).y

        x_bot = round((landmarks.part(46).x + landmarks.part(47).x) /2)
        y_bot = round((landmarks.part(46).y + landmarks.part(47).y) /2)

        """ cv.circle(frame, (x_left,y_left), 3, (0,0,255), 2)
        cv.circle(frame, (x_top,y_top), 3, (0,0,255), 2)
        cv.circle(frame, (x_right,y_right), 3, (0,0,255), 2)
        cv.circle(frame, (x_bot,y_bot), 3, (0,0,255), 2) """

        xc = round((x_left + x_right) / 2)
        yc = round((y_top + y_bot) / 2)

        p1 = np.array([xc, yc])
        p2 = np.array([x_left, y_left])

        r = round((np.linalg.norm(p1 - p2))*2.5)

        img = cv.circle(gray, (xc, yc), r, (0,0,255), 3)

        """ cv.rectangle(frame, (round(xc - r), round(yc + r)), (round(xc+r), round(yc - r)), 3) """

        topx = round(yc - r)  # Coordenada superior (Y)
        topy = round(xc - r)  # Coordenada superior (X)
        botx = round(yc + r)  # Coordenada inferior (Y)
        boty = round(xc + r)  # Coordenada inferior (X)

        # Extraer la región de interés (ROI)
        img = gray[topx:botx, topy:boty]

        img_gray_resized = cv.resize(img, (224, 224))  # Ajustar tamaño
        img_array = np.expand_dims(img_gray_resized, axis=0)  # Añadir una dimensión extra para el batch
        img_array = img_array / 255.0  # Normalizar


        pred = model.predict(img_array)
        class_id = np.argmax(pred, axis=1)
        print("prediccion", class_id)

        frame_count += 1

        cv.putText(frame, labels[class_id[0]], (xtext, ytext), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3, cv.LINE_AA)
        #cv.imshow("img", img)

    cv.imshow('frame', frame)

    if cv.waitKey(1) == ord('q'):
        break



cap.release()
cv.destroyAllWindows()

