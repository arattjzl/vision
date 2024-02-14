import cv2 as cv
import numpy as np
import dlib

cap = cv.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

while True:
    _, frame = cap.read()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        x, y = face.left(), face.top()
        x1, y1 = face.right(), face.bottom()
        #cv.rectangle(frame, (x, y), (x1, y1), (0, 0, 255), 2)
        landmarks = predictor(gray, face)
        x_left, y_left = landmarks.part(36).x, landmarks.part(36).y

        x_right, y_right = landmarks.part(39).x, landmarks.part(39).y

        x_top, y_top = landmarks.part(37).x, landmarks.part(37).y

        x_bottom, y_bottom = landmarks.part(41).x, landmarks.part(41).y

        cv.circle(frame, (x_left, y_left), 3, (0,0,255), 2)
        cv.circle(frame, (x_right, y_right), 3, (0,0,255), 2)

        if x_bottom == x_top and y_bottom == x_top:
            print("eye closed")

    cv.imshow('frame', frame)

    k = cv.waitKey(1)
    if k == 27:
        break

cap.release()
cv.destroyAllWindows()