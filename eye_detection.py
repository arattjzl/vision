import cv2 as cv
import numpy as np
import dlib
from math import hypot

cap = cv.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
font = cv.FONT_HERSHEY_SIMPLEX

def midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)   

while True:
    _, frame = cap.read()

    #matrix = cv.imread(frame)

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        #cv.rectangle(frame, (x, y), (x1, y1), (0, 0, 255), 2)
        landmarks = predictor(gray, face)

        left_point = (landmarks.part(36).x, landmarks.part(36).y)
        right_point = (landmarks.part(39).x, landmarks.part(39).y)

        center_top = midpoint(landmarks.part(37), landmarks.part(38))
        center_bottom = midpoint(landmarks.part(41), landmarks.part(40))

        hor_line = cv.line(frame, left_point, right_point, (0, 255, 0), 1)
        vert_line = cv.line(frame, center_top, center_bottom, (0, 255, 0), 1)

        ver_line_len = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))
        hor_line_len = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))

        ratio = hor_line_len / ver_line_len

        if ratio > 3.9:
            cv.putText(frame, 'BLINKING', (landmarks.part(8).x, landmarks.part(8).y), font, 2, (0, 0,255))

    cv.imshow('frame', frame)

    k = cv.waitKey(1)
    if k == 27:
        break

cap.release()
cv.destroyAllWindows()