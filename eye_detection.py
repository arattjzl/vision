import cv2 as cv
import numpy as np
import dlib

cap = cv.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
eye_closed_count = 0

def midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)   

while True:
    _, frame = cap.read()

    #matrix = cv.imread(frame)

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        x, y = face.left(), face.top()
        x1, y1 = face.right(), face.bottom()

        #cv.rectangle(frame, (x, y), (x1, y1), (0, 0, 255), 2)
        landmarks = predictor(gray, face)
        x_36, y_36 = landmarks.part(36).x, landmarks.part(36).y

        x_39, y_39 = landmarks.part(39).x, landmarks.part(39).y

        x_37, y_37 = landmarks.part(37).x, landmarks.part(37).y

        x_38, y_38 = landmarks.part(38).x, landmarks.part(38).y

        x_41, y_41 = landmarks.part(41).x, landmarks.part(41).y

        x_40, y_40 = landmarks.part(40).x, landmarks.part(40).y

        #TODO quiero obtener la mitad entre top y bottom y si ese punto cambia de color (de blanco a otro ) quiere decir que cerro el ojo
        #middle_of_eye = abs((x_top-x_bottom)/2)

        left_point = (landmarks.part(36).x, landmarks.part(36).y)
        right_point = (landmarks.part(39).x, landmarks.part(39).y)

        center_top = midpoint(landmarks.part(37), landmarks.part(38))
        center_bottom = midpoint(landmarks.part(41), landmarks.part(40))

        hor_line = cv.line(frame, left_point, right_point, (0, 255, 0), 1)
        vert_line = cv.line(frame, center_top, center_bottom, (0, 255, 0), 1)

    cv.imshow('frame', frame)

    k = cv.waitKey(1)
    if k == 27:
        break

cap.release()
cv.destroyAllWindows()