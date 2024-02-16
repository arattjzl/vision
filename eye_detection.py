import cv2 as cv
import numpy as np
import dlib

cap = cv.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
eye_closed_count = 0

while True:
    _, frame = cap.read()

    matrix = cv.imread(frame)

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

        x_bottom, y_bottom = landmarks.part(41).x, landmarks.part(42).y

        #TODO quiero obtener la mitad entre top y bottom y si ese punto cambia de color (de blanco a otro ) quiere decir que cerro el ojo
        middle_of_eye = abs((x_top-x_bottom)/2)

        if matrix[middle_of_eye][y_top] == matrix[x_top][y_top]:
            eye_closed_count += 1
            print("eye closed", eye_closed_count)

        cv.circle(frame, (x_left, y_left), 3, (0,0,255), 2)
        cv.circle(frame, (x_right, y_right), 3, (0,0,255), 2)
        cv.circle(frame, (x_top, y_top), 3, (0,0,255), 2)
        cv.circle(frame, (x_bottom, y_bottom), 3, (0,0,255), 2)

    cv.imshow('frame', frame)

    k = cv.waitKey(1)
    if k == 27:
        break

cap.release()
cv.destroyAllWindows()