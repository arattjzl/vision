import cv2 as cv
import numpy as np
import dlib
from math import hypot
import time

cap = cv.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
font = cv.FONT_HERSHEY_SIMPLEX

MORSE_CODE = {
    'a': '.-',
    'b': '-...',
    'c': '-.-.',
    'd': '-..',
    'e': '.',
    'f': '..-.',
    'g': '--.',
    'h': '....',
    'i': '..',
    'j': '.---',
    'k': '-.-',
    'l': '.-..',
    'm': '--',
    'n': '-.',
    'o': '---',
    'p': '.--.',
    'q': '--.-',
    'r': '.-.',
    's': '...',
    't': '-',
    'u': '..-',
    'v': '...-',
    'w': '.--',
    'x': '-..-',
    'y': '-.--',
    'z': '--..',
    '0': '-----',
    '1': '.----',
    '2': '..---',
    '3': '...--',
    '4': '....-',
    '5': '.....',
    '6': '-....',
    '7': '--...',
    '8': '---..',
    '9': '----.',
}

CODE = {}

def midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)   

def get_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)

    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

    #hor_line = cv.line(frame, left_point, right_point, (0, 255, 0), 1)
    #vert_line = cv.line(frame, center_top, center_bottom, (0, 255, 0), 1)

    ver_line_len = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))
    hor_line_len = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))

    return hor_line_len / ver_line_len

def handle_blink():
    #
    global algo 
    
while True:
    _, frame = cap.read()

    height, width, _ = frame.shape

    #matrix = cv.imread(frame)

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces = detector(gray)

    for face in faces:
        #cv.rectangle(frame, (x, y), (x1, y1), (0, 0, 255), 2)
        landmarks = predictor(gray, face)

        # blinking detection

        left_eye_ratio = get_blinking_ratio([36,37,38,39,40,41], landmarks)
        right_eye_ratio = get_blinking_ratio([42,43,44,45,46,47], landmarks)
        blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2

        # i want to be in this if until i open my eyes again
        if blinking_ratio > 5.3:
            global start 
            start = time.time()
            
            cv.putText(frame, 'BLINKING', (0, int(frame.shape[0]-10)), font, 3, (0, 0, 255))
        
        end = time.time()
        print(end-start)

            #if (end - start) 

        # gaze detection
        
        """ left_eye_region = np.array([
            (landmarks.part(36).x, landmarks.part(36).y),
            (landmarks.part(37).x, landmarks.part(37).y),
            (landmarks.part(38).x, landmarks.part(38).y),
            (landmarks.part(39).x, landmarks.part(39).y),
            (landmarks.part(40).x, landmarks.part(40).y),
            (landmarks.part(41).x, landmarks.part(41).y),
        ], np.int32)

        #cv.polylines(frame, [left_eye_region], True, [0, 0, 255], 2)

        mask = np.zeros((height, width), np.uint8)

        cv.polylines(frame, [left_eye_region], True, 255, 2)
        cv.fillPoly(mask, [left_eye_region], 255)
        left_eye = cv.bitwise_and(gray, gray, mask=mask)

        min_x = np.min(left_eye_region[:,0])
        max_x = np.max(left_eye_region[:,0])
        min_y = np.min(left_eye_region[:,1])
        max_y = np.max(left_eye_region[:,1])

        gray_eye = left_eye[min_y:max_y, min_x:max_x]

        _, threshold_eye = cv.threshold(gray_eye, 70, 255, cv.THRESH_BINARY)

        eye = cv.resize(gray_eye, None, fx=5, fy=5)
        threshold_eye = cv.resize(threshold_eye, None, fx=5, fy=5)

        cv.imshow('eye', eye)
        cv.imshow('threshold eye', threshold_eye) """

    cv.imshow('frame', frame)

    k = cv.waitKey(5)
    if k == 27:
        break

cap.release()
cv.destroyAllWindows()