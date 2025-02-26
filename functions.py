import numpy as np
import dlib

predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

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

def summary(inputs):
    open = 0
    close = 0
    total = []
    for i in range(1, len(inputs)):
        if inputs[i-1] == 0:
            close -= 1
        else:
            open += 1

        if inputs[i-1] != inputs[i]:
            if inputs[i-1] == 0:
                total.append(close)
            else:
                total.append(open)
            open = 0
            close = 0

    print(total)
    
    morse = []
    for i in total:
        if i > 30:
            morse.append(' ')
        elif i > 70:
            morse.append('  ')
        elif i < 0 and i > -9:
            morse.append('.')
        elif i < 0 and i > -20:
            morse.append('-')

    return morse
