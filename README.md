# 👁️ Computer Vision: Blink Detection as Morse Code

## 📌 Description

This project uses **computer vision and neural networks** to detect eye blinks and translate them into **Morse code signals**. The goal is to provide an alternative communication interface, especially for people with limited mobility.
You need to download [shape predictor](https://github.com/davisking/dlib-models/blob/master/shape_predictor_68_face_landmarks.dat.bz2) to be able to run this project.

## 🚀 Technologies Used

- **Python 3.x**
- **OpenCV** (Image processing)
- **Mediapipe** (Face and eye detection)
- **TensorFlow/Keras** (For improved detection using neural networks)

## 📸 How It Works

1. The camera captures the user's face in real-time.
2. Eyes are detected, and blink tracking is performed.
3. An algorithm measures the duration and frequency of blinks.
4. Short and long blinks are converted into Morse code signals using morse3 library.
5. The message is translated and displayed on the cmd.

## 🎯 Applications

- Assistive communication for people with disabilities.
- Contactless interfaces for devices.
- Biometric security and gesture-based control.

## 📩 Contact

If you have questions or suggestions, contact me at [**arattplz@gmail.com**](mailto:arattplz@gmail.com) or [**ulisesjafetmontoya@gmail.com**](mailto:ulisesjafetmontoya@gmail.com).
