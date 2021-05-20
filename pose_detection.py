#!/usr/bin/python
# The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
#
#   This example program shows how to find frontal human faces in a webcam stream using OpenCV.
#   It is also meant to demonstrate that rgb images from Dlib can be used with opencv by just
#   swapping the Red and Blue channels.
#
#   You can run this program and see the detections from your webcam by executing the
#   following command:
#       ./opencv_face_detection.py
#
#   This face detector is made using the now classic Histogram of Oriented
#   Gradients (HOG) feature combined with a linear classifier, an image
#   pyramid, and sliding window detection scheme.  This type of object detector
#   is fairly general and capable of detecting many types of semi-rigid objects
#   in addition to human faces.  Therefore, if you are interested in making
#   your own object detectors then read the train_object_detector.py example
#   program.  
#
#
# COMPILING/INSTALLING THE DLIB PYTHON INTERFACE
#   You can install dlib using the command:
#       pip install dlib
#
#   Alternatively, if you want to compile dlib yourself then go into the dlib
#   root folder and run:
#       python setup.py install
#
#   Compiling dlib should work on any operating system so long as you have
#   CMake installed.  On Ubuntu, this can be done easily by running the
#   command:
#       sudo apt-get install cmake
#
#   Also note that this example requires Numpy which can be installed
#   via the command:
#       pip install numpy

import dlib
import cv2
from imutils import face_utils

detector = dlib.get_frontal_face_detector()
fileName = "shape_predictor_68_face_landmarks.dat"
poseModel = dlib.shape_predictor(fileName)
cam = cv2.VideoCapture(0)
color_green = (0,255,0)
line_width = 3
while True:
    ret_val, img = cam.read()
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    dets = detector(img, 1)
    num_faces = len(dets)
    # if num_faces == 0:
    #     print("Sorry, there were no faces found.")
    #     exit()

    faces = dlib.full_object_detections()
    for detection in dets:
        faces.append(poseModel(img, detection))

    for face in faces:
        point = face.part(33)
        cv2.circle(img, (point.x, point.y), radius=1, color=color_green, thickness=2)
        # rect = face.rect
        # cv2.rectangle(img,(rect.left(), rect.top()), (rect.right(), rect.bottom()), color_green, line_width)
        # for idx in face_utils.FACIAL_LANDMARKS_68_IDXS['nose']:
        #     point = face.part(idx)
        #     cv2.circle(img, (point.x, point.y), radius=1, color=color_green, thickness=2)
    cv2.imshow('my webcam', img)
    if cv2.waitKey(1) == 27:
        break  # esc to quit
cv2.destroyAllWindows()
