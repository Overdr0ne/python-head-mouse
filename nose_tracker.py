#!/usr/bin/python
# Copyright (C) 2021 overdr0ne

# Author: overdr0ne
# Version: 0.1
# URL: https://github.com/Overdr0ne/python-head-mouse

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import dlib
import cv2
from imutils import face_utils

detector = dlib.get_frontal_face_detector()
fileName = "shape_predictor_68_face_landmarks.dat"
poseModel = dlib.shape_predictor(fileName)
cam = cv2.VideoCapture(0)
color_green = (0, 255, 0)
line_width = 3
while True:
    ret_val, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (640, 480))

    dets = detector(gray, 0)
    num_faces = len(dets)

    faces = dlib.full_object_detections()
    for detection in dets:
        faces.append(poseModel(img, detection))

    for face in faces:
        point = face.part(33)
        cv2.circle(img, (point.x, point.y), radius=1, color=color_green, thickness=2)

    cv2.imshow('my webcam', img)
    if cv2.waitKey(1) == 27:
        break  # esc to quit
cv2.destroyAllWindows()
cam.release()
