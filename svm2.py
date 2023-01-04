from tkinter import *
import tkinter
from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np
import imutils
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import dlib
import cv2
import pygame #For playing sound
import time


main = tkinter.Tk()
main.title("Driver Drowsiness Monitoring")
main.geometry("800x600")
pygame.mixer.init()
pygame.mixer.music.load('se/audio/audio_alert.wav')



def EAR(drivereye):
    point1 = dist.euclidean(drivereye[1], drivereye[5])
    point2 = dist.euclidean(drivereye[2], drivereye[4])
    # compute the euclidean distance between the horizontal
    distance = dist.euclidean(drivereye[0], drivereye[3])
    # compute the eye aspect ratio
    ear_aspect_ratio = (point1 + point2) / (2.0 * distance)
    return ear_aspect_ratio


webcamera = cv2.VideoCapture(0)
svm_predictor_path = SVC()
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 10
MOU_AR_THRESH = 0.75

COUNTER = 0
yawnStatus = False
yawns = 0
svm_detector = dlib.get_frontal_face_detector()
svm_predictor = dlib.shape_predictor()
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    
while True:
    ret, frame = webcamera.read()
    frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_yawn_status = yawnStatus
    rects = svm_detector(gray, 0)
    for rect in rects:
        shape = svm_predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = EAR(leftEye)
        rightEAR = EAR(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 255), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 255), 1)
        cv2.putText(frame,"Visual Behaviour & Machine Learning Drowsiness Detection @ Drowsiness",(370,470),cv2.FONT_HERSHEY_COMPLEX,0.6,(153,51,102),1)
        if(ear < EYE_AR_THRESH):
            COUNTER += 1
            #If no. of frames is greater than threshold frames,
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                pygame.mixer.music.play(-1)
                cv2.putText(frame, "You are Drowsy", (150,200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2)
        else:
            pygame.mixer.music.stop()
            COUNTER = 0
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
cv2.destroyAllWindows()
webcamera.release()    

  