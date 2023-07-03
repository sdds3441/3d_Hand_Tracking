import cv2
from cvzone.HandTrackingModule import HandDetector
import mediapipe

#Webcam

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

#Hand Detect
detector = HandDetector(maxHands=1, detectionCon=0.8) #최대 손 1개, 신뢰도? 0.8

while True:
    success, img = cap.read()

    hand, img = detector.findHands(img)

    cv2.imshow("Image", img)
    cv2.waitKey(1)