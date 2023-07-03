import cv2
from cvzone.HandTrackingModule import HandDetector
import socket

width, height = 1280, 720
#Webcam

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

#Hand Detect
detector = HandDetector(maxHands=1, detectionCon=0.8) #최대 손 1개, 신뢰도? 0.8

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
serverAddressPort = ("127.0.0.1", 5052)

while True:
    success, img = cap.read()

    hands, img = detector.findHands(img)

    data = []

    if hands:
        hand = hands[0]

        lmList = hand['lmList']
        for lm in lmList:
            data.extend([lm[0], height - lm[1], lm[2]])
        sock.sendto(str.encode(str(data)), serverAddressPort)
        print(data)

    cv2.imshow("Image", img)
    cv2.waitKey(1)