import cv2
from cvzone.HandTrackingModule import HandDetector
import socket
import mediapipe as mp

width, height = 1280, 720
# Webcam

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Hand Detect
detector = HandDetector(maxHands=1, detectionCon=0.8)  # 최대 손 1개, 신뢰도? 0.8

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
serverAddressPort = ("127.0.0.1", 5052)

buttonDelay = 30
buttonPressed = False
buttonCounter = 0
addObject = '0'

while True:
    success, img = cap.read()

    hands, img = detector.findHands(img)

    data = []

    if hands:
        hand = hands[0]
        pointLeft = hand['lmList'][2].copy()
        pointLeft.pop()
        pointRight = hand['lmList'][17].copy()
        pointRight.pop()
        w, _ = detector.findDistance(pointLeft, pointRight)
        W = 11

        f = 480
        d = (W * f) / w

        d = d - 50
        lmList = hand['lmList']

        if buttonPressed is False:
            hand
            print(hand)
            fingers = detector.fingersUp(hand)
            lmList = hand['lmList']
            print(lmList)

            indexFinger = lmList[8][0], lmList[8][1]

            if fingers == [1, 0, 0, 0, 0]:
                addObject = "cube"
                # buttonPressed = True

            elif fingers == [0, 0, 0, 0, 1]:
                addObject = "orb"
                # buttonPressed = True

            else:
                addObject = "None"
                # buttonPressed = True
        # print(addObject)

        lmList = hand['lmList']
        for lm in lmList:
            if lm[2] == 0:
                if buttonPressed is True:
                    addObject = 'None'
                else:
                    buttonPressed = True
                data.extend([lm[0], height - lm[1], d * 10, addObject])

            else:
                data.extend([lm[0], height - lm[1], ((d * 10) + lm[2]), 'None'])

        sock.sendto(str.encode(str(data)), serverAddressPort)

    if buttonPressed:
        buttonCounter += 1
        if buttonCounter > buttonDelay:
            buttonCounter = 0
            buttonPressed = False

    img = cv2.resize(img, (0, 0), None, 0.5, 0.5)
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
