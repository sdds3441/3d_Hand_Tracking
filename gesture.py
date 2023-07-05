import os
import numpy as np
import cv2
from cvzone.HandTrackingModule import HandDetector


def hangulFilePathImageRead(filePath):
    stream = open(filePath.encode("utf-8"), "rb")
    bytes = bytearray(stream.read())
    numpyArray = np.asarray(bytes, dtype=np.uint8)

    return cv2.imdecode(numpyArray, cv2.IMREAD_UNCHANGED)


folderPath = "C:\\Users\\김효찬\\PycharmProjects\\3d_Hand_Tracking\\presentation"

cap = cv2.VideoCapture(0)

cap.set(3, 1280)
cap.set(4, 720)

pathImg = os.listdir(folderPath)
imgNum = 0
hs, ws = int(120 * 1), int(213 * 1)
gestureThreshold = 400
buttonPressed = False
buttonCounter = 0
buttonDelay = 30

detector = HandDetector(detectionCon=0.8, maxHands=1)
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    pathFullImages = os.path.join(folderPath, pathImg[imgNum])
    # imgCurrent=cv2.imread(pathFullImages)
    imgCurrent = hangulFilePathImageRead(pathFullImages)

    hands, img = detector.findHands(img)
    cv2.line(img, (0, gestureThreshold), (1280, gestureThreshold), (0, 255, 255), 10)

    if hands and buttonPressed is False:
        hand = hands[0]
        fingers = detector.fingersUp(hand)
        cx, cy = hand['center']
        lmList = hand['lmList']
        indexFinger = lmList[8][0], lmList[8][1]

        if cy <= gestureThreshold:

            if fingers == [1, 0, 0, 0, 0]:
                print("Left")
                if imgNum > 0:
                    buttonPressed = True
                    imgNum -= 1

            if fingers == [0, 0, 0, 0, 1]:
                print("Right")
                if imgNum < len(pathImg) - 1:
                    buttonPressed = True
                    imgNum += 1

        if fingers == [0, 1, 1, 0, 0]:
            print("Right")
            cv2.circle(imgCurrent, indexFinger, 12, (255, 0, 0), cv2.FILLED)

    if buttonPressed:
        buttonCounter += 1
        if buttonCounter > buttonDelay:
            buttonCounter = 0
            buttonPressed = False

    imgSmall = cv2.resize(img, (ws, hs))
    h, w, _ = imgCurrent.shape
    imgCurrent[0:hs, w - ws:w] = imgSmall

    # cv2.imshow("Image", img)
    cv2.imshow("Slide", imgCurrent)
    key = cv2.waitKey(1)

    if key == ord('q'):
        break
