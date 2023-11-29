import cv2
from cvzone.HandTrackingModule import HandDetector
import socket
import numpy as np
from keras.models import load_model
import time

width, height = 1280, 720
# Webcam
actions = ['grab', 'release']
seq_length = 30

cap = cv2.VideoCapture('C:\\Users\\김효찬\\PycharmProjects\\3d Hand Tracking\\videos\\video.mp4')
cap.set(3, width)
cap.set(4, height)

model = load_model('../models/model.h5')

# Hand Detect
detector = HandDetector(maxHands=1, detectionCon=0.8)  # 최대 손 1개, 신뢰도? 0.8

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
serverAddressPort = ("127.0.0.1", 5052)

buttonDelay = 10
buttonPressed = False
buttonCounter = 0
addObject = '0'

seq = []
action_seq = []
action_list = []
time_counter = True

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
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
        joint = np.zeros((21, 4))
        for j, lm in enumerate(lmList):
            joint[j] = [lm[0] / width, lm[1] / height, lm[2] / width, 0.0]
            if lm[2] == 0:
                data.extend([lm[0], height - lm[1], round(d * 10)])

            else:
                data.extend([lm[0], height - lm[1], round((d * 10) + lm[2])])

        v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :3]  # Parent joint
        v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :3]  # Child joint
        v = v2 - v1  # [20, 3]

        # Normalize v
        v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

        # Get angle using arcos of dot product
        angle = np.arccos(np.einsum('nt,nt->n',
                                    v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                    v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))  # [15,]

        angle = np.degrees(angle)  # Convert radian to degree

        degree = np.concatenate([joint.flatten(), angle])

        seq.append(degree)

        if len(seq) < seq_length:
            continue

        input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)

        y_pred = model.predict(input_data).squeeze()

        i_pred = int(np.argmax(y_pred))
        conf = y_pred[i_pred]

        if conf < 0.9:
            continue

        action = actions[i_pred]
        action_seq.append(action)

        if len(action_seq) < 3:
            continue

        action_list = []

        if action_seq[-1] == action_seq[-2] == action_seq[-3]:
            action_list.append(action)

        else:
            action_list = ["None"]

        if buttonPressed is True:

            this_action = 'None'
        else:
            this_action = max(set(action_list), key=action_list.count)
            buttonPressed = True
        data.append(this_action)
        if this_action != 'None':
            print(this_action)
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
