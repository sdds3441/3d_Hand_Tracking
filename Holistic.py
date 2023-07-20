import mediapipe as mp
import cv2
import numpy as np
import socket
from cvzone.HandTrackingModule import HandDetector

def detect_dist (hands):
    detector = HandDetector(maxHands=1, detectionCon=0.8)
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
            return d

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
serverAddressPort = ("127.0.0.1", 5052)

buttonDelay = 10
buttonPressed = False
buttonCounter = 0
addObject = 'None'

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()

        height = frame.shape[0]
        width = frame.shape[1]

        data = []
        counter = 0

        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Make Detections
        results = holistic.process(image)

        # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks

        # Recolor image back to BGR for rendering
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 2. Right hand
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                                  )

        # 3. Left Hand
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                                  )

        # 4. Pose Detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )
        if results.pose_landmarks.landmark is not None:
            if results.pose_landmarks.landmark[15] is not None:
                R_wrist = results.pose_landmarks.landmark[15]
            if results.pose_landmarks.landmark[14] is not None:
                L_wrist = results.pose_landmarks.landmark[14]

            if (results.right_hand_landmarks is None or results.left_hand_landmarks is None) and (
                    results.right_hand_landmarks is not None or results.left_hand_landmarks is not None):
                if results.right_hand_landmarks is not None:
                    joint_data = []
                    joint = np.zeros((21, 4))
                    for j, lm in enumerate(results.right_hand_landmarks.landmark):
                        if j == 0:
                            z = round(results.pose_landmarks.landmark[15].z * width)
                        else:
                            z = round((results.pose_landmarks.landmark[15].z * width) + (lm.z * width))
                        joint[j] = [lm.x, lm.y, lm.z, lm.visibility]
                        data.extend([round(lm.x * width), round(height - (lm.y * height)), z])
                    data.append(addObject)

                if results.left_hand_landmarks is not None:
                    joint_data = []
                    joint = np.zeros((21, 4))
                    for j, lm in enumerate(results.left_hand_landmarks.landmark):
                        if j == 0:
                            z = round(results.pose_landmarks.landmark[14].z * width)
                        else:
                            z = round((results.pose_landmarks.landmark[14].z * width) + (lm.z * width))
                        joint[j] = [lm.x, lm.y, lm.z, lm.visibility]
                        data.extend([round(lm.x * width), round(height - (lm.y * height)), z])
                    data.append(addObject)

            elif results.right_hand_landmarks is not None and results.left_hand_landmarks is not None:
                L_joint_data = []
                L_joint = np.zeros((21, 4))
                for j, lm in enumerate(results.left_hand_landmarks.landmark):
                    if j == 0:
                        z = round(results.pose_landmarks.landmark[14].z * width)
                    else:
                        z = round((results.pose_landmarks.landmark[14].z * width) + (lm.z * width))
                    L_joint[j] = [lm.x, lm.y, lm.z, lm.visibility]
                    data.extend([round(lm.x * width), round(height - (lm.y * height)), z])
                data.append(addObject)

                R_joint_data = []
                R_joint = np.zeros((21, 4))
                for j, lm in enumerate(results.right_hand_landmarks.landmark):
                    if j == 0:
                        z = round(results.pose_landmarks.landmark[15].z * width)
                    else:
                        z = round((results.pose_landmarks.landmark[15].z * width) + (lm.z * width))
                    R_joint[j] = [lm.x, lm.y, lm.z, lm.visibility]
                    data.extend([round(lm.x * width), round(height - (lm.y * height)), z])
                data.append(addObject)
        #print(data)
        sock.sendto(str.encode(str(data)), serverAddressPort)

        cv2.imshow('Raw Webcam Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
