import timeit
import mediapipe as mp
import cv2
import numpy as np
import socket


def cal_extend_data(hand_landmarks, pose_landmarks, which_hand):
    if which_hand == 'R':
        wrist = 16
    elif which_hand == 'L':
        wrist = 15

    h_joint = np.zeros((21, 3))
    for j, lm in enumerate(hand_landmarks.landmark):
        if j == 0:
            x = pose_landmarks.landmark[wrist].x
            y = pose_landmarks.landmark[wrist].y
            z = pose_landmarks.landmark[wrist].z
            pri_x = x
            pri_y = y
        else:
            x = pose_landmarks.landmark[wrist].x - (pri_x - lm.x)
            y = pose_landmarks.landmark[wrist].y - (pri_y - lm.y)
            z = (pose_landmarks.landmark[wrist].z + lm.z)

        data.extend([round(x * width), round(height - (y * height)), round(z, 5)])
        h_joint[j] = [lm.x, lm.y, lm.z]
    return h_joint


def cal_gesture(hand_joint, which):
    action = 0

    v1 = hand_joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19],
         :3]  # Parent joint
    v2 = hand_joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
         :3]  # Child joint
    v = v2 - v1  # [20, 3]

    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

    angle = np.arccos(np.einsum('nt,nt->n',
                                v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19],
                                :]))  # [15,]

    angle = np.degrees(angle)  # Convert radian to degree

    joint_data = np.array([angle], dtype=np.float32)
    ret, results, neighbours, dist = knn.findNearest(joint_data, 3)
    idx = int(results[0][0])

    # Draw gesture result
    if idx in rps_gesture.keys():
        if idx == 1:
            action = 1
        elif idx == 2:
            action = 2
        else:
            action = 0

    return action


mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
serverAddressPort = ("127.0.0.1", 5052)

rps_gesture = {0: 'stop', 1: 'flame', 2: 'release'}
data = []

width = 1280
height = 720

knn = cv2.ml.KNearest_create()
file = np.genfromtxt('particle.csv', delimiter=',')
angle = file[:, :-1].astype(np.float32)
label = file[:, -1].astype(np.float32)
knn.train(angle, cv2.ml.ROW_SAMPLE, label)
cap = cv2.VideoCapture(1)
cap.set(3, width)
cap.set(4, height)

# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        start_t = timeit.default_timer()

        data = []

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            results = holistic.process(image)

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

                for i, lm in enumerate(results.pose_landmarks.landmark):

                    if lm.visibility < 0.01:
                        visible = False
                    else:
                        visible = True

                    data.extend([round(lm.x * width), round(height - (lm.y * height)), round(lm.z, 3)])

                if (results.right_hand_landmarks is None or results.left_hand_landmarks is None) and (
                        results.right_hand_landmarks is not None or results.left_hand_landmarks is not None):

                    if results.right_hand_landmarks is not None:
                        joint = cal_extend_data(results.right_hand_landmarks, results.pose_landmarks, "R")

                    elif results.left_hand_landmarks is not None:
                        joint = cal_extend_data(results.left_hand_landmarks, results.pose_landmarks, "L")

                elif results.right_hand_landmarks is not None and results.left_hand_landmarks is not None:
                    cal_extend_data(results.right_hand_landmarks, results.pose_landmarks, "R")
                    cal_extend_data(results.left_hand_landmarks, results.pose_landmarks, "L")

            sock.sendto(str.encode(str(data)), serverAddressPort)
            print(data)
        except:
            a=1+1
        #print("detecting fail")

        #cv2.imshow('Raw Webcam Feed', image)

        terminate_t = timeit.default_timer()

        FPS = int(1. / (terminate_t - start_t))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
