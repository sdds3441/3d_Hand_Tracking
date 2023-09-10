import timeit
import mediapipe as mp
import cv2
import numpy as np
import threading
import sys


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


def cal_gesture(hand_joint):
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

    return joint_data


def timer():
    global sec
    sec += 1

    timers = threading.Timer(1, timer)
    timers.start()

    if sec == 3:
        timers.cancel()


rps_gesture = {0: 'grab', 5: 'release'}

width = 1280
height = 720

cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

sec = 0
pos_idx = 0
first = True
counter = 0

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()

        start_t = timeit.default_timer()

        data = []
        R_action = 0
        L_action = 0

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

        if results.right_hand_landmarks is not None:

            joint = cal_extend_data(results.right_hand_landmarks, results.pose_landmarks, "R")
            grab = cal_gesture(joint)

            if sec == 0:
                timer()
            # print(counter)

            if sec == 3:
                grab = np.append(grab, pos_idx)
                grab = np.expand_dims(grab, axis=0)
                if first:
                    grab_data = grab
                    first = False
                else:
                    grab_data = np.append(grab_data, grab, axis=0)
                print(grab_data.shape)
                sec = 0
                counter += 1
            if counter == 2:
                pos_idx += 1
                counter = 0

            """if cv2.waitKey(1) & 0xFF == ord('0'):
                grab = np.append(grab, int(0))
                grab = np.expand_dims(grab, axis=0)
                if counter == 0:
                    grab_data = grab
                else:
                    print(grab_data.shape)
                    grab_data = np.append(grab_data, grab, axis=0)

            elif cv2.waitKey(1) & 0xFF == ord('1'):
                grab = np.append(grab, int(1))
                print(grab)
                if counter == 0:
                    grab_data = grab
                else:
                    grab_data = np.append(grab_data, grab, axis=counter)
                counter += 1"""

        cv2.imshow('Raw Webcam Feed', image)
        print(pos_idx)
        if cv2.waitKey(1) & 0xFF == ord('q') or pos_idx==2:
            np.savetxt("light_gestures.csv", grab_data, delimiter=",", fmt="%.5f")
            exit()
