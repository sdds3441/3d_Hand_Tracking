import mediapipe as mp
import cv2
import numpy as np
import socket

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
serverAddressPort = ("127.0.0.1", 5052)

buttonDelay = 10
buttonPressed = False
buttonCounter = 0
addObject = 'None'

width = 1280
height = 720

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()

        # height = frame.shape[0]
        # width = frame.shape[1]

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

            if results.pose_landmarks.landmark[16] is not None:
                R_wrist = results.pose_landmarks.landmark[16]

            if results.pose_landmarks.landmark[15] is not None:
                L_wrist = results.pose_landmarks.landmark[15]

            if (results.right_hand_landmarks is None or results.left_hand_landmarks is None) and (
                    results.right_hand_landmarks is not None or results.left_hand_landmarks is not None):

                if results.right_hand_landmarks is not None:
                    joint_data = []
                    joint = np.zeros((21, 4))
                    for j, lm in enumerate(results.right_hand_landmarks.landmark):
                        if j == 0:
                            z = round(results.pose_landmarks.landmark[16].z)
                        else:
                            z = round(results.pose_landmarks.landmark[16].z + lm.z)
                        joint[j] = [lm.x, lm.y, lm.z, lm.visibility]
                        data.extend([round(lm.x * width), round(height - (lm.y * height)), z])

                if results.left_hand_landmarks is not None:
                    joint_data = []
                    joint = np.zeros((21, 4))
                    for j, lm in enumerate(results.left_hand_landmarks.landmark):
                        if j == 0:
                            z = round(results.pose_landmarks.landmark[15].z)
                        else:
                            z = round(results.pose_landmarks.landmark[15].z + lm.z)
                        joint[j] = [lm.x, lm.y, lm.z, lm.visibility]
                        data.extend([round(lm.x * width), round(height - (lm.y * height)), z])

            elif results.right_hand_landmarks is not None and results.left_hand_landmarks is not None:
                L_joint_data = []
                L_joint = np.zeros((21, 4))
                for j, lm in enumerate(results.left_hand_landmarks.landmark):
                    if j == 0:
                        z = round(results.pose_landmarks.landmark[15].z, 3)
                    else:
                        z = round(results.pose_landmarks.landmark[15].z + lm.z, 3)
                    L_joint[j] = [lm.x, lm.y, lm.z, lm.visibility]
                    data.extend([round(lm.x * width), round(height - (lm.y * height)), z])

                R_joint_data = []
                R_joint = np.zeros((21, 4))
                for j, lm in enumerate(results.right_hand_landmarks.landmark):
                    if j == 0:
                        z = round(results.pose_landmarks.landmark[16].z, 3)
                    else:
                        z = round(results.pose_landmarks.landmark[16].z + lm.z, 3)
                    R_joint[j] = [lm.x, lm.y, lm.z, lm.visibility]
                    data.extend([round(lm.x * width), round(height - (lm.y * height)), z])
                #data.append(None)
                data.append(None)

            for i, lm in enumerate(results.pose_landmarks.landmark):

                if lm.visibility < 0.01:
                    visible = False
                else:
                    visible = True

                data.extend([round(lm.x * width), round(height - (lm.y * height)), round(lm.z, 3), visible])
            data.append(addObject)

        sock.sendto(str.encode(str(data)), serverAddressPort)

        cv2.imshow('Raw Webcam Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
