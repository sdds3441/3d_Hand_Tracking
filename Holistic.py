import mediapipe as mp
import cv2
import numpy as np
import socket
from keras.models import load_model

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
serverAddressPort = ("127.0.0.1", 5052)

buttonDelay = 10
buttonPressed = False
buttonCounter = 0
addObject = 'None'

actions = ['come', 'hi', 'spin']
seq_length = 30


seq = []
action_seq = []
action_list = []

width = 1280
height = 720

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

model = load_model('models/model.h5')
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
                            x = results.pose_landmarks.landmark[16].x
                            y = results.pose_landmarks.landmark[16].y
                            z = round(results.pose_landmarks.landmark[16].z)
                            pre_x = x
                            pre_y = y
                        else:
                            x = results.pose_landmarks.landmark[16].x - (pre_x - lm.x)
                            y = results.pose_landmarks.landmark[16].y - (pre_y - lm.y)
                            z = results.pose_landmarks.landmark[16].z + lm.z
                        joint[j] = [lm.x, lm.y, lm.z, lm.visibility]
                        data.extend([round(x * width), round(height - (y * height)), z])

                    v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :3]  # Parent joint
                    v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                         :3]  # Child joint
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


                if results.left_hand_landmarks is not None:
                    joint_data = []
                    joint = np.zeros((21, 4))
                    for j, lm in enumerate(results.left_hand_landmarks.landmark):
                        if j == 0:
                            x = results.pose_landmarks.landmark[15].x
                            y = results.pose_landmarks.landmark[15].y
                            z = round(results.pose_landmarks.landmark[15].z)
                            pre_x=x
                            pre_y=y
                        else:
                            x = results.pose_landmarks.landmark[15].x - (pre_x-lm.x)
                            y = results.pose_landmarks.landmark[15].y - (pre_y-lm.y)
                            z = round(results.pose_landmarks.landmark[15].z + lm.z)
                        joint[j] = [lm.x, lm.y, lm.z, lm.visibility]
                        data.extend([round(x * width), round(height - (y * height)), z])

                sock.sendto(str.encode(str(data)), serverAddressPort)

            elif results.right_hand_landmarks is not None and results.left_hand_landmarks is not None:
                L_joint_data = []
                L_joint = np.zeros((21, 4))
                for j, lm in enumerate(results.left_hand_landmarks.landmark):
                    if j == 0:
                        x = results.pose_landmarks.landmark[15].x
                        y = results.pose_landmarks.landmark[15].y
                        z = round(results.pose_landmarks.landmark[15].z, 3)
                        pre_x = x
                        pre_y = y
                    else:
                        x = results.pose_landmarks.landmark[15].x - (pre_x - lm.x)
                        y = results.pose_landmarks.landmark[15].y - (pre_y - lm.y)
                        z = round(results.pose_landmarks.landmark[15].z + lm.z, 3)
                    L_joint[j] = [lm.x, lm.y, lm.z, lm.visibility]
                    data.extend([round(x * width), round(height - (y * height)), z])

                R_joint_data = []
                R_joint = np.zeros((21, 4))
                for j, lm in enumerate(results.right_hand_landmarks.landmark):
                    if j == 0:
                        x = results.pose_landmarks.landmark[16].x
                        y = results.pose_landmarks.landmark[16].y
                        z = round(results.pose_landmarks.landmark[16].z, 3)
                        pre_x = x
                        pre_y = y
                    else:
                        x = results.pose_landmarks.landmark[16].x - (pre_x - lm.x)
                        y = results.pose_landmarks.landmark[16].y - (pre_y - lm.y)
                        z = round(results.pose_landmarks.landmark[16].z + lm.z, 3)
                    R_joint[j] = [lm.x, lm.y, lm.z, lm.visibility]
                    data.extend([round(x * width), round(height - (y * height)), z])

                data.append(None)
                data.append(None)

            for i, lm in enumerate(results.pose_landmarks.landmark):

                if lm.visibility < 0.01:
                    visible = False
                else:
                    visible = True

                data.extend([round(lm.x * width), round(height - (lm.y * height)), round(lm.z, 3), visible])
            data.append(addObject)

        sock.sendto(str.encode(str(data)), serverAddressPort)

        if buttonPressed:
            buttonCounter += 1
            if buttonCounter > buttonDelay:
                buttonCounter = 0
                buttonPressed = False

        cv2.imshow('Raw Webcam Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
