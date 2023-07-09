import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector

cap = cv2.VideoCapture(0)
detector = FaceMeshDetector(maxFaces=1)

while True:
    success, img = cap.read()
    img, faces = detector.findFaceMesh(img, draw=False)

    if faces:
        face = faces[0]
        pointLeft = face[145]
        pointRight = face[374]
        print(face)
        #cv2.circle(img, pointLeft, 5, (255, 0, 255), cv2.FILLED)
        #cv2.circle(img, pointRight, 5, (255, 0, 255), cv2.FILLED)
        #cv2.line(img, pointLeft, pointRight, (0, 200, 0), 3)
        w, _ = detector.findDistance(pointLeft, pointRight)
        W=6.3
        #d=50
        #f=(w*d)/W
        #print(f)

        f=480
        d=(W*f)/w
        print(d)

        cvzone.putTextRect(img, f'Depth:{int(d)}cm', (face[10][0]-75,face[10][1]-50),scale=2)
    cv2.imshow("image", img)
    cv2.waitKey(1)
