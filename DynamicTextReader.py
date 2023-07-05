import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
import numpy as np
from PIL import ImageFont, ImageDraw, Image

cap = cv2.VideoCapture(0)
detector = FaceMeshDetector(maxFaces=1)

textList=["saebulk 5sie","jamdo anogo","sibal egae muhanen gerji","joatgotne"]

while True:
    success, img = cap.read()
    imgText = np.zeros_like(img)
    img, faces = detector.findFaceMesh(img, draw=False)

    if faces:
        face = faces[0]
        pointLeft = face[145]
        pointRight = face[374]
        w, _ = detector.findDistance(pointLeft, pointRight)
        W = 6.3

        f = 480
        d = (W * f) / w
        print(pointLeft)

        cvzone.putTextRect(img, f'Depth:{int(d)}cm', (face[10][0] - 75, face[10][1] - 50), scale=2)

        for i, text in enumerate(textList):
            singleHeight=50
            scale=0.4 + d/20
            cv2.putText(imgText, text, (50, 50+(i*singleHeight)), cv2.FONT_HERSHEY_PLAIN, scale, (255,255,255),2)

    imageStacked=cvzone.stackImages([img,imgText],2,1)
    cv2.imshow("image", imageStacked)
    cv2.waitKey(1)
