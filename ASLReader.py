import cv2
from HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import tensorflow

#set up camera
cap = cv2.VideoCapture(0)
width  = int(cap.get(3))   # float `width`
height = int(cap.get(4))  # float `height`

#setup hand detection
detector = HandDetector(maxHands = 1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
tensorflow.keras.utils.disable_interactive_logging()

#for imageCrop
offset = 20
imgSize = 300

folder = "Data/Z"
counter = 0

labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
#retrain data
while True:
    #get camera input
    success, img = cap.read()
    #detect hands
    hands, img, img2 = detector.findHands(img, width, height)
    #get values of hand position
    if hands:
        hand = hands[0]
        x,y,w,h = hand['bbox']

        #setup for square image
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255

        #crop hand image

        imgCrop = img2[y-offset:y+h+offset, x-offset:x+w+offset]
        imgCropShape = imgCrop.shape
        
        aspectRatio = h/w
        #if height is too big
        if aspectRatio > 1:
            k = imgSize/h
            wCal = math.ceil(k*w)
            
            #make all images square and centered
            try:
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize-wCal)/2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            except:
                pass
            

        #if width is too big
        else:
            k = imgSize/w
            hCal = math.ceil(k*h)
            
            #make all images square and centered
            try:
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize-hCal)/2)
                imgWhite[hGap:hCal + hGap,:] = imgResize
            except:
                pass
        

        prediction, index = classifier.getPrediction(imgWhite, draw=False)

        cv2.rectangle(img, (x-offset-2, y-offset - 50), (x-offset+90, y - offset), (255,0,255), cv2.FILLED)
        cv2.putText(img, labels[index], (x+5, y-26), cv2.FONT_HERSHEY_COMPLEX, 1.5,(255,255,255), 2)
        cv2.rectangle(img, (x-offset, y-offset), (x+w+offset, y+h+offset), (255,0,255), 4)


    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
            break