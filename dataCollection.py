import cv2
from HandTrackingModule import HandDetector
import numpy as np
import math
import time

#set up camera
cap = cv2.VideoCapture(0)
width  = int(cap.get(3))   # float `width`
height = int(cap.get(4))  # float `height`

#setup hand detection
detector = HandDetector(maxHands = 1)

#for imageCrop
offset = 20
imgSize = 300

folder = "Data/Z"
counter = 0
#retrain data
#42:30
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
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize-wCal)/2)
            imgWhite[:, wGap:wCal + wGap] = imgResize

        #if width is too big
        else:
            k = imgSize/w
            hCal = math.ceil(k*h)
            
            #make all images square and centered
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize-hCal)/2)
            imgWhite[hGap:hCal + hGap,:] = imgResize

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
            break
    if key == ord("s"):
         counter += 1
         cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
         print(counter)