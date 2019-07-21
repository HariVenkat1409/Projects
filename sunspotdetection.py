import cv2
import numpy as np
from matplotlib import pyplot as plt
def getMaxContour(contours,minArea=0.1):
    maxC=None
    maxArea=minArea

    mc1=None
    mc2=None
    mc3=None

    ma1=minArea
    ma2=minArea
    ma3=minArea

    for cnt in contours:
        area=cv2.contourArea(cnt)
        if(area>ma1):
            mc1=cnt
            ma1=area


    for cnt in contours:
        area=cv2.contourArea(cnt)
        if(area>ma2) and (area<ma1):
            mc2=cnt
            ma2=area

    for cnt in contours:
        area=cv2.contourArea(cnt)
        if(area>ma1) and (area<ma2):
            mc3=cnt
            ma3=area
    

    return mc1,mc2,mc3

img = cv2.imread('one1.jpg',0)
kernel9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(80,80))
np.array([[1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1]], dtype=np.uint8)
blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel9)
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blackhat,cmap = 'gray')
plt.title('Blackhat Image'), plt.xticks([]), plt.yticks([])
plt.show()
thresh = cv2.threshold(blackhat, 80,90, cv2.THRESH_BINARY)[1]
plt.subplot(121),plt.imshow(blackhat,cmap = 'gray')
plt.title('Blackhat Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(thresh,cmap = 'gray')
plt.title('Threshold Image'), plt.xticks([]), plt.yticks([])
plt.show()
cont, hier =cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contA,contB,contC=getMaxContour(cont,400) 
((x, y), radius) = cv2.minEnclosingCircle(contA)
M = cv2.moments(contA)
center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
cv2.circle(img, (int(x), int(y)), int(radius),(0, 0, 255), 7)
cv2.circle(img, center, 5, (0, 0, 255), -1)
((x, y), radius) = cv2.minEnclosingCircle(contB)
M = cv2.moments(contB)
center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
cv2.circle(img, (int(x), int(y)), int(radius),(0, 0, 255), 7)
cv2.circle(img, center, 5, (0, 0, 255), -1)
img = cv2.resize(img,(540,540))
while(1):
    cv2.imshow('TEST',img)
    k = 0xFF & cv2.waitKey(10)
    if k == 27:
        break
