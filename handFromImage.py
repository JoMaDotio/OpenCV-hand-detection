import cv2 as cv
import numpy as np

img_path = './resourses/demo1.jpg'
img = cv.imread(img_path)
#cv.imshow('palm image', img)
print(img)
img_half = cv.resize(img, (0, 0), fx=0.2, fy=0.2)
cv.imshow('Half Image', img_half)

hsvim = cv.cvtColor(img_half, cv.COLOR_BGR2HSV)
lower = np.array([0, 80, 80], dtype="uint8")
upper = np.array([255, 255, 255], dtype="uint8")
skinRegionHSV = cv.inRange(hsvim, lower, upper)
blurred = cv.blur(skinRegionHSV, (2, 2))
ret, thresh = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY)
cv.imshow("thresh", thresh)

contours, hierarchy = cv.findContours(
    thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
contours = max(contours, key=lambda x: cv.contourArea(x))
cv.drawContours(img_half, [contours], -1, (255, 255, 0), 2)
cv.imshow("contours", img_half)


hull = cv.convexHull(contours)
cv.drawContours(img_half, [hull], -1, (0, 255, 255), 2)
cv.imshow("hull", img_half)
cv.waitKey()
cv.destroyAllWindows()
