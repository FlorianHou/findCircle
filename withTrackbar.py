from operator import countOf
from turtle import circle
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


def nothing(x):
    pass


img_raw = cv.imread("./pictures/10.bmp")
img_bgr = img_raw.copy()
img_lab = cv.cvtColor(img_bgr, cv.COLOR_BGR2LAB)
img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)


cv.namedWindow("image")
cv.createTrackbar("thresh", "image", 125, 255, nothing)
cv.setTrackbarMin("thresh", "image", 1)
cv.createTrackbar("clipLimit", "image", 8, 16, nothing)
cv.setTrackbarPos("clipLimit", "image", 8)
cv.setTrackbarMin("clipLimit", "image", 1)
while(1):
    img = img_lab[:, :, 1]
    thresh_val = cv.getTrackbarPos("thresh", "image")
    clipLimit_val = cv.getTrackbarPos("clipLimit", "image")
    # 使用局部亮度优化算法的效果比直接使用全局直方图亮度优化的效果要好
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(
        clipLimit_val, clipLimit_val))
    img = clahe.apply(img)
    # img = cv.equalizeHist(img)
    # img = cv.medianBlur(img, 3)
    img = cv.GaussianBlur(img, (5, 5), 0)
    # _, img = cv.threshold(img, thresh_val, 255, cv.THRESH_BINARY)
    img = cv.adaptiveThreshold(
        img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 51*2+1, 15)

    # kernel = np.ones((2,2), np.uint8)
    # img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
    kernel = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]], np.uint8)
    img = cv.dilate(img, kernel, iterations=1)

    contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    img = cv.Canny(img, 100, 200)
    h, w = np.shape(img)
    print(w, h)
    # 最小间距设置为h，从而只会找到一个圆
    circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT,
                              1, h, minRadius=int(h/8), maxRadius=int(h/3))
    circles = np.uint16(np.around(circles))
    for circle in circles[0, :]:
        center = (circle[0], circle[1])
        cv.circle(img_rgb, center, 15, (0, 100, 100), 8)
        radius = circle[2]
        cv.circle(img_rgb, center, radius, (0, 0, 255), 6)
        print("circle~")
    # # sobel 64F
    # sobel64 = cv.Sobel(img, cv.CV_64F, 1, 1, ksize=7)
    # img = np.uint8(abs(sobel64))
    # img = np.hstack((cv.equalizeHist(img_lab[:, :, 1]), img))\
    # lap64F = cv.Laplacian(img, cv.CV_64F, ksize=5)
    # img = np.uint8(abs(lap64F))
    cv.imshow("image", cv.resize(img, None, fx=1, fy=1))
    k = cv.waitKey(100)
    if k & 0xFF == ord('q'):
        break
cv.destroyAllWindows()
plt.imshow(img_rgb)
plt.show()
