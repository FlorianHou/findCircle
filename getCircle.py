import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def showim(img):
    cv.namedWindow("window", cv.WINDOW_NORMAL)
    cv.imshow("window", img)
    k = cv.waitKey(0)
    if k == 'q':
        cv.destroyAllWindows()
    return 0


img_raw = cv.imread("./pictures/25.bmp")
img_bgr = img_raw.copy()
img_lab = cv.cvtColor(img_bgr, cv.COLOR_BGR2LAB)
img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)

img = img_lab[:,:,1]
# img = cv.equalizeHist(img)
img = cv.GaussianBlur(img, (5,5),0)
_, img = cv.threshold(img, 0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)


plt.imshow(img,cmap="gray")
plt.show()
