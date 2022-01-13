import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


img_raw = cv.imread("./pictures/25.bmp")
img_bgr = img_raw.copy()
img_lab = cv.cvtColor(img_bgr, cv.COLOR_BGR2LAB)
img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)


img = img_lab[:,:,1]
# img = cv.equalizeHist(img)
img = cv.GaussianBlur(img, (11,11),0)
_, img = cv.threshold(img, 160,255,cv.THRESH_BINARY)


plt.imshow(img,cmap="gray")
plt.show()
