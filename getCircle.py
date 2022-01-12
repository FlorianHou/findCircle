import cv2 as cv
import numpy as np

img_bgr = cv.imread("./pictures/25.bmp")
print(np.shape(img_bgr))
img_lab = cv.cvtColor(img_bgr, cv.COLOR_BGR2LAB)


img = img_bgr
cv.imshow("window", img)
k = cv.waitKey()
if k == 'q':
    cv.destroyAllWindows()
