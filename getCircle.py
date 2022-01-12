import cv2 as cv
import numpy as np

img_bgr = cv.imread("./cv2/2017-06-22.png")
print(np.shape(img_bgr))
img_lab = cv.cvtColor(img_bgr, cv.COLOR_BGR2LAB)


img = img_bgr
cv.imshow("window", img)
k = cv.waitKey()
if k == 'q':
    cv.destroyAllWindows()
