import cv2 as cv
import numpy as np

def apply_low_pass_filter(img):
	img = cv.filter2D(img, -1, kernel)
	cv.imshow("Low-pass Filter", img)
	cv.waitKey()
	return img