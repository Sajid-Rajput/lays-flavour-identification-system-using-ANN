import cv2 as cv
from cv2 import imshow
import numpy as np
import matplotlib.pyplot as plt
from cv2 import INTER_AREA

for image_number in range(1, 11):
    image = cv.imread(f'Photos/Paprika/Paprika{image_number}.jpg')
    image_resize = cv.resize(image, (800,800), interpolation=INTER_AREA)

    hsv = cv.cvtColor(image_resize, cv.COLOR_BGR2HSV)
    lower_blue = np.array([90, 50, 70])
    upper_blue = np.array([128, 255, 255])

    blue_mask = cv.inRange(hsv, lower_blue, upper_blue)
    blue = cv.bitwise_and(image_resize, image_resize, mask=blue_mask)

    grayscale_image = cv.cvtColor(blue, cv.COLOR_BGR2GRAY)

    hist, bin_edges = np.histogram(grayscale_image, bins=5, range=(0, 255))

    print(hist)
    # cv.imshow("Paprika Masked", grayscale_image)
    # cv.waitKey(0)
