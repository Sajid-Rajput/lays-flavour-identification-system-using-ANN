import cv2 as cv
from cv2 import imshow
from cv2 import split
import numpy as np
from cv2 import INTER_AREA

for image_number in range(1, 11):
    image = cv.imread(f'image_dataset\Paprika\Paprika{image_number}.jpg')
    image_resize = cv.resize(image, (800,800), interpolation=INTER_AREA)

    hsv = cv.cvtColor(image_resize, cv.COLOR_BGR2HSV)
    hsv = cv.cvtColor(image_resize, cv.COLOR_BGR2HSV)

    lower_orange = np.array([10, 50, 70])
    upper_orange = np.array([24, 255, 255])

    orange_mask = cv.inRange(hsv, lower_orange, upper_orange)
    orange = cv.bitwise_and(image_resize, image_resize, mask=orange_mask)

    blue_histogram = cv.calcHist([orange], [0], None, [5], [0, 255]).ravel()
    green_histogram = cv.calcHist([orange], [1], None, [5], [0, 255]).ravel()
    red_histogram = cv.calcHist([orange], [2], None, [5], [0, 255]).ravel()

    stack_hist = np.row_stack((blue_histogram, green_histogram, red_histogram))
    mean_stack_hist = np.mean(stack_hist, axis=0)

    print(mean_stack_hist)