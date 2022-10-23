import cv2 as cv
from cv2 import imshow
from cv2 import split
import numpy as np
from cv2 import INTER_AREA

for image_number in range(1, 11):
    image = cv.imread(f'image_dataset\Paprika\Paprika{image_number}.jpg')
    image_resize = cv.resize(image, (800,800), interpolation=INTER_AREA)

    hsv = cv.cvtColor(image_resize, cv.COLOR_BGR2HSV)
    lower_blue = np.array([90, 50, 70])
    upper_blue = np.array([128, 255, 255])

    blue_mask = cv.inRange(hsv, lower_blue, upper_blue)
    blue = cv.bitwise_and(image_resize, image_resize, mask=blue_mask)

    blue, green, red = cv.split(blue)
    blue_hist, blue_bin_edges = np.histogram(blue, bins=5, range=(0, 255))
    green_hist, green_bin_edges = np.histogram(green, bins=5, range=(0, 255))
    red_hist, red_bin_edges = np.histogram(red, bins=5, range=(0, 255))

    stack_hist = np.row_stack((blue_hist, green_hist, red_hist))
    mean_stack_hist = np.mean(stack_hist, axis=0)

    print(mean_stack_hist)
    # cv.imshow("Paprika Masked", grayscale_image)