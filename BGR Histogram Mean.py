import cv2 as cv
import numpy as np
from cv2 import INTER_AREA
import matplotlib.pyplot as plt

# for image_number in range(1, 11):
    # image = cv.imread(f'Photos/Paprika/Paprika{image_number}.jpg')
    # image_resize = cv.resize(image, (900,900), interpolation=INTER_AREA)
    # image_to_grayscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # hist, bin_edges = np.histogram(image_to_grayscale, bins=5, range=(0, 255))
    # print(hist)

image = cv.imread(f'Photos/Paprika/Paprika6.jpg')
image_resize = cv.resize(image, (900,900), interpolation=INTER_AREA)
image_to_grayscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
image_histogram = cv.calcHist([image_to_grayscale], [0], None, [5], [0, 256])
hist, bin_edges = np.histogram(image_histogram, bins=5, range=(0, 255))

plt.figure()
plt.plot(bin_edges[0:-1], hist)
plt.xlabel("Bins")
plt.xlabel("No. of pixels")
plt.title("Numpy Histogram")
plt.show()

plt.plot(image_histogram)
plt.xlabel("Bins")
plt.xlabel("No. of pixels")
plt.title("OpenCV Histogram")
plt.show()
