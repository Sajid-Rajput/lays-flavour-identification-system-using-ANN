import cv2 as cv
from cv2 import INTER_AREA
import numpy as np
import matplotlib.pyplot as plt

for image_number in range(1, 11):
    image = cv.imread(f'Photos/Paprika/Paprika{image_number}.jpg')
    # image_resize = cv.resize(image, (300,300), interpolation=INTER_AREA)
    image_to_grayscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    hist, bin_edges = np.histogram(image_to_grayscale, bins=5, range=(0, 255))
    print(hist)

# plt.figure()
# plt.plot(bin_edges[0: -1], hist)
# plt.title("Grayscale Histogram")
# plt.xlabel("Bins")
# plt.ylabel("No. of Pixels")
# plt.show()
