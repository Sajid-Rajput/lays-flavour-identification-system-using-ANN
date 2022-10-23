import cv2 as cv
from cv2 import imread
from cv2 import INTER_AREA
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = cv.imread("Photos/Paprika/Paprika1.jpg")
# img_resize = cv.resize(img, (300, 300), interpolation=INTER_AREA)
grayscale_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
histogram_grayscale = cv.calcHist([grayscale_img], [0], None, [256], [0, 255])

# plt.figure()
# plt.plot(histogram_grayscale)
# plt.title("GrayScale Histogram")
# plt.xlabel("Bins")
# plt.ylabel("No. of Pixels")
# plt.xlim([0, 255])
# plt.ylim([0, 7])
#
# # image = mpimg.imread("Photos/Paprika/Paprika1.jpg")
# # plt.imshow(image)
# plt.show()

print(histogram_grayscale)
cv.waitKey(0)
