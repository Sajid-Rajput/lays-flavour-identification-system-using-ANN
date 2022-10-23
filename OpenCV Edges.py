import cv2 as cv
import matplotlib.pyplot as plt


img = cv.imread("Photos/Paprika/Paprika10.jpg")
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# gray_hist = cv.calcHist([gray], [0], None, [256], [0, 256])


# plt.figure()
# plt.title("GrayScale Histogram")
# plt.xlabel("Bins")
# plt.ylabel("No. of pixels")
# plt.plot(gray_hist)
# plt.xlim([0, 256])
# plt.show()
#
#
plt.figure()
plt.title("BGR Histogram")
plt.xlabel("Bins")
plt.ylabel("No. of pixels")
# colors = ('b', 'g', 'r')

# for i, col in enumerate(colors):
hist = cv.calcHist(img, [3], None, [256], [0, 256])
plt.plot(hist, color='r')
plt.xlim([0, 256])

plt.show()

cv.waitKey(0)

