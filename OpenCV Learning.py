import cv2 as cv


def rescale_frame(frame, scale=0.2):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimension = (width, height)
    return cv.resize(frame, dimension, interpolation=cv.INTER_AREA)


# Methods for reading images
img = cv.imread("Photos/Paprika/Paprika1.jpg")
# resized_image = rescale_frame(img)
resized_image = cv.resize(img, (300, 250), interpolation=cv.INTER_AREA)
cv.imshow('Lays', resized_image)
cv.waitKey(0)


