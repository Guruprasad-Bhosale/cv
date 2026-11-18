import cv2

img = cv2.imread('car.jpg')

cv2.imshow('Image',img)

cv2.waitKey(0)
cv2.imwrite('car_copy.jpg', img)

cv2.destroyAllWindows()