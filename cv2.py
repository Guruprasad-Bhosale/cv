import cv2

img = cv2.imread('car.jpg',0)

small = cv2.resize(img,(200,200))
img[50:100,50:100] = 255
cv2.imshow('Image', small)

cv2.waitKey(0)

cv2.imwrite('car.jpg', img)

cv2.destroyAllWindows()
