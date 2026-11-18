
import cv2

face_cascade = cv2.CascadeClassifier('G:\Edrive\FInal Year\CV\haar_cascade.xml')

img = cv2.imread('Dune.jpeg')
img2 = cv2.imread('Face.jpeg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1)
faces2 = face_cascade.detectMultiScale(gray2, scaleFactor=1.1, minNeighbors=10)

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
for (x, y, w, h) in faces2:
    cv2.rectangle(img2, (x, y), (x + w, y + h), (255, 0, 0), 2)

cv2.imshow('img', img)
cv2.imshow('img2', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
