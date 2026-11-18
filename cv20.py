
import cv2
import numpy as np

recognizer = cv2.face.EigenFaceRecognizer_create()

faces = []
labels = []

img1 = cv2.imread('face1.jpg', 0)
img2 = cv2.imread('face2.jpg', 0)

img1 = cv2.resize(img1, (200, 200))
img2 = cv2.resize(img2, (200, 200))
faces.append(img1)
faces.append(img2)

labels = np.array([0, 1])

recognizer.train(faces, labels)

label, confidence = recognizer.predict(img1)

print("Predicted Label:", label)
print("Confidence:", confidence)
