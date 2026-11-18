import cv2

cap = cv2.VideoCapture('car.mp4')
fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()

    if not ret:
        break
    fgmask = fgbg.apply(frame)
    fgmask = cv2.medianBlur(fgmask, 5)

    cv2.imshow('Foreground Mask', fgmask)

    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()