import cv2

alg = "haarcascade_frontalface_default.xml"
haar_cascade = cv2.CascadeClassifier(alg)  # Loading algorithm

cam = cv2.VideoCapture(0)  # Cam id initialization

while True:
    ret, img = cam.read()  # Reading the frame from cam
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Converting color image to gray

    face = haar_cascade.detectMultiScale(grayImg, 1.3, 4)  # Getting coordinates

    for (x, y, w, h) in face:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Drawing rectangle around face

    cv2.imshow("Face Detection", img)

    key = cv2.waitKey(10)
    if key == 27:  # Press 'Esc' to exit
        break

cam.release()
cv2.destroyAllWindows()
