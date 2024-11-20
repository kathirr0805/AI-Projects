import cv2

# Load the Haar Cascade for face detection
alg = "haarcascade_frontalface_default.xml"
haar_cascade = cv2.CascadeClassifier(alg)

# Uncomment the lines below if you want to work with a video instead of a single image

# Replace 'check.mp4' with the path to your video file
video_path = "video.mp4"
cam = cv2.VideoCapture(video_path)

# Uncomment the lines below if you want to work with a single image
# image_path = "pic.jpg"
# img = cv2.imread(image_path)

# # Check if the image was loaded correctly
# if img is None:
#     print("Error loading image. Exiting...")
#     exit()

#For processing a video (uncomment and use instead of the image code):
while True:
    ret, img = cam.read()
    if not ret:
        print("Error reading video file. Exiting...")
        break

# Convert the image to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces in the image or video frame
faces = haar_cascade.detectMultiScale(gray_img, scaleFactor=1.3, minNeighbors=4)

# Draw rectangles around detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (155, 155, 255), 2)

# Display the image or video frame with detected faces
cv2.imshow("Face Detection", img)

# Wait for a key press and close the window
# For video processing, use cv2.waitKey(1) instead of cv2.waitKey(0)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Uncomment the lines below if you are using a video to release resources
# cam.release()
