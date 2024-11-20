import imutils
import time
import cv2
import csv
import os

# Update the path to the Haar Cascade file
cascade = r'F:\Documents\My Notes\AI\Handson\Attendance System\haarcascade_frontalface_default.xml'

# Check if the cascade file exists at the given path
if not os.path.isfile(cascade):
    print(f"Error: Cascade file not found at {cascade}")
    exit()

# Load the cascade
detector = cv2.CascadeClassifier(cascade)

# Check if the detector is loaded correctly
if detector.empty():
    print("Error: Failed to load the cascade classifier")
    exit()

# Rest of the code remains the same
Name = input("Enter your Name: ")
Roll_Number = int(input("Enter your Roll Number: "))

# Dataset path
dataset = 'dataset'
sub_data = Name  # This will be the folder name

# Create the main 'dataset' directory if it doesn't exist
if not os.path.exists(dataset):
    os.mkdir(dataset)
    print(f"Created directory: {dataset}")

# Create the subdirectory for the user's name
path = os.path.join(dataset, sub_data)
if not os.path.isdir(path):
    os.mkdir(path)
    print(f"Created subdirectory: {path}")

# Save user information to CSV file
info = [str(Name), str(Roll_Number)]
with open('student.csv', 'a', newline='') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(info)

# Start video capture
print("Starting video stream...")
cam = cv2.VideoCapture(0)  # Change to 0 for default webcam
time.sleep(2.0)
total = 0

while total < 50:
    print(total)
    ret, frame = cam.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Resize the frame for faster processing
    img = imutils.resize(frame, width=400)
    
    # Detect faces in the grayscale frame
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangle around detected faces and save the image
    for (x, y, w, h) in rects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        p = os.path.sep.join([path, "{}.png".format(str(total).zfill(5))])
        cv2.imwrite(p, img)
        total += 1

    # Display the frame
    cv2.imshow("Frame", frame)
    
    # Break if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the camera and close windows
cam.release()
cv2.destroyAllWindows()
