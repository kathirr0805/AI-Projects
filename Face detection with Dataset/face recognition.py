import cv2
import numpy as np
import os

haar_file = 'haarcascade_frontalface_default.xml'  # Ensure the path is correct
datasets = 'datasets'  # Folder containing your dataset

print('Training...')

# Initialize variables
(images, labels, names, id) = ([], [], {}, 0)

# Walk through the dataset directory and load images and labels
for (subdirs, dirs, files) in os.walk(datasets):
    for subdir in dirs:
        names[id] = subdir  # Map subdir names to labels
        subjectpath = os.path.join(datasets, subdir)
        for filename in os.listdir(subjectpath):
            path = subjectpath + '/' + filename
            label = id
            images.append(cv2.imread(path, 0))  # Load image in grayscale
            labels.append(int(label))  # Store the label
        id += 1  # Increment id for the next person

# Convert lists to numpy arrays
(images, labels) = [np.array(lst) for lst in [images, labels]]

# Initialize the LBPHFaceRecognizer (instead of FisherFaceRecognizer)
model = cv2.face.LBPHFaceRecognizer_create()
model.train(images, labels)  # Train the model

# Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(haar_file)

# Initialize webcam
webcam = cv2.VideoCapture(0)  # Use 0 for default webcam
cnt = 0

while True:
    ret, im = webcam.read()
    if not ret:
        print("Failed to capture image")
        break

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # Detect faces

    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]  # Extract face region
        face_resize = cv2.resize(face, (130, 100))  # Resize face

        prediction = model.predict(face_resize)  # Predict using the trained model

        if prediction[1] < 80:
            cv2.putText(im, f'{names[prediction[0]]} - {prediction[1]:.0f}', (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
            print(f'Person: {names[prediction[0]]}, Confidence: {prediction[1]:.0f}')
            cnt = 0
        else:
            cnt += 1
            cv2.putText(im, 'Unknown', (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 2)
            if cnt > 100:
                print("Unknown Person")
                cv2.imwrite("unknown.jpg", im)  # Save the image of the unknown person
                cnt = 0

        # Draw rectangle around the face
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # Display the output
    cv2.imshow('OpenCV', im)

    # Press 'Esc' to exit
    if cv2.waitKey(10) & 0xFF == 27:
        break

# Release the webcam and destroy all windows
webcam.release()
cv2.destroyAllWindows()
