from imutils import paths
import numpy as np
import imutils
import pickle
import cv2
import os

# Set up paths for dataset, embeddings, and models
dataset = "dataset"
embeddingFile = "output/embeddings.pickle"  # File to save embeddings
embeddingModel = "openface_nn4.small2.v1.t7"  # Pre-trained PyTorch model for embedding
prototxt = "model/deploy.prototxt"  # Path to Caffe prototxt file for face detection
model = "model/res10_300x300_ssd_iter_140000.caffemodel"  # Caffe model for face detection

# Minimum confidence threshold to consider a detection valid
conf = 0.5

# Load Caffe model for face detection
detector = cv2.dnn.readNetFromCaffe(prototxt, model)

# Load PyTorch model for facial embeddings
embedder = cv2.dnn.readNetFromTorch(embeddingModel)

# Get all image paths from the dataset
imagePaths = list(paths.list_images(dataset))

# Initialize lists to store extracted names and embeddings
knownEmbeddings = []
knownNames = []
total = 0

# Process each image in the dataset
for (i, imagePath) in enumerate(imagePaths):
    print(f"Processing image {i + 1}/{len(imagePaths)}")

    # Extract the person's name from the image path
    name = imagePath.split(os.path.sep)[-2]

    # Load the image and resize it
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]

    # Prepare the image as a blob for face detection
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
    )

    # Detect faces in the image
    detector.setInput(imageBlob)
    detections = detector.forward()

    # Ensure at least one face was found
    if len(detections) > 0:
        # Get the detection with the highest confidence
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections based on confidence
        if confidence > conf:
            # Compute the (x, y) coordinates of the bounding box for the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Extract the face ROI
            face = image[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            # Ensure the face ROI is sufficiently large
            if fW < 20 or fH < 20:
                continue

            # Prepare the face ROI as a blob for the embedding model
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)

            # Get the 128-d facial embedding
            vec = embedder.forward()

            # Append the embedding and name to the lists
            knownNames.append(name)
            knownEmbeddings.append(vec.flatten())
            total += 1

            print(f"Processed face embedding {total}")

# Save the embeddings and corresponding names to disk
data = {"embeddings": knownEmbeddings, "names": knownNames}
with open(embeddingFile, "wb") as f:
    f.write(pickle.dumps(data))

print(f"Saved {total} face embeddings to {embeddingFile}")
