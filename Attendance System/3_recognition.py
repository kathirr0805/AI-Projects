from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle

# Paths to embedding, recognizer, and label encoder files
embeddingFile = "output/embeddings.pickle"  # Pre-saved face embeddings
recognizerFile = "output/recognizer.pickle"  # File to save the trained recognizer
labelEncFile = "output/le.pickle"  # File to save the label encoder

# Load the facial embeddings from the embeddings pickle file
print("Loading face embeddings...")
with open(embeddingFile, "rb") as f:
    data = pickle.load(f)

# Encode the labels (names) into numerical format
print("Encoding labels...")
labelEnc = LabelEncoder()
labels = labelEnc.fit_transform(data["names"])

# Train the Support Vector Classifier (SVC) on the embeddings
print("Training model...")
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(data["embeddings"], labels)

# Save the trained recognizer model to disk
with open(recognizerFile, "wb") as f:
    f.write(pickle.dumps(recognizer))

# Save the label encoder to disk
with open(labelEncFile, "wb") as f:
    f.write(pickle.dumps(labelEnc))

print(f"Model and label encoder saved to {recognizerFile} and {labelEncFile} respectively.")
