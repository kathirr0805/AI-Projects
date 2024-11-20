import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json

# Load the dataset
dataset = np.loadtxt('diabetes.csv', delimiter=',')

# Split the dataset into input (X) and output (Y) variables
X = dataset[:, 0:8]
Y = dataset[:, 8]

# Print Input and Output if needed
# print("Input:", X)
# print("Output:", Y)

# Define the Keras model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the Keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Model Training
model.fit(X, Y, epochs=70, batch_size=10)

# Evaluate the model
accuracy = model.evaluate(X, Y)
print('Accuracy: %.2f' % (accuracy[1] * 100))

# Save the model to disk
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# Save the model weights
model.save_weights("model.weights.h5")

print("Saved model to disk")

# Save the entire model
model.save("model.h5")
