from numpy import loadtxt
from keras.models import model_from_json

# Load the dataset
dataset = loadtxt('diabetes.csv', delimiter=',')
x = dataset[:, 0:8]
y = dataset[:, 8]

# Load the model from JSON file
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

# Load weights into the model
model = model_from_json(loaded_model_json)
model.load_weights("model.h5")
print("Loaded model from disk")

# Make predictions
predictions = model.predict(x)

# Display a few predictions
for i in range(10, 15):
    print('%s => Predicted: %.2f, Expected: %d' % (x[i].tolist(), predictions[i], y[i]))
