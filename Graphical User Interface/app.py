import tkinter as tk
from tkinter import filedialog
from tkinter import Label, Button
from keras.preprocessing import image
from keras.models import model_from_json
import numpy as np

# Initialize main window
win = tk.Tk()

# Function to load model and make predictions
def bl_click():
    global path2
    try:
        # Load model architecture from JSON file
        with open('model.json', 'r') as json_file:
            loaded_model_json = json_file.read()
        
        # Load model from JSON
        loaded_model = model_from_json(loaded_model_json)
        
        # Load weights into the model
        loaded_model.load_weights("model.h5")

        # Load and preprocess the image
        test_image = image.load_img(path2, target_size=(128, 128))  # Ensure target_size matches model input
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)

        # Predict using the loaded model
        result = loaded_model.predict(test_image)
        
        # Define labels
        labels = [
            "Apple Apple_scab", "Apple Black rot", "Apple Cedar_apple_rust", "Apple Healthy",
            "Corn (maize) Cercospora_leaf_spot Gray_leaf_spot", "Corn (maize) Common rust", "Corn (maize) Healthy",
            "Corn (maize) Northern Leaf Blight", "Grape Black rot", "Grape Esca (Black_Measles)", "Grape Healthy",
            "Grape Leaf_blight_(Isariopsis_Leaf_Spot)", "Potato Early_blight", "Potato Healthy", "Potato Late_blight",
            "Tomato Early_blight", "Tomato Healthy", "Tomato Late_blight", "Tomato Septoria_leaf_spot",
            "Tomato Spider_mites Two-spotted_spider_mite", "Tomato Tomato_Yellow_Leaf_Curl_Virus", "Tomato Tomato_mosaic_virus"
        ]
        
        # Get the label with the highest probability
        max_prob_index = np.argmax(result)
        predicted_label = labels[max_prob_index]
        
        # Update label with the prediction result
        lbl.configure(text=predicted_label)
    
    except IOError as e:
        lbl.configure(text="Error loading model or image.")
        print(f"Error: {e}")

# Function to browse and select an image
def browse_image():
    global path2
    path2 = filedialog.askopenfilename()
    lbl.configure(text=f"Selected image: {path2}")

# Add widgets to the window
label1 = Label(win, text="GUI For Leaf Disease Detection using OpenCV", fg='blue')
label1.pack()

b1 = Button(win, text="Browse Image", width=25, height=3, fg='red', command=browse_image)
b1.pack()

bl = Button(win, text="Classify Image", width=25, height=3, fg='green', command=bl_click)
bl.pack()

lbl = Label(win, text="", fg='black')
lbl.pack()

# Set window properties
win.geometry("550x300")
win.title("Leaf Disease Detection using OpenCV")

# Start the Tkinter event loop
win.mainloop()
