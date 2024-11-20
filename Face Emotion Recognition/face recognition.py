from facial_emotion_recognition import EmotionRecognition
import cv2

# Initialize EmotionRecognition object with CPU
er = EmotionRecognition(device='cpu')

# Initialize the webcam (use 0 for default webcam, change to 1 if using an external camera)
cam = cv2.VideoCapture(0)

while True:
    # Read the frame from the webcam
    success, frame = cam.read()
    
    if not success:
        print("Failed to capture image")
        break

    # Recognize emotion in the frame
    frame = er.recognise_emotion(frame, return_type='BGR')

    # Display the frame with emotion recognition
    cv2.imshow("Frame", frame)

    # Check if 'Esc' key is pressed to exit
    key = cv2.waitKey(1)
    if key == 27:  # 27 is the ASCII code for the 'Esc' key
        break

# Release the webcam and close all OpenCV windows
cam.release()
cv2.destroyAllWindows()
