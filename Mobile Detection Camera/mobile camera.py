import urllib.request
import cv2
import numpy as np
import imutils

url = 'http://192.168.232.224:8080/shot.jpg'

while True:
    # Get the image from the URL
    imgPath = urllib.request.urlopen(url)
    
    # Convert the image to a numpy array
    imgNp = np.array(bytearray(imgPath.read()), dtype=np.uint8)
    
    # Decode the image array to OpenCV format
    frame = cv2.imdecode(imgNp, -1)
    
    # Resize the frame
    frame = imutils.resize(frame, width=450)
    
    # Display the frame in a window
    cv2.imshow("Frame", frame)
    
    # Exit when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources and close windows
cv2.destroyAllWindows()
