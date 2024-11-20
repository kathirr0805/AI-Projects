import cv2
import imutils

# Define the lower and upper boundaries of the red color in HSV color space
redLower = (0, 0, 241)
redUpper = (179, 124, 255)

# Initialize the camera (use 0 if 1 doesn't work)
camera = cv2.VideoCapture(0)

while True:
    grabbed, frame = camera.read()  # Read the frame
    if not grabbed:
        print("Error reading frame. Exiting...")
        break

    frame = imutils.resize(frame, width=1000)  # Resize the frame
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)  # Blur the frame
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)  # Convert to HSV color space

    # Mask the red color
    mask = cv2.inRange(hsv, redLower, redUpper)
    mask = cv2.erode(mask, None, iterations=2)  # Erode the mask
    mask = cv2.dilate(mask, None, iterations=2)  # Dilate the mask

    # Find contours in the mask
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[-2]
    center = None

    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)  # Find the largest contour
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        if radius > 10:
            # Draw the circle and centroid on the frame
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
            print(center, radius)

            # Control logic based on the position and radius of the detected object
            if radius > 250:
                print("Stop")
            else:
                if center[0] < 150:
                    print("Right")
                elif center[0] > 450:
                    print("Left")
                elif radius < 250:
                    print("Front")
                else:
                    print("Stop")
        else:
            print("Object too small")

    cv2.imshow("Frame", frame)  # Show the frame

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):  # Exit the loop if 'q' is pressed
        break

# Release the camera and close windows
camera.release()
cv2.destroyAllWindows()
