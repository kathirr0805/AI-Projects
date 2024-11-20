import cv2
import pytesseract

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

# Read the image file
image = cv2.imread('car2.JPG')
cv2.imshow("Original", image)

# Convert to Grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray Image", gray_image)

# Canny Edge Detection
canny_edge = cv2.Canny(gray_image, 170, 200)
cv2.imshow("Canny Edge", canny_edge)

# Find contours based on edges
contours, _ = cv2.findContours(canny_edge.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

# Initialize license plate contour and x, y, w, h coordinates
contour_with_license_plate = None
license_plate = None
x = y = w = h = None

# Create a copy of the image to draw contours on
contour_image = image.copy()

# Draw contours on the image
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
cv2.imshow("Image with Contours", contour_image)

# Find the contour with 4 corners (assumed to be the license plate)
for contour in contours:
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.01 * perimeter, True)
    print("Contour Approximation Points:", len(approx))

    if len(approx) == 4:  # Looking for a quadrilateral shape
        contour_with_license_plate = approx
        x, y, w, h = cv2.boundingRect(contour)
        license_plate = gray_image[y:y + h, x:x + w]
        break

# If a license plate is found, proceed with further processing
if license_plate is not None:
    # Thresholding the license plate area
    _, license_plate = cv2.threshold(license_plate, 127, 255, cv2.THRESH_BINARY)
    cv2.imshow("License Plate Region", license_plate)

    # Noise reduction using a bilateral filter
    license_plate = cv2.bilateralFilter(license_plate, 11, 17, 17)

    # Text recognition using Tesseract
    text = pytesseract.image_to_string(license_plate)
    print("License Plate Text:", text)

    # Draw the bounding box and recognized text on the original image
    image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 3)
    image = cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Image with Detected License Plate", image)

# Wait for a key press and close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()
