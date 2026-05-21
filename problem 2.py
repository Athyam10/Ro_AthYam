import cv2
import numpy as np

# Load image
image = cv2.imread("image.jpeg")

# Check if image loaded
if image is None:
    print("Could not load image")
    exit()

# Show original image
cv2.imshow("Original Image", image)

while True:
    values = input(
        "Enter LOW and HIGH threshold values (example: 50 150) or q to quit: "
    )

    # Quit option
    if values.lower() == 'q':
        break

    try:
        low_threshold, high_threshold = map(int, values.split())
    except ValueError:
        print("Invalid input. Enter two numbers like: 50 150")
        continue

    # Step 1: Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 2: Noise reduction using Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Step 3 + 4:
    # Gradient calculation, non-maximum suppression,
    # and thresholding using Canny Edge Detection
    edges = cv2.Canny(blurred, low_threshold, high_threshold)

    # Show edge image
    cv2.imshow("Edge Detection", edges)

    cv2.waitKey(1)

# Close all windows
cv2.destroyAllWindows()