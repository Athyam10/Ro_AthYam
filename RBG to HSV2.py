import cv2
import numpy as np

# Load original image
image = cv2.imread("image.jpeg")

# Check image
if image is None:
    print("Could not load image")
    exit()

# Show original image once
cv2.imshow("Original Image", image)

while True:
    values = input(
        "Enter brightness, saturation, contrast percentages (-100 to 100) separated by spaces, or q to quit: "
    )

    if values.lower() == 'q':
        break

    try:
        brightness_str, saturation_str, contrast_str = values.split()
        brightness = int(brightness_str)
        saturation = int(saturation_str)
        contrast = int(contrast_str)
    except ValueError:
        print("Invalid input. Please enter three integer values or q.")
        continue

    if not all(-100 <= v <= 100 for v in (brightness, saturation, contrast)):
        print("Values must be between -100 and 100.")
        continue

    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Split channels
    h, s, v = cv2.split(hsv)

    # Saturation adjustment
    sat_factor = 1.0 + saturation / 100.0
    s = np.clip(s.astype(np.float32) * sat_factor, 0, 255)

    # Brightness adjustment
    v = v.astype(np.float32) + brightness * 255.0 / 100.0

    # Contrast adjustment around mid-point
    contrast_factor = 1.0 + contrast / 100.0
    v = (v - 128.0) * contrast_factor + 128.0

    # Clip and convert back to uint8
    s = np.clip(s, 0, 255).astype(np.uint8)
    v = np.clip(v, 0, 255).astype(np.uint8)

    # Merge HSV
    final_hsv = cv2.merge((h, s, v))

    # HSV -> BGR
    adjusted_image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    cv2.imshow("Adjusted Image", adjusted_image)
    cv2.waitKey(1)

# Close windows
cv2.destroyAllWindows()