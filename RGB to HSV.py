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

    # Ask brightness
    brightness = input(
        "Enter brightness percentage (-100 to 100) or q to quit: "
    )

    # Exit
    if brightness.lower() == 'q':
        break

    brightness = int(brightness)

    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Split channels
    h, s, v = cv2.split(hsv)

    # Brightness adjustment
    if brightness > 0:
        v = np.clip(
            v + (brightness * 255 / 100),
            0,
            255
        ) - 40

    else:
        v = np.clip(
            v + (brightness * 255 / 100),
            0,
            255
        )

    # Convert back to uint8
    v = v.astype(np.uint8)

    # Merge HSV
    final_hsv = cv2.merge((h, s, v))

    # HSV -> BGR
    brightened_image = cv2.cvtColor(
        final_hsv,
        cv2.COLOR_HSV2BGR
    )

    # Update SAME edited window
    cv2.imshow(
        "Brightness Changed Image",
        brightened_image
    )

    # Small refresh
    cv2.waitKey(1)

# Close windows
cv2.destroyAllWindows()