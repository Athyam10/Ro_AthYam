import cv2
import numpy as np

# Load image
image = cv2.imread("image 3.jpeg")

# Check image
if image is None:
    print("Could not load image")
    exit()

# Create copies
output_all = image.copy()
output_separate = image.copy()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Blur to reduce noise
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# -------------------------
# DETECT LINES
# -------------------------
lines = cv2.HoughLinesP(
    blur,
    1,
    np.pi / 180,
    threshold=100,
    minLineLength=50,
    maxLineGap=10
)

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]

        # Draw on combined image
        cv2.line(output_all, (x1, y1), (x2, y2), (0, 255, 0), 3)

# -------------------------
# DETECT CIRCLES
# -------------------------
circles = cv2.HoughCircles(
    blur,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=30,
    param1=100,
    param2=30,
    minRadius=10,
    maxRadius=0
)

if circles is not None:
    circles = np.round(circles[0, :]).astype("int")

    for (x, y, r) in circles:

        # Draw circle
        cv2.circle(output_all, (x, y), r, (255, 0, 0), 3)

        # Draw center point
        cv2.circle(output_all, (x, y), 2, (0, 0, 255), 3)

# -------------------------
# DETECT RECTANGLES / SQUARES
# -------------------------
edges = cv2.Canny(blur, 50, 150)

contours, _ = cv2.findContours(
    edges,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

for contour in contours:

    # Approximate contour shape
    approx = cv2.approxPolyDP(
        contour,
        0.02 * cv2.arcLength(contour, True),
        True
    )

    # Rectangle or square has 4 points
    if len(approx) == 4:

        area = cv2.contourArea(approx)

        # Ignore tiny shapes
        if area > 500:

            cv2.drawContours(
                output_all,
                [approx],
                -1,
                (0, 255, 255),
                3
            )

# -------------------------
# SHOW RESULTS
# -------------------------

# OPTION 1:
# Everything in ONE window
cv2.imshow("All Shapes Detected", output_all)

# OPTION 2:
# Separate windows

# Original
cv2.imshow("Original Image", image)

# Separate line image
line_image = image.copy()

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 3)

cv2.imshow("Lines", line_image)

# Separate circle image
circle_image = image.copy()

if circles is not None:
    for (x, y, r) in circles:
        cv2.circle(circle_image, (x, y), r, (255, 0, 0), 3)
        cv2.circle(circle_image, (x, y), 2, (0, 0, 255), 3)

cv2.imshow("Circles", circle_image)

# Separate rectangle image
rectangle_image = image.copy()

for contour in contours:

    approx = cv2.approxPolyDP(
        contour,
        0.02 * cv2.arcLength(contour, True),
        True
    )

    if len(approx) == 4:

        area = cv2.contourArea(approx)

        if area > 500:

            cv2.drawContours(
                rectangle_image,
                [approx],
                -1,
                (0, 255, 255),
                3
            )

cv2.imshow("Rectangles / Squares", rectangle_image)

# Wait until key press
cv2.waitKey(0)

# Close windows
cv2.destroyAllWindows()