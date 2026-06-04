import cv2
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------
# Load image
# ---------------------------------------------------
image = cv2.imread("img2.jpeg")

if image is None:
    print("Image not found")
    exit()

original = image.copy()

# ---------------------------------------------------
# Convert to grayscale
# ---------------------------------------------------
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ---------------------------------------------------
# Apply Gaussian Blur
# ---------------------------------------------------
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# ---------------------------------------------------
# Edge Detection
# ---------------------------------------------------
edges = cv2.Canny(blur, 50, 150)

# ---------------------------------------------------
# Define Region of Interest
# ---------------------------------------------------
height, width = edges.shape

mask = np.zeros_like(edges)

polygon = np.array([[
    (0, height),
    (width, height),
    (width // 2, int(height * 0.55))
]], np.int32)

cv2.fillPoly(mask, polygon, 255)

cropped = cv2.bitwise_and(edges, mask)

# ---------------------------------------------------
# Detect line segments using Hough Transform
# ---------------------------------------------------
lines = cv2.HoughLinesP(
    cropped,
    rho=2,
    theta=np.pi / 180,
    threshold=50,
    minLineLength=40,
    maxLineGap=20
)

# ---------------------------------------------------
# Separate left and right lane points
# ---------------------------------------------------
left_points = []
right_points = []

if lines is not None:

    for line in lines:

        x1, y1, x2, y2 = line.reshape(4)

        # Avoid division by zero
        if x2 - x1 == 0:
            continue

        slope = (y2 - y1) / (x2 - x1)

        # Ignore near-horizontal lines
        if abs(slope) < 0.3:
            continue

        # Ignore extreme slopes
        if abs(slope) > 10:
            continue

        # Separate lanes
        if slope < 0:
            left_points.extend([(x1, y1), (x2, y2)])
        else:
            right_points.extend([(x1, y1), (x2, y2)])

# ---------------------------------------------------
# Extrapolate line to fixed field-of-view points
# ---------------------------------------------------
def extrapolate_lane(points, y_bottom, y_top, width):

    if len(points) < 2:
        return None

    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])

    # Linear regression
    fit = np.polyfit(y, x, 1)

    slope = fit[0]
    intercept = fit[1]

    # Safeguard against unstable slopes
    if abs(slope) < 0.001:
        return None

    # Calculate endpoints
    x_bottom = int(slope * y_bottom + intercept)
    x_top = int(slope * y_top + intercept)

    # Boundary checking
    x_bottom = max(0, min(width - 1, x_bottom))
    x_top = max(0, min(width - 1, x_top))

    return (x_bottom, y_bottom, x_top, y_top)

# ---------------------------------------------------
# Fixed extrapolation points
# ---------------------------------------------------
y_bottom = height
y_top = int(height * 0.55)

left_lane = extrapolate_lane(
    left_points,
    y_bottom,
    y_top,
    width
)

right_lane = extrapolate_lane(
    right_points,
    y_bottom,
    y_top,
    width
)

# ---------------------------------------------------
# Draw extrapolated lanes
# ---------------------------------------------------
lane_image = np.zeros_like(image)

if left_lane is not None:

    cv2.line(
        lane_image,
        (left_lane[0], left_lane[1]),
        (left_lane[2], left_lane[3]),
        (0, 255, 0),
        5
    )

if right_lane is not None:

    cv2.line(
        lane_image,
        (right_lane[0], right_lane[1]),
        (right_lane[2], right_lane[3]),
        (0, 255, 0),
        5
    )

# ---------------------------------------------------
# Overlay lanes on original image
# ---------------------------------------------------
result = cv2.addWeighted(
    original,
    0.8,
    lane_image,
    1,
    1
)

# ---------------------------------------------------
# Display results
# ---------------------------------------------------
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(edges, cmap='gray')
plt.title("Edge Detection")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(cropped, cmap='gray')
plt.title("Region of Interest")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title("Extrapolated Lane Lines")
plt.axis("off")

plt.tight_layout()
plt.show()