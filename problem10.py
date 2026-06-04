import cv2
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------
# Read image
# ---------------------------------------------------
image = cv2.imread("img3.jpeg")

if image is None:
    print("Image not found")
    exit()

original = image.copy()

# ---------------------------------------------------
# Convert to grayscale
# ---------------------------------------------------
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ---------------------------------------------------
# Gaussian blur
# ---------------------------------------------------
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# ---------------------------------------------------
# Edge detection
# ---------------------------------------------------
edges = cv2.Canny(blur, 50, 150)

# ---------------------------------------------------
# Region of interest
# ---------------------------------------------------
height, width = edges.shape

mask = np.zeros_like(edges)

polygon = np.array([[
    (0, height),
    (width, height),
    (width // 2, int(height * 0.5))
]], np.int32)

cv2.fillPoly(mask, polygon, 255)

cropped = cv2.bitwise_and(edges, mask)

# ---------------------------------------------------
# Hough Transform
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
# Separate left/right lane points
# ---------------------------------------------------
left_points = []
right_points = []

if lines is not None:

    for line in lines:

        x1, y1, x2, y2 = line.reshape(4)

        if x2 - x1 == 0:
            continue

        slope = (y2 - y1) / (x2 - x1)

        # Ignore horizontal lines
        if abs(slope) < 0.3:
            continue

        if slope < 0:
            left_points.extend([(x1, y1), (x2, y2)])
        else:
            right_points.extend([(x1, y1), (x2, y2)])

# ---------------------------------------------------
# Fit average lane using regression
# ---------------------------------------------------
def fit_lane(points, y_bottom, y_top):

    if len(points) < 2:
        return None

    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])

    # Linear regression
    fit = np.polyfit(y, x, 1)

    slope = fit[0]
    intercept = fit[1]

    x_bottom = int(slope * y_bottom + intercept)
    x_top = int(slope * y_top + intercept)

    return (x_bottom, y_bottom, x_top, y_top)


y_bottom = height
y_top = int(height * 0.55)

left_lane = fit_lane(left_points, y_bottom, y_top)
right_lane = fit_lane(right_points, y_bottom, y_top)

# ---------------------------------------------------
# Draw averaged lanes
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
# Overlay result
# ---------------------------------------------------
result = cv2.addWeighted(original, 0.8, lane_image, 1, 1)

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
plt.title("Average Lane Detection")
plt.axis("off")

plt.tight_layout()
plt.show()