# ============================================================
# SMART ROAD COUNTING + ROAD TYPE DETECTION
# ============================================================
#
# THIS VERSION FIXES:
#
# ❌ Old problem:
# Always detecting 2 roads
#
# ✅ New version:
#
# Automatically detects:
#
# - 1 road
# - 2 roads
#
# by checking:
#
# 1. Road boundary spacing
# 2. Divider width
# 3. Number of major vertical boundaries
#
# ============================================================

import cv2
import numpy as np

# ------------------------------------------------------------
# LOAD IMAGE
# ------------------------------------------------------------

image = cv2.imread("img2.jpeg")

if image is None:
    print("Image not found")
    exit()

original = image.copy()

height, width = image.shape[:2]

# ------------------------------------------------------------
# CONVERT TO GRAYSCALE
# ------------------------------------------------------------

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ------------------------------------------------------------
# BLUR
# ------------------------------------------------------------

blur = cv2.GaussianBlur(gray, (5, 5), 0)

# ------------------------------------------------------------
# EDGE DETECTION
# ------------------------------------------------------------

edges = cv2.Canny(blur, 50, 150)

# ------------------------------------------------------------
# REGION OF INTEREST
# ------------------------------------------------------------

mask = np.zeros_like(edges)

polygon = np.array([[
    
    (0, height),

    (width // 2 - width // 4, height // 2),

    (width // 2 + width // 4, height // 2),

    (width, height)

]], np.int32)

cv2.fillPoly(mask, [polygon], 255)

roi = cv2.bitwise_and(edges, mask)

# ------------------------------------------------------------
# HOUGH LINE TRANSFORM
# ------------------------------------------------------------

lines = cv2.HoughLinesP(
    roi,
    rho=2,
    theta=np.pi / 180,
    threshold=80,
    minLineLength=100,
    maxLineGap=50
)

# ------------------------------------------------------------
# STORE VALID ROAD LINES
# ------------------------------------------------------------

valid_lines = []

if lines is not None:

    for line in lines:

        x1, y1, x2, y2 = line[0]

        # Avoid divide by zero
        if x2 - x1 == 0:
            continue

        slope = (y2 - y1) / (x2 - x1)

        # Ignore horizontal lines
        if abs(slope) < 0.4:
            continue

        valid_lines.append((x1, y1, x2, y2, slope))

# ------------------------------------------------------------
# SORT LINES BASED ON X POSITION
# ------------------------------------------------------------

line_positions = []

for line in valid_lines:

    x1, y1, x2, y2, slope = line

    # Use bottom x position
    if y1 > y2:
        x_pos = x1
    else:
        x_pos = x2

    line_positions.append(x_pos)

# Sort from left to right
line_positions = sorted(line_positions)

# ------------------------------------------------------------
# REMOVE DUPLICATE NEARBY LINES
# ------------------------------------------------------------
#
# Many Hough lines overlap.
# We keep only major boundaries.
#
# ------------------------------------------------------------

major_boundaries = []

for x in line_positions:

    # First boundary
    if len(major_boundaries) == 0:
        major_boundaries.append(x)

    else:

        # Distance from previous boundary
        distance = abs(x - major_boundaries[-1])

        # Keep only clearly separated boundaries
        if distance > 80:
            major_boundaries.append(x)

# ------------------------------------------------------------
# DETERMINE NUMBER OF ROADS
# ------------------------------------------------------------
#
# LOGIC:
#
# 2 major boundaries:
# -> Single road
#
# 4 major boundaries:
# -> Double road
#
# ------------------------------------------------------------

number_of_roads = 0
road_type = "UNKNOWN"

boundary_count = len(major_boundaries)

# ------------------------------------------------------------
# SINGLE ROAD
# ------------------------------------------------------------

if boundary_count >= 2 and boundary_count < 4:

    number_of_roads = 1
    road_type = "SINGLE ROAD"

# ------------------------------------------------------------
# DOUBLE ROAD
# ------------------------------------------------------------

elif boundary_count >= 4:

    # Calculate spacing between boundaries
    gaps = []

    for i in range(len(major_boundaries) - 1):

        gap = major_boundaries[i + 1] - major_boundaries[i]

        gaps.append(gap)

    # --------------------------------------------------------
    # CHECK FOR DIVIDER PATTERN
    # --------------------------------------------------------
    #
    # Example:
    #
    # Road width = large
    # Divider width = small
    # Road width = large
    #
    # --------------------------------------------------------

    if len(gaps) >= 3:

        road1 = gaps[0]
        divider = gaps[1]
        road2 = gaps[2]

        # Divider should be smaller
        if divider < road1 * 0.6 and divider < road2 * 0.6:

            number_of_roads = 2
            road_type = "DOUBLE ROAD"

        else:

            number_of_roads = 1
            road_type = "WIDE SINGLE ROAD"

    else:

        number_of_roads = 1
        road_type = "SINGLE ROAD"

# ------------------------------------------------------------
# FALLBACK
# ------------------------------------------------------------

else:

    number_of_roads = 1
    road_type = "SINGLE ROAD"

# ------------------------------------------------------------
# CREATE OUTPUT IMAGE
# ------------------------------------------------------------

final = original.copy()

# ------------------------------------------------------------
# DRAW DETECTED ROAD LINES
# ------------------------------------------------------------

for line in valid_lines:

    x1, y1, x2, y2, slope = line

    # Left lines
    if slope < 0:
        color = (0, 255, 0)

    # Right lines
    else:
        color = (255, 0, 0)

    cv2.line(
        final,
        (x1, y1),
        (x2, y2),
        color,
        3
    )

# ------------------------------------------------------------
# DRAW MAJOR BOUNDARIES
# ------------------------------------------------------------

for x in major_boundaries:

    cv2.line(
        final,
        (x, height),
        (x, height - 200),
        (0, 255, 255),
        5
    )

# ------------------------------------------------------------
# DISPLAY ROAD INFORMATION
# ------------------------------------------------------------

cv2.putText(
    final,
    f"ROAD TYPE: {road_type}",
    (40, 50),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 0, 255),
    3
)

cv2.putText(
    final,
    f"NUMBER OF ROADS: {number_of_roads}",
    (40, 100),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (255, 0, 0),
    3
)

cv2.putText(
    final,
    f"BOUNDARIES DETECTED: {boundary_count}",
    (40, 150),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 0),
    3
)

# ------------------------------------------------------------
# SHOW RESULTS
# ------------------------------------------------------------

cv2.imshow("Edges", edges)

cv2.imshow("ROI", roi)

cv2.imshow("Road Detection", final)

cv2.waitKey(0)

cv2.destroyAllWindows()

# ============================================================
# HOW THIS FIX WORKS
# ============================================================
#
# OLD VERSION:
#
# If divider detected:
# -> Always 2 roads
#
#
# NEW VERSION:
#
# It checks:
#
# 1. Number of major boundaries
#
# 2. Width pattern:
#
#    Large Gap
#    Small Gap
#    Large Gap
#
#    = Double Road
#
#
# Example:
#
# |------ROAD------|
#
# -> 1 road
#
#
# Example:
#
# |---ROAD---||DIVIDER||---ROAD---|
#
# -> 2 roads
#
# ============================================================



# ============================================================
# TERMINAL COMMANDS
# ============================================================
#
# Install libraries:
#
# pip install opencv-python numpy
#
#
# Run:
#
# python lane_detection.py
#
# ============================================================