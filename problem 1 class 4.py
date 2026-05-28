# ============================================================
# REFINED UNIVERSAL ROAD EDGE + LANE DETECTION
# ============================================================
#
# THIS VERSION FIXES:
#
# ❌ Detecting random edges
# ❌ Detecting trees/grass
# ❌ Wrong road count
# ❌ Weak lane detection
#
# ✅ NEW FEATURES:
#
# 1. Detects ONLY:
#       - Outer road edges
#       - Important lane markings
#
# 2. Automatically detects:
#       - Single road
#       - Double road
#
# 3. Uses:
#       - Perspective-based ROI
#       - Strong edge filtering
#       - Slope filtering
#       - Position filtering
#       - Line grouping
#
# 4. Detects:
#       - LEFT TURN
#       - RIGHT TURN
#       - STRAIGHT
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
# CONVERT TO HSV
# ------------------------------------------------------------
#
# HSV helps isolate white road markings
#
# ------------------------------------------------------------

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# ------------------------------------------------------------
# DETECT WHITE ROAD LINES
# ------------------------------------------------------------
#
# Keeps only bright white markings
#
# ------------------------------------------------------------

lower_white = np.array([0, 0, 170])
upper_white = np.array([255, 80, 255])

white_mask = cv2.inRange(
    hsv,
    lower_white,
    upper_white
)

# ------------------------------------------------------------
# BLUR
# ------------------------------------------------------------

blur = cv2.GaussianBlur(white_mask, (5, 5), 0)

# ------------------------------------------------------------
# EDGE DETECTION
# ------------------------------------------------------------

edges = cv2.Canny(blur, 50, 150)

# ------------------------------------------------------------
# REGION OF INTEREST
# ------------------------------------------------------------
#
# ONLY focus on road region
#
# ------------------------------------------------------------

mask = np.zeros_like(edges)

polygon = np.array([[
    
    # Bottom left
    (0, height),

    # Top left
    (width // 2 - 120, height // 2 + 50),

    # Top right
    (width // 2 + 120, height // 2 + 50),

    # Bottom right
    (width, height)

]], np.int32)

cv2.fillPoly(mask, [polygon], 255)

roi = cv2.bitwise_and(edges, mask)

# ------------------------------------------------------------
# HOUGH LINE TRANSFORM
# ------------------------------------------------------------

lines = cv2.HoughLinesP(
    roi,
    rho=1,
    theta=np.pi / 180,
    threshold=60,
    minLineLength=120,
    maxLineGap=40
)

# ------------------------------------------------------------
# STORE VALID LINES
# ------------------------------------------------------------

valid_lines = []

# ------------------------------------------------------------
# FILTER LINES
# ------------------------------------------------------------
#
# Keeps ONLY road-like lines
#
# ------------------------------------------------------------

if lines is not None:

    for line in lines:

        x1, y1, x2, y2 = line[0]

        # Avoid divide by zero
        if x2 - x1 == 0:
            continue

        slope = (y2 - y1) / (x2 - x1)

        # ----------------------------------------------------
        # REMOVE HORIZONTAL LINES
        # ----------------------------------------------------

        if abs(slope) < 0.5:
            continue

        # ----------------------------------------------------
        # REMOVE EXTREMELY STEEP LINES
        # ----------------------------------------------------

        if abs(slope) > 5:
            continue

        # ----------------------------------------------------
        # KEEP ONLY LOWER IMAGE LINES
        # ----------------------------------------------------

        if y1 < height * 0.45 or y2 < height * 0.45:
            continue

        # ----------------------------------------------------
        # STORE VALID ROAD LINE
        # ----------------------------------------------------

        valid_lines.append((x1, y1, x2, y2, slope))

# ------------------------------------------------------------
# SORT LINES BY POSITION
# ------------------------------------------------------------

line_positions = []

for line in valid_lines:

    x1, y1, x2, y2, slope = line

    # Use bottom-most x coordinate
    if y1 > y2:
        bottom_x = x1
    else:
        bottom_x = x2

    line_positions.append(bottom_x)

line_positions = sorted(line_positions)

# ------------------------------------------------------------
# GROUP SIMILAR LINES
# ------------------------------------------------------------
#
# Removes duplicate nearby detections
#
# ------------------------------------------------------------

major_lines = []

for x in line_positions:

    if len(major_lines) == 0:

        major_lines.append(x)

    else:

        distance = abs(x - major_lines[-1])

        # Only keep separated lines
        if distance > 70:

            major_lines.append(x)

# ------------------------------------------------------------
# DETERMINE ROAD COUNT
# ------------------------------------------------------------

number_of_roads = 1
road_type = "SINGLE ROAD"

# ------------------------------------------------------------
# DOUBLE ROAD DETECTION
# ------------------------------------------------------------
#
# Pattern:
#
# Large gap
# Small divider gap
# Large gap
#
# ------------------------------------------------------------

if len(major_lines) >= 4:

    gaps = []

    for i in range(len(major_lines) - 1):

        gap = major_lines[i + 1] - major_lines[i]

        gaps.append(gap)

    if len(gaps) >= 3:

        road1 = gaps[0]
        divider = gaps[1]
        road2 = gaps[2]

        # Divider smaller than roads
        if divider < road1 * 0.6 and divider < road2 * 0.6:

            number_of_roads = 2
            road_type = "DOUBLE ROAD"

# ------------------------------------------------------------
# DETECT ROAD DIRECTION
# ------------------------------------------------------------

direction = "STRAIGHT"

if len(valid_lines) > 0:

    bottom_points = []
    top_points = []

    for line in valid_lines:

        x1, y1, x2, y2, slope = line

        # Bottom point
        if y1 > y2:

            bottom_points.append(x1)
            top_points.append(x2)

        else:

            bottom_points.append(x2)
            top_points.append(x1)

    avg_bottom = np.mean(bottom_points)
    avg_top = np.mean(top_points)

    shift = avg_top - avg_bottom

    # --------------------------------------------------------
    # DETERMINE TURN
    # --------------------------------------------------------

    if shift < -25:

        direction = "LEFT TURN"

    elif shift > 25:

        direction = "RIGHT TURN"

    else:

        direction = "STRAIGHT"

# ------------------------------------------------------------
# DRAW FINAL RESULTS
# ------------------------------------------------------------

final = original.copy()

# ------------------------------------------------------------
# DRAW DETECTED ROAD LINES
# ------------------------------------------------------------

for line in valid_lines:

    x1, y1, x2, y2, slope = line

    # Left-leaning lines
    if slope < 0:
        color = (0, 255, 0)

    # Right-leaning lines
    else:
        color = (255, 0, 0)

    cv2.line(
        final,
        (x1, y1),
        (x2, y2),
        color,
        4
    )

# ------------------------------------------------------------
# DRAW MAJOR ROAD BOUNDARIES
# ------------------------------------------------------------

for x in major_lines:

    cv2.line(
        final,
        (x, height),
        (x, height - 250),
        (0, 255, 255),
        5
    )

# ------------------------------------------------------------
# DISPLAY INFORMATION
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
    f"DIRECTION: {direction}",
    (40, 150),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 0),
    3
)

# ------------------------------------------------------------
# SHOW RESULTS
# ------------------------------------------------------------

cv2.imshow("White Line Mask", white_mask)

cv2.imshow("Edges", edges)

cv2.imshow("ROI", roi)

cv2.imshow("Refined Road Detection", final)

cv2.waitKey(0)

cv2.destroyAllWindows()

# ============================================================
# WHY THIS VERSION WORKS BETTER
# ============================================================
#
# 1. HSV WHITE FILTERING
# ------------------------------------------------------------
# Detects ONLY white lane markings.
#
# Removes:
# - Grass
# - Trees
# - Sky
# - Shadows
#
#
# 2. ROI FILTERING
# ------------------------------------------------------------
# Only detects lower road area.
#
#
# 3. SLOPE FILTERING
# ------------------------------------------------------------
# Removes:
# - Horizontal edges
# - Unrealistic steep edges
#
#
# 4. POSITION FILTERING
# ------------------------------------------------------------
# Removes upper-image detections.
#
#
# 5. LINE GROUPING
# ------------------------------------------------------------
# Prevents duplicate lane detections.
#
#
# 6. SMART ROAD COUNTING
# ------------------------------------------------------------
#
# Single road:
#
# |-----------ROAD-----------|
#
#
# Double road:
#
# |---ROAD---||DIVIDER||---ROAD---|
#
# ============================================================



# ============================================================
# TERMINAL COMMANDS
# ============================================================
#
# Install:
#
# pip install opencv-python numpy
#
#
# Run:
#
# python lane_detection.py
#
# ============================================================