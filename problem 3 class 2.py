# =========================================================
# EXACT HIGHWAY ROAD HIGHLIGHT
# FIXED FOR YOUR IMAGE
# =========================================================
#
# THIS VERSION:
# ----------------
# - Starts polygons from screen edges
# - Roads highlighted correctly
# - Divider ignored
# - Perspective matches highway
# - Top points reach horizon
#
# =========================================================

import cv2
import numpy as np


# =========================================================
# LOAD IMAGE
# =========================================================

image = cv2.imread("img2.jpeg")

if image is None:
    print("ERROR: img2.jpeg not found")
    exit()

# Resize image
image = cv2.resize(image, (1365, 768))

# Copy image
final = image.copy()

# =========================================================
# LEFT ROAD POLYGON
# =========================================================
#
# POINT ORDER:
# -------------
# bottom-left      -> screen corner
# top-left         -> horizon left road
# top-right        -> near divider
# bottom-right     -> divider bottom
#
# =========================================================

left_road = np.array([

    [0, 768],        # SCREEN BOTTOM LEFT CORNER

    [740, 300],      # TOP RIGHT

    [700, 300],      # TOP LEFT

    [420, 768]       # BOTTOM RIGHT

], np.int32)

# =========================================================
# RIGHT ROAD POLYGON
# =========================================================

right_road = np.array([

    [945, 768],      # BOTTOM LEFT

    [820, 300],      # TOP LEFT

    [860, 300],      # TOP RIGHT

    [1365, 768]      # SCREEN BOTTOM RIGHT CORNER

], np.int32)

# =========================================================
# DIVIDER POLYGON
# =========================================================

divider = np.array([

    [420, 768],

    [700, 300],

    [820, 300],

    [945, 768]

], np.int32)

# =========================================================
# CREATE OVERLAY
# =========================================================

overlay = final.copy()

# ---------------------------------------------------------
# LEFT ROAD FILL
# ---------------------------------------------------------

cv2.fillPoly(
    overlay,
    [left_road],
    (255, 180, 0)
)

# ---------------------------------------------------------
# RIGHT ROAD FILL
# ---------------------------------------------------------

cv2.fillPoly(
    overlay,
    [right_road],
    (0, 255, 100)
)

# =========================================================
# BLEND
# =========================================================

cv2.addWeighted(
    overlay,
    0.35,
    final,
    0.65,
    0,
    final
)

# =========================================================
# REMOVE DIVIDER
# =========================================================

cv2.fillPoly(
    final,
    [divider],
    (40, 40, 40)
)

# =========================================================
# DRAW BOUNDARIES
# =========================================================

cv2.polylines(
    final,
    [left_road],
    True,
    (255, 255, 255),
    5
)

cv2.polylines(
    final,
    [right_road],
    True,
    (255, 255, 255),
    5
)

# =========================================================
# TEXT
# =========================================================

cv2.putText(
    final,
    "ROAD GOES STRAIGHT",
    (25, 70),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.5,
    (0, 255, 255),
    4
)

# =========================================================
# SHOW RESULTS
# =========================================================

cv2.imshow("Original Image", image)

cv2.imshow("Final Two-Road Highlight", final)

cv2.waitKey(0)

cv2.destroyAllWindows()