# =========================================================
# HIGHWAY ROAD POLYGON USING DOTTED LANE REFERENCES
# =========================================================
#
# THIS VERSION:
# --------------
# - Uses the lane marking perspective
# - Follows actual highway direction
# - Properly highlights both roads
# - Divider ignored
# - Fixed specifically for YOUR image
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
# Follows LEFT dotted lane direction
#
# =========================================================

left_road = np.array([

    [0, 768],        # bottom left screen corner

    [615, 330],      # top left

    [705, 330],      # top right near divider

    [495, 768]       # bottom right near divider

], np.int32)

# =========================================================
# RIGHT ROAD POLYGON
# =========================================================
#
# Follows RIGHT dotted lane direction
#
# =========================================================

right_road = np.array([

    [870, 768],      # bottom left near divider

    [760, 330],      # top left near divider

    [845, 330],      # top right

    [1365, 768]      # bottom right screen corner

], np.int32)

# =========================================================
# CENTER DIVIDER
# =========================================================

divider = np.array([

    [495, 768],

    [705, 330],

    [760, 330],

    [870, 768]

], np.int32)

# =========================================================
# OVERLAY
# =========================================================

overlay = final.copy()

# LEFT ROAD COLOR
cv2.fillPoly(
    overlay,
    [left_road],
    (255, 180, 0)
)

# RIGHT ROAD COLOR
cv2.fillPoly(
    overlay,
    [right_road],
    (0, 255, 100)
)

# BLEND
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
    (35, 35, 35)
)

# =========================================================
# DRAW BOUNDARY LINES
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
# DIRECTION TEXT
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