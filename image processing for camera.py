# =========================================================
# IMPROVED LANE EDGE DETECTION
# =========================================================
#
# THIS VERSION:
# -------------
# Detects ONLY the black/blue lane tape lines
# and ignores the walls/floor/background.
#
# HOW?
# ----
# 1. Uses HSV color filtering
# 2. Detects only DARK BLUE / BLACK colors
# 3. Uses Canny edge detection ONLY on lane mask
# 4. Draws contours only around lane tape
#
# =========================================================

import cv2
import numpy as np


# =========================================================
# CLASS FOR LANE DETECTION
# =========================================================
class LaneDetector:

    # -----------------------------------------------------
    # Constructor
    # -----------------------------------------------------
    def __init__(self):

        # HSV threshold for DARK BLUE / BLACK lane tape
        #
        # You can adjust these later if needed

        self.lower_lane = np.array([90, 40, 20])
        self.upper_lane = np.array([140, 255, 255])

    # -----------------------------------------------------
    # Preprocess frame
    # -----------------------------------------------------
    def preprocess_frame(self, frame):

        # Resize image
        frame = cv2.resize(frame, (640, 480))

        # Slight blur to reduce noise
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)

        return blurred

    # -----------------------------------------------------
    # Detect only lane color
    # -----------------------------------------------------
    def detect_lane_mask(self, frame):

        # Convert image to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create mask for blue/black lane tape
        mask = cv2.inRange(hsv,
                           self.lower_lane,
                           self.upper_lane)

        # Morphological cleanup
        kernel = np.ones((5, 5), np.uint8)

        mask = cv2.morphologyEx(mask,
                                cv2.MORPH_OPEN,
                                kernel)

        mask = cv2.morphologyEx(mask,
                                cv2.MORPH_CLOSE,
                                kernel)

        return mask

    # -----------------------------------------------------
    # Edge detection ONLY on lane mask
    # -----------------------------------------------------
    def detect_edges(self, mask):

        # Detect edges from mask only
        edges = cv2.Canny(mask, 50, 150)

        return edges

    # -----------------------------------------------------
    # Draw detected lane edges
    # -----------------------------------------------------
    def draw_lane_edges(self, frame, edges):

        # Find contours from edges
        contours, _ = cv2.findContours(edges,
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        lane_centers = []

        for contour in contours:

            area = cv2.contourArea(contour)

            # Ignore tiny noise
            if area > 200:

                # Draw contour in GREEN
                cv2.drawContours(frame,
                                 [contour],
                                 -1,
                                 (0, 255, 0),
                                 3)

                # Bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)

                # Center point
                cx = x + w // 2
                cy = y + h // 2

                lane_centers.append((cx, cy))

                # Draw center point
                cv2.circle(frame,
                           (cx, cy),
                           5,
                           (0, 0, 255),
                           -1)

        return lane_centers

    # -----------------------------------------------------
    # Calculate center path
    # -----------------------------------------------------
    def calculate_path(self, frame, lane_centers):

        height, width, _ = frame.shape

        image_center = width // 2

        # Draw center line
        cv2.line(frame,
                 (image_center, 0),
                 (image_center, height),
                 (255, 0, 0),
                 2)

        # Need at least 2 lane contours
        if len(lane_centers) >= 2:

            # Sort left to right
            lane_centers = sorted(lane_centers,
                                  key=lambda x: x[0])

            left_lane = lane_centers[0]
            right_lane = lane_centers[-1]

            # Calculate midpoint
            mid_x = (left_lane[0] + right_lane[0]) // 2
            mid_y = (left_lane[1] + right_lane[1]) // 2

            # Draw midpoint
            cv2.circle(frame,
                       (mid_x, mid_y),
                       10,
                       (255, 255, 0),
                       -1)

            # Draw steering line
            cv2.line(frame,
                     (image_center, height),
                     (mid_x, mid_y),
                     (0, 255, 255),
                     3)

            # Steering decision
            if mid_x < image_center - 30:
                direction = "TURN LEFT"

            elif mid_x > image_center + 30:
                direction = "TURN RIGHT"

            else:
                direction = "MOVE STRAIGHT"

            # Display direction
            cv2.putText(frame,
                        direction,
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 255),
                        3)

    # -----------------------------------------------------
    # Main processing function
    # -----------------------------------------------------
    def process_frame(self, frame):

        # Step 1: Preprocess
        processed = self.preprocess_frame(frame)

        # Step 2: Get lane-only mask
        mask = self.detect_lane_mask(processed)

        # Step 3: Edge detection on mask only
        edges = self.detect_edges(mask)

        # Step 4: Draw lane edges
        lane_centers = self.draw_lane_edges(processed,
                                            edges)

        # Step 5: Calculate path
        self.calculate_path(processed,
                            lane_centers)

        return processed, mask, edges


# =========================================================
# MAIN PROGRAM
# =========================================================

# Load image
image = cv2.imread("img.jpg")

# Check image
if image is None:
    print("Could not load image")
    exit()

# Create detector object
detector = LaneDetector()

# Process image
result, mask, edges = detector.process_frame(image)

# =========================================================
# DISPLAY RESULTS
# =========================================================

# Original image
cv2.imshow("Original Image", image)

# Lane-only mask
cv2.imshow("Lane Mask", mask)

# Edge detection ONLY on lanes
cv2.imshow("Lane Edges", edges)

# Final output
cv2.imshow("Final Lane Detection", result)

# Wait for key press
cv2.waitKey(0)

# Close windows
cv2.destroyAllWindows()