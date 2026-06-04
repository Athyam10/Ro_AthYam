import cv2
import numpy as np

# =========================================================
# VIDEO INPUT
# =========================================================
video_path = "video1.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("ERROR: Cannot open video.")
    exit()

# =========================================================
# VIDEO SETTINGS
# =========================================================
FRAME_WIDTH = 1024
FRAME_HEIGHT = 576

fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out = cv2.VideoWriter(
    "lane_output.mp4",
    fourcc,
    fps,
    (FRAME_WIDTH, FRAME_HEIGHT)
)

# =========================================================problem
# STABILITY VARIABLES
# =========================================================
prev_left_fit = None
prev_right_fit = None

SMOOTHING = 0.80

MIN_LANE_WIDTH = 220
MAX_LANE_WIDTH = 700

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def smooth_fit(current, previous):

    if current is None:
        return previous

    if previous is None:
        return current

    smoothed = []

    for c, p in zip(current[:4], previous[:4]):

        smoothed.append(
            int(SMOOTHING * p + (1 - SMOOTHING) * c)
        )

    smoothed.append(current[4])

    return tuple(smoothed)


def fit_lane(points, y_bottom, y_top, width):

    if len(points) < 4:
        return None

    xs = np.array([p[0] for p in points])
    ys = np.array([p[1] for p in points])

    try:

        coeff = np.polyfit(ys, xs, 1)

        x_bottom = int(np.clip(
            np.polyval(coeff, y_bottom),
            0,
            width - 1
        ))

        x_top = int(np.clip(
            np.polyval(coeff, y_top),
            0,
            width - 1
        ))

        return (
            x_bottom,
            y_bottom,
            x_top,
            y_top,
            coeff
        )

    except:
        return None


# =========================================================
# MAIN LOOP
# =========================================================
while True:

    ret, image = cap.read()

    if not ret:
        break

    # -----------------------------------------------------
    # RESIZE
    # -----------------------------------------------------
    image = cv2.resize(
        image,
        (FRAME_WIDTH, FRAME_HEIGHT)
    )

    H, W = image.shape[:2]

    # -----------------------------------------------------
    # PREPROCESSING
    # -----------------------------------------------------
    blur_bgr = cv2.GaussianBlur(
        image,
        (5, 5),
        0
    )

    gray = cv2.cvtColor(
        blur_bgr,
        cv2.COLOR_BGR2GRAY
    )

    hls = cv2.cvtColor(
        blur_bgr,
        cv2.COLOR_BGR2HLS
    )

    # White lane mask
    white_mask = cv2.inRange(
        hls,
        np.array([0, 200, 0]),
        np.array([255, 255, 30])
    )

    # Yellow lane mask
    yellow_mask = cv2.inRange(
        hls,
        np.array([18, 100, 100]),
        np.array([35, 255, 255])
    )

    # Combine masks
    color_mask = cv2.bitwise_or(
        white_mask,
        yellow_mask
    )

    # Morphological cleanup
    kernel = np.ones((3, 3), np.uint8)

    color_mask = cv2.dilate(
        color_mask,
        kernel,
        iterations=1
    )

    color_mask = cv2.morphologyEx(
        color_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    masked_gray = cv2.bitwise_and(
        gray,
        gray,
        mask=color_mask
    )

    # -----------------------------------------------------
    # EDGE DETECTION
    # -----------------------------------------------------
    edges = cv2.Canny(
        masked_gray,
        50,
        150
    )

    # -----------------------------------------------------
    # ROI
    # -----------------------------------------------------
    roi_vertices = np.array([[
        (int(W * 0.10), H),
        (int(W * 0.90), H),
        (int(W * 0.58), int(H * 0.50)),
        (int(W * 0.42), int(H * 0.50)),
    ]], dtype=np.int32)

    roi_mask = np.zeros_like(edges)

    cv2.fillPoly(
        roi_mask,
        [roi_vertices[0]],
        255
    )

    cropped = cv2.bitwise_and(
        edges,
        roi_mask
    )

    # -----------------------------------------------------
    # HOUGH LINE DETECTION
    # -----------------------------------------------------
    lines = None

    for p in [
        {"threshold": 40, "minLineLength": 50, "maxLineGap": 50},
        {"threshold": 25, "minLineLength": 30, "maxLineGap": 80},
        {"threshold": 15, "minLineLength": 15, "maxLineGap": 120},
    ]:

        lines = cv2.HoughLinesP(
            cropped,
            1,
            np.pi / 180,
            threshold=p["threshold"],
            minLineLength=p["minLineLength"],
            maxLineGap=p["maxLineGap"]
        )

        if lines is not None and len(lines) > 0:
            break

    if lines is None:
        lines = []

    # -----------------------------------------------------
    # COLLECT LEFT / RIGHT POINTS
    # -----------------------------------------------------
    left_points = []
    right_points = []

    for line in lines:

        x1, y1, x2, y2 = line[0]

        if x2 == x1:
            continue

        slope = (y2 - y1) / (x2 - x1)

        # Reject flat lines
        if abs(slope) < 0.3:
            continue

        # LEFT LANE
        if (
            slope < 0 and
            x1 < W * 0.55 and
            x2 < W * 0.55
        ):

            left_points.extend([
                (x1, y1),
                (x2, y2)
            ])

        # RIGHT LANE
        elif (
            slope > 0 and
            x1 > W * 0.45 and
            x2 > W * 0.45
        ):

            right_points.extend([
                (x1, y1),
                (x2, y2)
            ])

    # -----------------------------------------------------
    # LANE FITTING
    # -----------------------------------------------------
    y_bottom = H
    y_top = int(H * 0.50)

    left_fit = fit_lane(
        left_points,
        y_bottom,
        y_top,
        W
    )

    right_fit = fit_lane(
        right_points,
        y_bottom,
        y_top,
        W
    )

    # -----------------------------------------------------
    # PREVENT LANE CROSSING
    # -----------------------------------------------------
    if left_fit and right_fit:

        lx_bottom = left_fit[0]
        rx_bottom = right_fit[0]

        lx_top = left_fit[2]
        rx_top = right_fit[2]

        # Left lane must remain left
        if lx_bottom >= rx_bottom or lx_top >= rx_top:

            left_fit = None
            right_fit = None

        else:

            lane_width_bottom = (
                rx_bottom - lx_bottom
            )

            lane_width_top = (
                rx_top - lx_top
            )

            # Width sanity check
            if (
                lane_width_bottom < MIN_LANE_WIDTH or
                lane_width_bottom > MAX_LANE_WIDTH
            ):
                left_fit = None
                right_fit = None

            if (
                lane_width_top < MIN_LANE_WIDTH // 2 or
                lane_width_top > MAX_LANE_WIDTH
            ):
                left_fit = None
                right_fit = None

    # -----------------------------------------------------
    # SMOOTHING
    # -----------------------------------------------------
    left_fit = smooth_fit(
        left_fit,
        prev_left_fit
    )

    right_fit = smooth_fit(
        right_fit,
        prev_right_fit
    )

    prev_left_fit = left_fit
    prev_right_fit = right_fit

    # -----------------------------------------------------
    # STEERING CALCULATION
    # -----------------------------------------------------
    vehicle_center_x = W // 2

    lane_center_x = vehicle_center_x

    look_ahead_dist = H - y_top

    detection_case = "NO LANES"

    left_x_bottom = None
    right_x_bottom = None

    if left_fit and right_fit:

        left_x_bottom = left_fit[0]
        right_x_bottom = right_fit[0]

        lane_center_x = (
            left_x_bottom + right_x_bottom
        ) // 2

        detection_case = "BOTH LANES"

    elif left_fit:

        LANE_WIDTH_PX = 300

        left_x_bottom = left_fit[0]

        right_x_bottom = (
            left_x_bottom + LANE_WIDTH_PX
        )

        lane_center_x = (
            left_x_bottom + right_x_bottom
        ) // 2

        detection_case = "LEFT ONLY"

    elif right_fit:

        LANE_WIDTH_PX = 300

        right_x_bottom = right_fit[0]

        left_x_bottom = (
            right_x_bottom - LANE_WIDTH_PX
        )

        lane_center_x = (
            left_x_bottom + right_x_bottom
        ) // 2

        detection_case = "RIGHT ONLY"

    # -----------------------------------------------------
    # STEERING ANGLE
    # -----------------------------------------------------
    lane_offset_px = (
        lane_center_x - vehicle_center_x
    )

    steering_angle = np.degrees(
        np.arctan2(
            lane_offset_px,
            look_ahead_dist
        )
    )

    # -----------------------------------------------------
    # COMMAND
    # -----------------------------------------------------
    if abs(steering_angle) < 3:

        command = "GO STRAIGHT"

    elif steering_angle > 0:

        command = f"TURN RIGHT {steering_angle:.1f}"

    else:

        command = f"TURN LEFT {abs(steering_angle):.1f}"

    # -----------------------------------------------------
    # DRAWING
    # -----------------------------------------------------
    final_img = image.copy()

    # ROI
    cv2.polylines(
        final_img,
        [roi_vertices[0]],
        True,
        (180, 100, 255),
        2
    )

    # -----------------------------------------------------
    # DRAW SAFE LANE AREA
    # -----------------------------------------------------
    if left_fit and right_fit:

        lxb, lyb, lxt, lyt, _ = left_fit
        rxb, ryb, rxt, ryt, _ = right_fit

        if lxb < rxb and lxt < rxt:

            poly = np.array([
                [lxb, lyb],
                [lxt, lyt],
                [rxt, ryt],
                [rxb, ryb]
            ], np.int32)

            overlay = final_img.copy()

            cv2.fillPoly(
                overlay,
                [poly],
                (0, 200, 80)
            )

            cv2.addWeighted(
                overlay,
                0.30,
                final_img,
                0.70,
                0,
                final_img
            )

            # Left lane
            cv2.line(
                final_img,
                (lxb, lyb),
                (lxt, lyt),
                (0, 255, 255),
                5
            )

            # Right lane
            cv2.line(
                final_img,
                (rxb, ryb),
                (rxt, ryt),
                (255, 255, 0),
                5
            )

    # -----------------------------------------------------
    # CENTER LINES
    # -----------------------------------------------------
    cv2.line(
        final_img,
        (vehicle_center_x, H),
        (vehicle_center_x, y_top),
        (0, 0, 255),
        2
    )

    cv2.line(
        final_img,
        (lane_center_x, H),
        (lane_center_x, y_top),
        (255, 255, 255),
        2
    )

    # -----------------------------------------------------
    # OFFSET ARROW
    # -----------------------------------------------------
    cv2.arrowedLine(
        final_img,
        (vehicle_center_x, H - 40),
        (lane_center_x, H - 40),
        (0, 255, 255),
        3,
        tipLength=0.3
    )

    # -----------------------------------------------------
    # LOOK AHEAD POINT
    # -----------------------------------------------------
    cv2.circle(
        final_img,
        (lane_center_x, y_top + 20),
        10,
        (0, 255, 0),
        -1
    )

    # -----------------------------------------------------
    # STEERING ARC
    # -----------------------------------------------------
    cx = vehicle_center_x
    cy = H - 20

    radius = 80

    start_angle = 270
    end_angle = int(270 + steering_angle)

    color_arc = (
        (0, 255, 0)
        if abs(steering_angle) < 3
        else (0, 165, 255)
    )

    cv2.ellipse(
        final_img,
        (cx, cy),
        (radius, radius // 2),
        0,
        min(start_angle, end_angle),
        max(start_angle, end_angle),
        color_arc,
        3
    )

    # -----------------------------------------------------
    # HUD
    # -----------------------------------------------------
    cv2.rectangle(
        final_img,
        (10, 10),
        (520, 170),
        (20, 20, 20),
        -1
    )

    def draw_text(text, y, color):

        cv2.putText(
            final_img,
            text,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            2
        )

    draw_text(
        f"Command : {command}",
        40,
        (0, 255, 255)
    )

    draw_text(
        f"Steering Angle : {steering_angle:+.2f} deg",
        75,
        (0, 200, 255)
    )

    draw_text(
        f"Lane Offset : {lane_offset_px:+d} px",
        110,
        (255, 255, 180)
    )

    draw_text(
        f"Detection : {detection_case}",
        145,
        (200, 200, 255)
    )

    # -----------------------------------------------------
    # DISPLAY
    # -----------------------------------------------------
    cv2.imshow(
        "Advanced Lane Tracking",
        final_img
    )

    # Save frame
    out.write(final_img)

    # Exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =========================================================
# CLEANUP
# =========================================================
cap.release()
out.release()

cv2.destroyAllWindows()

print("Saved output video: video1.mp4")