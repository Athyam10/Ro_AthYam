import cv2
import numpy as np

# ============================================================
# HOUGH TRANSFORM LINE DETECTION
# ============================================================
# This program:
# 1. Loads an image
# 2. Performs Canny Edge Detection
# 3. Detects lines using:
#       - Standard Hough Transform
#       - Probabilistic Hough Transform
# 4. Draws detected lines on the original image
#
# You can change parameters from the terminal
# to understand how they affect detection.
# ============================================================

# ----------------------------
# Load image
# ----------------------------
image = cv2.imread("img2.jpeg")

# Check if image loaded correctly
if image is None:
    print("Could not load image")
    exit()

# Show original image
cv2.imshow("Original Image", image)

while True:

    # ========================================================
    # USER INPUT SECTION
    # ========================================================

    print("\n================ PARAMETER INPUT =================")

    # Canny thresholds
    canny_input = input(
        "Enter LOW and HIGH Canny thresholds "
        "(example: 50 150) or q to quit: "
    )

    # Quit option
    if canny_input.lower() == 'q':
        break

    try:
        low_threshold, high_threshold = map(int, canny_input.split())
    except ValueError:
        print("Invalid input. Example: 50 150")
        continue

    # Standard Hough parameters
    hough_input = input(
        "\nSTANDARD HOUGH PARAMETERS\n"
        "Enter rho theta threshold\n"
        "Example: 1 1 120\n"
        "Recommended: 1 1 100\n"
        "\nMeaning:\n"
        "rho = distance resolution in pixels\n"
        "theta = angle resolution in degrees\n"
        "threshold = votes needed to detect line\n"
        "\nInput: "
    )

    try:
        rho, theta_deg, threshold = map(float, hough_input.split())

        # Threshold must be integer
        threshold = int(threshold)

    except ValueError:
        print("Invalid input.")
        continue

    # Probabilistic Hough parameters
    houghp_input = input(
        "\nPROBABILISTIC HOUGH PARAMETERS\n"
        "Enter rho theta threshold minLineLength maxLineGap\n"
        "Example: 1 1 50 50 10\n"
        "Recommended: 1 1 50 80 20\n"
        "\nMeaning:\n"
        "minLineLength = minimum accepted line length\n"
        "maxLineGap = maximum gap between line segments\n"
        "\nInput: "
    )

    try:
        (
            p_rho,
            p_theta_deg,
            p_threshold,
            min_line_length,
            max_line_gap
        ) = map(float, houghp_input.split())

        p_threshold = int(p_threshold)

    except ValueError:
        print("Invalid input.")
        continue

    # ========================================================
    # IMAGE PROCESSING
    # ========================================================

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Reduce noise using Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny Edge Detection
    edges = cv2.Canny(blurred, low_threshold, high_threshold)

    # ========================================================
    # STANDARD HOUGH TRANSFORM
    # ========================================================
    # Detects infinite lines using rho and theta
    # More computationally expensive
    # Returns lines in polar coordinates
    # ========================================================

    standard_output = image.copy()

    lines = cv2.HoughLines(
        edges,
        rho=rho,
        theta=np.deg2rad(theta_deg),
        threshold=threshold
    )

    # Check if lines detected
    if lines is not None:

        for line in lines:

            # Extract rho and theta
            rho_val, theta_val = line[0]

            # Convert polar coordinates to Cartesian
            a = np.cos(theta_val)
            b = np.sin(theta_val)

            x0 = a * rho_val
            y0 = b * rho_val

            # Create long line points
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))

            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            # Draw red line
            cv2.line(
                standard_output,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                2
            )

    # ========================================================
    # PROBABILISTIC HOUGH TRANSFORM
    # ========================================================
    # Detects actual line segments
    # Faster and more efficient
    # Returns endpoints directly
    # ========================================================

    probabilistic_output = image.copy()

    lines_p = cv2.HoughLinesP(
        edges,
        rho=p_rho,
        theta=np.deg2rad(p_theta_deg),
        threshold=p_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )

    # Check if lines detected
    if lines_p is not None:

        for line in lines_p:

            # Extract endpoints
            x1, y1, x2, y2 = line[0]

            # Draw green line
            cv2.line(
                probabilistic_output,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3
            )

    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    cv2.imshow("Edge Detection", edges)

    cv2.imshow(
        "Standard Hough Transform (Red Lines)",
        standard_output
    )

    cv2.imshow(
        "Probabilistic Hough Transform (Green Lines)",
        probabilistic_output
    )

    # ========================================================
    # PARAMETER EXPLANATION
    # ========================================================

    print("\n================ EXPLANATION =================")

    print("\n1. RHO")
    print("- Distance resolution in pixels.")
    print("- Smaller rho = more accurate but slower.")
    print("- Larger rho = faster but less precise.")

    print("\n2. THETA")
    print("- Angle resolution in degrees.")
    print("- Smaller theta = detects more angles.")
    print("- Smaller theta increases computation.")

    print("\n3. THRESHOLD")
    print("- Minimum votes needed to detect a line.")
    print("- Low threshold = more lines + more noise.")
    print("- High threshold = fewer but stronger lines.")

    print("\n4. MIN LINE LENGTH")
    print("- Minimum size of accepted line segment.")
    print("- Small value detects tiny lines/noise.")
    print("- Large value keeps only long lines.")

    print("\n5. MAX LINE GAP")
    print("- Maximum gap between segments to connect.")
    print("- Small gap keeps segments separated.")
    print("- Large gap merges broken lines.")

    print("\n================================================")
    print("STANDARD VS PROBABILISTIC HOUGH")
    print("================================================")

    print("\nSTANDARD HOUGH TRANSFORM")
    print("- Detects infinite lines.")
    print("- Uses all edge points.")
    print("- More computationally expensive.")
    print("- Better for theoretical line detection.")

    print("\nPROBABILISTIC HOUGH TRANSFORM")
    print("- Detects line segments directly.")
    print("- Uses random subset of edge points.")
    print("- Faster and more efficient.")
    print("- Better for real-world applications.")

    print("\n================================================")
    print("ACCURACY VS COMPUTATIONAL EFFICIENCY")
    print("================================================")

    print("\nHigher Accuracy:")
    print("- Smaller rho and theta")
    print("- Lower threshold")
    print("- Detects more detailed lines")
    print("- BUT slower processing")

    print("\nHigher Efficiency:")
    print("- Larger rho and theta")
    print("- Higher threshold")
    print("- Fewer lines processed")
    print("- Faster execution")

    # Small wait so windows refresh properly
    cv2.waitKey(1)

# Close all windows
cv2.destroyAllWindows()