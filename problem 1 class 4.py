import argparse
import glob
import os
import sys

import cv2
import numpy as np


def resize_image(image, max_width=1200):
    height, width = image.shape[:2]
    if width <= max_width:
        return image
    scale = max_width / width
    return cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)


def create_road_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sat_mask = cv2.inRange(hsv[:, :, 1], np.array([0], dtype=np.uint8), np.array([90], dtype=np.uint8))
    val_mask = cv2.inRange(hsv[:, :, 2], np.array([50], dtype=np.uint8), np.array([230], dtype=np.uint8))

    a_channel = lab[:, :, 1]
    b_channel = lab[:, :, 2]
    a_mask = cv2.inRange(a_channel, np.array([105], dtype=np.uint8), np.array([145], dtype=np.uint8))
    b_mask = cv2.inRange(b_channel, np.array([103], dtype=np.uint8), np.array([153], dtype=np.uint8))
    ab_mask = cv2.bitwise_and(a_mask, b_mask)

    gray_mask = cv2.inRange(gray, np.array([40], dtype=np.uint8), np.array([220], dtype=np.uint8))
    combined = cv2.bitwise_and(sat_mask, val_mask)
    combined = cv2.bitwise_and(combined, ab_mask)
    combined = cv2.bitwise_and(combined, gray_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
    combined = cv2.GaussianBlur(combined, (5, 5), 0)

    _, combined = cv2.threshold(combined, 50, 255, cv2.THRESH_BINARY)
    return combined


def find_road_contours(mask, image_area):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    road_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < image_area * 0.015:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / max(h, 1)
        if area < image_area * 0.04 and (aspect_ratio < 0.25 or aspect_ratio > 4):
            continue
        road_contours.append((contour, area, x, y, w, h))

    road_contours.sort(key=lambda entry: entry[1], reverse=True)
    return road_contours


def detect_boundary_lines(mask):
    edges = cv2.Canny(mask, 50, 150)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=60, minLineLength=80, maxLineGap=30)
    if lines is None:
        return edges, []

    boundaries = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        dy = y2 - y1
        if abs(dx) < 10 and abs(dy) < 10:
            continue
        slope = dy / dx if dx != 0 else np.inf
        if abs(slope) < 0.3 and abs(dx) > abs(dy):
            continue
        mid_x = (x1 + x2) / 2
        boundaries.append(mid_x)

    boundaries = sorted(boundaries)
    merged = []
    for x in boundaries:
        if not merged or abs(x - merged[-1]) > 80:
            merged.append(x)
    return edges, merged


def draw_detection(image, road_contours, boundaries):
    output = image.copy()
    for contour, _, x, y, w, h in road_contours:
        cv2.drawContours(output, [contour], -1, (0, 255, 0), 3)
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

    height = output.shape[0]
    for boundary in boundaries:
        x = int(boundary)
        cv2.line(output, (x, 0), (x, height), (0, 0, 255), 2)

    return output


def estimate_road_count(road_contours, boundary_count):
    if len(road_contours) > 1:
        return len(road_contours)
    if boundary_count >= 4:
        return 2
    if boundary_count >= 2:
        return 1
    return 0


def format_result(road_count, boundary_count):
    if road_count == 0:
        return "No roads detected"
    if road_count == 1:
        return f"Roads detected: {road_count} (single road, {boundary_count} boundaries)"
    return f"Roads detected: {road_count} (multiple roads, {boundary_count} boundaries)"


def get_available_images(directory="."):
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(directory, pattern)))
    return sorted(files, key=str.lower)


def prompt_for_image_path(default="img2.jpeg"):
    images = get_available_images()
    if images:
        print("Available image files in current folder:")
        for idx, img in enumerate(images, start=1):
            print(f"  {idx}. {img}")
        print("Enter a filename or number from the list.")
    else:
        print("No .jpg/.jpeg/.png/.bmp/.tif images found in the current folder.")

    choice = input(f"Image path to process [{default}]: ").strip()
    if not choice:
        choice = default
    if choice.isdigit():
        index = int(choice) - 1
        if 0 <= index < len(images):
            return images[index]
    return choice


def parse_args():
    parser = argparse.ArgumentParser(description="Road boundary and road count detection")
    parser.add_argument("image", nargs="?", default="img2.jpeg", help="Image path to process")
    parser.add_argument("--save", action="store_true", help="Save result images to disk")
    parser.add_argument("--no-show", action="store_true", help="Do not display GUI windows")
    parser.add_argument("--show", action="store_true", help="Force GUI windows even if non-interactive")
    return parser.parse_args()


def main():
    args = parse_args()
    image_path = args.image

    if not os.path.isfile(image_path):
        print(f"Image not found: {image_path}")
        if sys.stdin.isatty():
            image_path = prompt_for_image_path()
            if not os.path.isfile(image_path):
                print(f"Image still not found: {image_path}")
                return
            print(f"Using image: {image_path}")
        else:
            print("Provide a real image filename instead of a placeholder.")
            return

    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to read image: {image_path}")
        return

    image = resize_image(image)
    image_area = image.shape[0] * image.shape[1]

    road_mask = create_road_mask(image)
    road_contours = find_road_contours(road_mask, image_area)
    edges, boundaries = detect_boundary_lines(road_mask)
    result_image = draw_detection(image, road_contours, boundaries)

    road_count = estimate_road_count(road_contours, len(boundaries))
    message = format_result(road_count, len(boundaries))
    print(message)

    cv2.putText(result_image, message, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    if args.save:
        base = os.path.splitext(os.path.basename(image_path))[0]
        cv2.imwrite(f"{base}_road_mask.png", road_mask)
        cv2.imwrite(f"{base}_road_edges.png", edges)
        cv2.imwrite(f"{base}_road_result.png", result_image)
        print(f"Saved: {base}_road_mask.png, {base}_road_edges.png, {base}_road_result.png")

    # Decide whether to show GUI windows. Default behavior is to show
    # unless `--no-show` is passed. `--show` forces windows even in non-interactive runs.
    show_windows = args.show or (not args.no_show)

    if show_windows:
        try:
            # Create and arrange windows so each result appears in its own window
            cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Road Mask", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Road Boundaries", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Result", cv2.WINDOW_NORMAL)

            # Resize windows for readability
            cv2.resizeWindow("Original", 640, 360)
            cv2.resizeWindow("Road Mask", 480, 360)
            cv2.resizeWindow("Road Boundaries", 480, 360)
            cv2.resizeWindow("Result", 800, 600)

            # Position windows (may be ignored by some window managers)
            cv2.moveWindow("Original", 50, 50)
            cv2.moveWindow("Road Mask", 720, 50)
            cv2.moveWindow("Road Boundaries", 50, 430)
            cv2.moveWindow("Result", 720, 430)

            # Show images
            cv2.imshow("Original", image)
            cv2.imshow("Road Mask", road_mask)
            cv2.imshow("Road Boundaries", edges)
            cv2.imshow("Result", result_image)

            print("Press any key on one of the image windows to close them.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print("Could not open GUI windows:", e)


if __name__ == "__main__":
    main()

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