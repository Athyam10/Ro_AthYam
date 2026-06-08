import os
import cv2
import numpy as np


def classify_lane_lines(lines, image_width):
    """Classify detected Hough lines into left and right lane markings."""

    left_lines = []
    right_lines = []
    center_x = image_width * 0.5

    for line in lines:
        x1, y1, x2, y2 = line[0]

        if x2 == x1:
            continue

        slope = (y2 - y1) / (x2 - x1)

        if abs(slope) < 0.3:
            continue

        x_average = (x1 + x2) / 2

        if slope < 0 and x_average < center_x:
            left_lines.append((x1, y1, x2, y2, slope))
        elif slope > 0 and x_average > center_x:
            right_lines.append((x1, y1, x2, y2, slope))

    return left_lines, right_lines


def remove_outliers(lines):
    """Remove outlier lines using slope statistics."""

    if not lines:
        return []

    slopes = np.array([line[4] for line in lines])
    mean_slope = np.mean(slopes)
    std_slope = np.std(slopes)
    if std_slope == 0:
        return lines

    return [
        line for line in lines
        if abs(line[4] - mean_slope) < 2 * std_slope
    ]


def draw_classification_text(image, left_count, right_count):
    label = f"Left lanes: {left_count}   Right lanes: {right_count}"
    cv2.putText(
        image,
        label,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )


def load_image(filenames):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for name in filenames:
        path = os.path.join(script_dir, name)
        image = cv2.imread(path)
        if image is not None:
            return image, path
    raise FileNotFoundError(f"None of the image files were found: {', '.join(filenames)}")


def region_of_interest(img):
    h, w = img.shape[:2]
    mask = np.zeros_like(img)
    # trapezoid covering lower part of the image
    polygon = np.array([
        [int(0.1 * w), h],
        [int(0.4 * w), int(0.6 * h)],
        [int(0.6 * w), int(0.6 * h)],
        [int(0.9 * w), h],
    ])
    cv2.fillPoly(mask, [polygon], 255)
    return cv2.bitwise_and(img, mask)


def average_lane_line(lines):
    """Fit a single line (slope, intercept) from multiple line segments."""
    if not lines:
        return None
    xs = []
    ys = []
    for x1, y1, x2, y2, _ in lines:
        xs += [x1, x2]
        ys += [y1, y2]
    if len(xs) < 2:
        return None
    # fit x = m*y + b  (so we can compute x for given y values)
    fit = np.polyfit(ys, xs, 1)
    m, b = fit[0], fit[1]
    return m, b


def make_line_points(y1, y2, fit):
    if fit is None:
        return None
    m, b = fit
    x1 = int(m * y1 + b)
    x2 = int(m * y2 + b)
    return (x1, int(y1), x2, int(y2))


def preprocess_and_detect(image):
    # smaller, faster copy for processing
    h, w = image.shape[:2]
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    hls = cv2.cvtColor(blur, cv2.COLOR_BGR2HLS)
    # white mask
    lower_white = np.array([0, 200, 0])
    upper_white = np.array([255, 255, 255])
    white_mask = cv2.inRange(hls, lower_white, upper_white)
    # yellow mask
    lower_yellow = np.array([15, 30, 115])
    upper_yellow = np.array([35, 204, 255])
    yellow_mask = cv2.inRange(hls, lower_yellow, upper_yellow)
    color_mask = cv2.bitwise_or(white_mask, yellow_mask)

    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    # save pre-mask edges for debugging
    try:
        cv2.imwrite(os.path.join(os.path.dirname(__file__), 'debug_edges_before.png'), edges)
    except Exception:
        pass
    # close and dilate mask to join dashed center lines
    kernel = np.ones((5, 5), np.uint8)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
    color_mask = cv2.dilate(color_mask, kernel, iterations=1)
    edges_masked = cv2.bitwise_and(edges, edges, mask=color_mask)
    # debug counts
    debug_path = os.path.join(os.path.dirname(__file__), 'debug_edges_before.png')
    nz_before = -1
    if os.path.exists(debug_path):
        debug_img = cv2.imread(debug_path, 0)
        if debug_img is not None:
            nz_before = int(cv2.countNonZero(debug_img))
    nz_after = int(cv2.countNonZero(edges_masked))
    print(f'Edges nonzero before: {nz_before}, after mask: {nz_after}')
    # perform Hough on ROI edges (without color mask) so segments aren't removed before detection
    edges_roi = region_of_interest(edges)
    edges_roi = cv2.medianBlur(edges_roi, 3)

    # Hough parameters: try multiple configurations from strict->permissive
    def try_hough_p(edges, cfgs):
        for cfg in cfgs:
            lines = cv2.HoughLinesP(
                edges,
                rho=cfg['rho'],
                theta=cfg['theta'],
                threshold=cfg['threshold'],
                minLineLength=cfg['minLineLength'],
                maxLineGap=cfg['maxLineGap'],
            )
            if lines is not None and len(lines) > 0:
                return lines
        return None

    cfgs = [
        # permissive first (helps detect dashed/short segments in this image)
        {'rho':1, 'theta':np.pi/180, 'threshold':10, 'minLineLength':10, 'maxLineGap':50},
        {'rho':1, 'theta':np.pi/180, 'threshold':max(20, int(w*0.02)), 'minLineLength':max(10, int(w*0.03)), 'maxLineGap':max(10, int(w*0.08))},
        {'rho':1, 'theta':np.pi/180, 'threshold':max(40, int(w*0.04)), 'minLineLength':int(w*0.15), 'maxLineGap':int(w*0.05)},
        {'rho':1, 'theta':np.pi/180, 'threshold':max(60, int(w*0.06)), 'minLineLength':int(w*0.25), 'maxLineGap':int(w*0.03)},
    ]

    lines = try_hough_p(edges_roi, cfgs)

    # fallback to standard HoughLines (rho/theta) and convert to segments
    if lines is None:
        raw = cv2.HoughLines(edges_roi, 1, np.pi/180, max(60, int(w*0.04)))
        if raw is not None:
            segs = []
            for rtheta in raw:
                rho,theta = rtheta[0]
                a = np.cos(theta); b = np.sin(theta)
                x0 = a*rho; y0 = b*rho
                # create two far points on the line
                pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
                pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
                segs.append(np.array([[pt1[0], pt1[1], pt2[0], pt2[1]]], dtype=np.int32))
            if segs:
                lines = np.vstack(segs)

    return lines, edges_roi, color_mask


def main():
    # Focus only on the provided image
    image, image_path = load_image([
        "image3.jpg",
    ])
    print(f"Loaded image: {image_path}")
    orig = image.copy()

    lines, edges, color_mask = preprocess_and_detect(image)

    left_lines = []
    right_lines = []

    if lines is None:
        print('Hough returned no lines')
    else:
        print(f'Hough returned {len(lines)} lines')
        # draw raw Hough segments for debug
        hough_vis = orig.copy()
        for l in lines:
            x1, y1, x2, y2 = l[0]
            cv2.line(hough_vis, (x1, y1), (x2, y2), (255, 0, 255), 2)
        try:
            cv2.imwrite(os.path.join(os.path.dirname(image_path), 'debug_hough.png'), hough_vis)
        except Exception:
            pass

        left_lines, right_lines = classify_lane_lines(lines, image.shape[1])
        print(f'Classified into left:{len(left_lines)} right:{len(right_lines)} before outlier removal')
        left_lines = remove_outliers(left_lines)
        right_lines = remove_outliers(right_lines)
        print(f'After outlier removal left:{len(left_lines)} right:{len(right_lines)}')

    # save debug images
    debug_dir = os.path.dirname(image_path)
    try:
        cv2.imwrite(os.path.join(debug_dir, 'debug_edges.png'), edges)
        cv2.imwrite(os.path.join(debug_dir, 'debug_mask.png'), color_mask)
    except Exception:
        pass

    # average to single line per side
    left_fit = average_lane_line(left_lines)
    right_fit = average_lane_line(right_lines)

    h = orig.shape[0]
    y1 = h
    y2 = int(h * 0.6)

    left_points = make_line_points(y1, y2, left_fit)
    right_points = make_line_points(y1, y2, right_fit)

    if left_points is not None:
        x1, y1a, x2, y2a = left_points
        cv2.line(orig, (x1, y1a), (x2, y2a), (255, 0, 0), 8)

    if right_points is not None:
        x1, y1a, x2, y2a = right_points
        cv2.line(orig, (x1, y1a), (x2, y2a), (0, 255, 0), 8)

    status_text = "Lane line classification"
    cv2.putText(
        orig,
        status_text,
        (20, orig.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    draw_classification_text(orig, 1 if left_points is not None else 0, 1 if right_points is not None else 0)

    # save final output for headless inspection
    try:
        out_path = os.path.join(debug_dir, 'debug_output.png')
        cv2.imwrite(out_path, orig)
        print(f"Saved debug output: {out_path}")
    except Exception:
        pass

    cv2.imshow("Lane Classification", orig)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()