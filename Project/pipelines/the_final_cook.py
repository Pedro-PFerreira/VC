import cv2
import os
import numpy as np
from sklearn.cluster import KMeans
from testing.testing import test, compute_testing_score


def overlap_area(rect1, rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    dx = min(x1 + w1, x2 + w2) - max(x1, x2)
    dy = min(y1 + h1, y2 + h2) - max(y1, y2)
    if (dx >= 0) and (dy >= 0):
        return dx * dy
    return 0


def threshold_bricks(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.medianBlur(gray, 35)
    smoothed = cv2.bilateralFilter(denoised, -1, 3, 3)
    thresholded = cv2.adaptiveThreshold(
        smoothed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 5, 1
    )
    return thresholded


def segment_light_artifacts(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 220, 0])
    upper = np.array([255, 240, 255])
    mask = cv2.inRange(hsv, lower, upper)
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    return masked_image


def get_dominant_color(image):
    pixels = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=1)
    kmeans.fit(pixels)
    dominant_color = kmeans.cluster_centers_[0]
    return dominant_color.astype(int)


def color_similarity(colors, threshold=15):
    unique_colors = set()
    for color in colors:
        found_similar = False
        for existing_color in unique_colors:
            if all(abs(existing_color[i] - color[i]) <= threshold for i in range(3)):
                found_similar = True
                break
        if not found_similar:
            unique_colors.add(tuple(color))
    return len(unique_colors)


def display_color_squares(colors, square_size=50, spacing=10):
    num_colors = len(colors)
    window_width = (square_size + spacing) * num_colors + spacing
    window_height = square_size + 2 * spacing

    window = np.ones((window_height, window_width, 3), dtype=np.uint8) * 255

    for i, color in enumerate(colors):
        # convert to tuple tu use un cv2.rectangle()
        color_tuple = tuple(map(int, color))

        x = spacing + i * (square_size + spacing)
        y = spacing

        cv2.rectangle(
            window, (x, y), (x + square_size, y + square_size), color_tuple, -1
        )
    return window


folder_path = "samples"

for filename in os.listdir(folder_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(folder_path, filename)
        thresholded_image = threshold_bricks(image_path)
        original_image = cv2.imread(image_path)
        segmented_image = segment_light_artifacts(original_image)
        combined_image = cv2.bitwise_or(
            thresholded_image, cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
        )

        kernel = np.ones((3, 3), np.uint8)
        combined_image = cv2.dilate(combined_image, kernel, iterations=2)

        contours, _ = cv2.findContours(
            combined_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 6500]
        mask = np.zeros_like(combined_image)
        cv2.drawContours(mask, filtered_contours, -1, (255), thickness=cv2.FILLED)
        result_image = cv2.bitwise_and(combined_image, mask)
        orb = cv2.ORB_create()
        keypoints = orb.detect(result_image, None)
        result_with_keypoints = cv2.drawKeypoints(
            result_image,
            keypoints,
            None,
            color=(0, 255, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )
        result_with_contours = original_image.copy()

        all_brick_boxes = []

        for cnt in filtered_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            all_brick_boxes.append((x, y, w, h))

        brick_boxes = []
        corner_threshold = 0.3

        height, width, _ = original_image.shape

        for cnt in filtered_contours:
            x, y, w, h = cv2.boundingRect(cnt)

            # table edges are normally on corners -> remove detections from those areas
            if (
                (x < corner_threshold and y < corner_threshold)
                or (x < corner_threshold and (height - y - h) < corner_threshold)
                or ((width - x - w) < corner_threshold and y < corner_threshold)
                or (
                    (width - x - w) < corner_threshold
                    and (height - y - h) < corner_threshold
                )
            ):
                continue

            brick_boxes.append((x, y, w, h))

        final_brick_boxes = []
        overlap_threshold = 0.3

        for i, box1 in enumerate(brick_boxes):
            x1, y1, w1, h1 = box1
            keep_box = True
            for j, box2 in enumerate(brick_boxes):
                if i != j:
                    x2, y2, w2, h2 = box2
                    overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                    overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                    overlap_area = overlap_x * overlap_y
                    area1 = w1 * h1
                    area2 = w2 * h2
                    if overlap_area > 0:
                        if (overlap_area / min(area1, area2)) > overlap_threshold:
                            if area1 < area2:
                                keep_box = False
                                break
            if keep_box:
                final_brick_boxes.append(box1)

        dominant_colors = []
        for box in final_brick_boxes:
            x, y, w, h = box
            cv2.rectangle(result_with_contours, (x, y), (x + w, y + h), (0, 0, 255), 10)
            roi = original_image[y : y + h, x : x + w]
            dominant_color = get_dominant_color(roi)
            dominant_colors.append(dominant_color)
            num_colors = color_similarity(dominant_colors)

        color_squares_image = display_color_squares(dominant_colors)
        resized_result = cv2.resize(
            np.hstack([result_with_keypoints, result_with_contours]), (800, 400)
        )

        color_squares_resized = cv2.resize(
            color_squares_image, (resized_result.shape[1], color_squares_image.shape[0])
        )

        stacked_image = np.vstack((resized_result, color_squares_resized))

        test(filename, len(final_brick_boxes), num_colors)

        cv2.imshow("Result", stacked_image)
        cv2.waitKey(0)

compute_testing_score()

cv2.destroyAllWindows()
