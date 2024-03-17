import cv2
import os
import random
import numpy as np


def detect_lego_bricks(image, margin=1):
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    edges = cv2.Canny(blurred, 50, 150)
    dilated = cv2.dilate(edges, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    brick_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if (
            x > margin
            and y > margin
            and x + w < width - margin
            and y + h < height - margin
        ):
            area = cv2.contourArea(contour)
            if area > 800:
                brick_boxes.append((x, y, w, h))
    return brick_boxes


def preprocess_image(image):
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(6, 6))
    enhanced_l_channel = clahe.apply(l_channel)
    enhanced_lab_image = cv2.merge([enhanced_l_channel, a_channel, b_channel])
    enhanced_bgr = cv2.cvtColor(enhanced_lab_image, cv2.COLOR_LAB2BGR)
    return enhanced_bgr


def detect_colors_hsv(image, brick_boxes, threshold=100):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    brick_colors = []
    for box in brick_boxes:
        x, y, w, h = box
        brick_roi = hsv_image[y : y + h, x : x + w]
        unique_colors, counts = np.unique(
            brick_roi.reshape(-1, brick_roi.shape[2]), axis=0, return_counts=True
        )
        distinct_colors = [
            color for color, count in zip(unique_colors, counts) if count >= threshold
        ]
        if distinct_colors:
            average_color = np.mean(distinct_colors, axis=0).astype(np.uint8)
            brick_colors.append(average_color)
    return len(brick_colors), brick_colors


def main():
    folder_path = "samples"
    image_files = [
        f for f in os.listdir(folder_path) if f.endswith(".jpg") or f.endswith(".png")
    ]
    num_images = int(input("Enter the number of images to process: "))
    selected_images = random.sample(image_files, min(num_images, len(image_files)))

    for filename in selected_images:
        image = cv2.imread(os.path.join(folder_path, filename))
        image_resized = cv2.resize(image, (0, 0), fx=0.4, fy=0.4)

        preprocessed_image = preprocess_image(image_resized)
        brick_boxes = detect_lego_bricks(preprocessed_image)
        num_bricks = len(brick_boxes)
        num_colors, brick_colors = detect_colors_hsv(preprocessed_image, brick_boxes)

        print(
            f"File: {filename}, Number of Lego Bricks: {num_bricks}, Number of Distinct Colors: {num_colors}"
        )

        for box in brick_boxes:
            x, y, w, h = box
            cv2.rectangle(preprocessed_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Result", preprocessed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
