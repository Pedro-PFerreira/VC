# Detect number of bricks in an image, considering the color of the bricks.

import cv2
import numpy as np
import os
import random

# FILTERING
# Reduce brightness and shadow effects

def reduce_brightness_and_shadows(image):
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply histogram equalization
    equalized = cv2.equalizeHist(gray)

    # Define the gamma value for gamma correction (adjust as needed)
    gamma = 0.5

    # Apply gamma correction
    corrected = np.uint8(((gray / 255.0) ** gamma) * 255)

    # Convert to BGR

    equalized = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)

    return [equalized, corrected]


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
            if area > 1200 and area < 40000:

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


def detect_colors_hsv(image, brick_boxes, threshold=10):

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

        most_frequent_color = unique_colors[np.argmax(counts)]

        # Use most frequent color as the average color
        if distinct_colors:

            brick_colors.append(most_frequent_color)        

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

        image_resized = cv2.resize(image, (0, 0), fx=0.2, fy=0.2)

        # preprocessed_image = preprocess_image(image_resized)

        enhanced_image = reduce_brightness_and_shadows(image_resized)[0]

        brick_boxes = detect_lego_bricks(enhanced_image)

        num_bricks = len(brick_boxes)

        if num_bricks < 3:

            num_colors, brick_colors = detect_colors_hsv(enhanced_image, brick_boxes)

        else:
            num_colors, brick_colors = detect_colors_hsv(enhanced_image, brick_boxes, threshold=100)

        print(f"File: {filename}, Number of Lego Bricks: {num_bricks}, Number of Colors: {num_colors}")

        # Draw the bounding boxes around the detected bricks with the area they occupy

        for box in (brick_boxes):
                
                x, y, w, h = box
                    
                cv2.rectangle(enhanced_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                cv2.putText(enhanced_image, f"{w*h} mm^2", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Result", enhanced_image)

        cv2.waitKey(0)

        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()