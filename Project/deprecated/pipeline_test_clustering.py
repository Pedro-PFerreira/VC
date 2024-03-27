import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import os
import random


# Use DBSCAN to detect the Lego bricks in the image, by analysing the color distances of the pixels
def detect_lego_bricks(image, eps=5, min_samples=10):
    
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
        l_channel, a_channel, b_channel = cv2.split(lab_image)
    
        pixel_values = np.column_stack([a_channel.flatten(), b_channel.flatten()])
    
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(pixel_values)
    
        labels = db.labels_
    
        num_labels = len(set(labels)) - (1 if -1 in labels else 0)
    
        brick_boxes = []
    
        for i in range(num_labels):
    
            mask = labels == i
    
            a_values = a_channel.flatten()[mask]
    
            b_values = b_channel.flatten()[mask]
    
            min_a, max_a = np.min(a_values), np.max(a_values)
    
            min_b, max_b = np.min(b_values), np.max(b_values)
    
            mask = np.zeros_like(labels, dtype=bool)
    
            mask[labels == i] = True
    
            y, x = np.where(mask.reshape(a_channel.shape))
    
            brick_boxes.append((min(x), min(y), max(x) - min(x), max(y) - min(y)))
    
        return brick_boxes

def preprocess_image(image):

    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    l_channel, a_channel, b_channel = cv2.split(lab_image)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6))

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

        preprocessed_image = preprocess_image(image_resized)

        brick_boxes = detect_lego_bricks(preprocessed_image)

        num_bricks = len(brick_boxes)

        if num_bricks < 3:

            num_colors, brick_colors = detect_colors_hsv(preprocessed_image, brick_boxes)

        else:
            num_colors, brick_colors = detect_colors_hsv(preprocessed_image, brick_boxes, threshold=100)

        print(f"File: {filename}, Number of Lego Bricks: {num_bricks}, Number of Colors: {num_colors}")

        # Draw the bounding boxes around the detected bricks with the area they occupy

        for box in (brick_boxes):
                
                x, y, w, h = box
                    
                cv2.rectangle(preprocessed_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                cv2.putText(preprocessed_image, f"{w*h} mm^2", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Result", preprocessed_image)

        cv2.waitKey(0)

        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()