import cv2
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from testing.testing import test, compute_testing_score

def get_random_image(folder_path):
    image_files = [
        f
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
    ]
    random_image = random.choice(image_files)
    return os.path.join(folder_path, random_image)


def remove_background(image_path, background_color):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Calculate the threshold values based on the provided background color
    _, thresh = cv2.threshold(
        gray, background_color * 255 - 20, 255, cv2.THRESH_BINARY_INV
    )
    return thresh


def get_background_color(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (17, 17), 0)
    hist = cv2.calcHist([blurred], [0], None, [256], [0, 256])
    background_color = np.argmax(hist) / 255.0  # Normalize to range [0, 1]
    return background_color


def get_image_dimensions(image_path):
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    return width, height


def plot_image(image, title):
    plt.imshow(image, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.show()


def count_bricks(contours, image, min_size_threshold, max_size_threshold):
    brick_count = 0
    brick_colors = []
    for contour in contours:
        # Flag as noise if contour is on the edge of the image
        x, y, w, h = cv2.boundingRect(contour)
        if x == 0 or y == 0 or x + w == image.shape[1] or y + h == image.shape[0]:
            continue  # Skip contour
        area = cv2.contourArea(contour)
        if min_size_threshold < area < max_size_threshold:
            brick_count += 1
            # Compute bounding box if contour size is within threshold
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Get mean color within the contour area
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            mean_color = cv2.mean(image, mask=mask)[:3]
            brick_colors.append(mean_color)
    return brick_count, brick_colors


# Folder containing images
folder_path = "samples"

# Get random image
random_image_path = get_random_image(folder_path)

# Get background color
background_color = get_background_color(random_image_path)

# Remove background from the original image using the background color
removed_background = remove_background(random_image_path, background_color)

# Find contours on the remaining objects
contours, _ = cv2.findContours(
    removed_background, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

# Draw contours and bounding boxes on the original image
original_image = cv2.imread(random_image_path)
# cv2.drawContours(original_image, contours, -1, (0, 255, 0), 2)

# Define minimum and maximum contour size thresholds
min_size_threshold = 10000  # Adjust as needed
max_size_threshold = 500000  # Adjust as needed

# Count bricks and get their colors
brick_count, brick_colors = count_bricks(
    contours, original_image, min_size_threshold, max_size_threshold
)

print("Random Image:", random_image_path)
print("Background Color (Gray):", background_color)
print("Image Dimensions:", get_image_dimensions(random_image_path))
print("Number of Bricks:", brick_count)
print("Brick Colors (BGR):", len(brick_colors))

test(random_image_path, brick_count, len(brick_colors))
compute_testing_score()

plot_image(
    cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB),
    "Original Image with Contours and Bounding Boxes",
)
