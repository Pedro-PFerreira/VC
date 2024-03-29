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

def remove_background(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, threshold_img = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = 255 - threshold_img
    result = cv2.bitwise_and(image, image, mask=mask)

    return result

def get_image_dimensions(image_path):
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    return width, height

def plot_image(image, title):

    image_resized = cv2.resize(image, (0, 0), fx=0.2, fy=0.2)

    cv2.imshow(title, image_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return image_resized

def validate_countours(image):

    # Get image dimensions
    height, width, channels = image.shape

    valid_countours = []
    contour_index = -1

    for contour in contours:
        contour_index += 1
        contour_area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)

        ratio = float(w) / h
        if x > 0 and y > 0 and x + w < width and y + h < height:
            if contour_area > 2000 and ratio <= 6:
                valid_countours.append(contour_index)

    print("Valid Countours size: ", len(valid_countours))

    return valid_countours

# Folder containing images
folder_path = "samples"

# Get random image
random_image_path = get_random_image(folder_path)

# Display the random image

image = cv2.imread(random_image_path)
image_resized = cv2.resize(image, (0, 0), fx=0.2, fy=0.2)

# Remove background from the original image
removed_background = remove_background(image_resized)

cv2.imshow("Removed Background", removed_background)

# # Find contours on the remaining objects
# contours, _ = cv2.findContours(
#     removed_background, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
# )

# # Draw contours on the original image
# original_image = cv2.imread(random_image_path)

# cv2.drawContours(original_image, contours, -1, (0, 255, 0), 2)

# # Display the original image with contours
# plot_image(
#     cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), "Original Image with Contours"
# )

# # Make the bounding box of the contours

# # Create a copy of the original image
# bounding_box_image = original_image.copy()
# brick_boxes = []

# valid_contours = validate_countours(bounding_box_image)

# for i in valid_contours:
#     cv2.drawContours(bounding_box_image, contours, i, (0, 255, 0), 2)
#     x, y, w, h = cv2.boundingRect(contours[i])
#     cv2.rectangle(bounding_box_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
#     brick_boxes.append((x, y, w, h))

# # Display the original image with bounding boxes
    
# plot_image(
#     cv2.cvtColor(bounding_box_image, cv2.COLOR_BGR2RGB), "Original Image with Bounding Boxes"
# )

# # Number of different colors found
# unique_colors = set()

# for box in brick_boxes:
#     x, y, w, h = box
#     brick_roi = removed_background[y : y + h, x : x + w]
#     colors = np.unique(brick_roi)
#     unique_colors.update(colors)

# num_colors = len(unique_colors)

# test(random_image_path, len(brick_boxes), num_colors)

# # Print image information
# print("Random Image:", random_image_path)
# print("Image Dimensions:", get_image_dimensions(random_image_path))


# compute_testing_score()