import cv2
import os
import numpy as np
from itertools import product
from testing.testing import test

def threshold_bricks(image_path, block_size=5, constant=1):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.medianBlur(gray, 15)
    smoothed = cv2.bilateralFilter(denoised, 3, 5, 5)
    thresholded = cv2.adaptiveThreshold(
        smoothed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, constant
    )
    return thresholded

def segment_light_artifacts(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([20, 150, 150])
    upper_yellow = np.array([120, 250, 250])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    blurred_masked_image = cv2.medianBlur(masked_image, 11)
    return blurred_masked_image

def hyperparameter_tuning(folder_path, block_sizes, constants, dilation_kernels, min_contour_areas):
    parameter_combinations = product(block_sizes, constants, dilation_kernels, min_contour_areas)
    brick_error_sum = 0
    color_error_sum = 0
    n_tests = 0

    for block_size, constant, kernel_size, min_area in parameter_combinations:
        brick_error_acc = 0
        color_error_acc = 0

        for filename in os.listdir(folder_path):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_path = os.path.join(folder_path, filename)
                thresholded_image = threshold_bricks(image_path, block_size, constant)
                original_image = cv2.imread(image_path)
                segmented_image = segment_light_artifacts(original_image)
                combined_image = cv2.bitwise_or(
                    thresholded_image, cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
                )

                combined_image = cv2.dilate(combined_image, np.ones((kernel_size, kernel_size), np.uint8), iterations=1)

                contours, _ = cv2.findContours(
                    combined_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

                brick_boxes = []

                for i in range(len(filtered_contours)):
                    x, y, w, h = cv2.boundingRect(filtered_contours[i])
                    brick_boxes.append((x, y, w, h))

                num_bricks = len(brick_boxes)
                num_colors = len(set([original_image[y:y+h, x:x+w].tobytes() for x, y, w, h in brick_boxes]))

                brick_error, color_error = test(filename, num_bricks, num_colors, print_feedback=False)
                brick_error_acc += brick_error
                color_error_acc += color_error
                n_tests += 1

        brick_error_sum += brick_error_acc
        color_error_sum += color_error_acc

        print(f"Average Brick Error: {brick_error_acc / n_tests}")
        print(f"Average Color Error: {color_error_acc / n_tests}")

def display_result(image_path, block_size, constant, kernel_size, min_area):
    thresholded_image = threshold_bricks(image_path, block_size, constant)
    original_image = cv2.imread(image_path)
    segmented_image = segment_light_artifacts(original_image)
    combined_image = cv2.bitwise_or(
        thresholded_image, cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
    )

    combined_image = cv2.dilate(combined_image, np.ones((kernel_size, kernel_size), np.uint8), iterations=1)

    contours, _ = cv2.findContours(
        combined_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

    brick_boxes = []

    for i in range(len(filtered_contours)):
        x, y, w, h = cv2.boundingRect(filtered_contours[i])
        brick_boxes.append((x, y, w, h))
        cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    result_image = np.hstack([combined_image, original_image])
    resized_result = cv2.resize(result_image, (800, 400))

    cv2.imshow("Result", resized_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main(tune_hyperparameters=False):
    folder_path = "samples"
    block_sizes = [3, 5, 7]  # Example block sizes for adaptive thresholding
    constants = [1, 2, 3]    # Example constants for adaptive thresholding
    dilation_kernels = [3, 5, 7]  # Example kernel sizes for dilation
    min_contour_areas = [8000, 9000, 10000]  # Example min contour areas

    if tune_hyperparameters:
        hyperparameter_tuning(folder_path, block_sizes, constants, dilation_kernels, min_contour_areas)
    else:
        for filename in os.listdir(folder_path):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_path = os.path.join(folder_path, filename)
                display_result(image_path, 3, 1, 7, 10000)

if __name__ == "__main__":
    main(tune_hyperparameters=False)  # Set to True to run hyperparameter tuning, False to display results for each image
