import cv2
import os
import numpy as np

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

        kernel = np.ones((5, 5), np.uint8)
        combined_image = cv2.dilate(combined_image, kernel, iterations=1)

        contours, _ = cv2.findContours(
            combined_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 9000]
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

        # Calculate and draw new countours on the original image
        original_image = cv2.imread(image_path)
        gray_original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

        _, thresholded_original_image = cv2.threshold(
            gray_original_image, 50, 255, cv2.THRESH_BINARY_INV
        )
        contours_2, _ = cv2.findContours(thresholded_original_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw bounding boxes around the bricks
        brick_boxes = []

        for i in range(len(filtered_contours)):
            x, y, w, h = cv2.boundingRect(filtered_contours[i])

            cv2.rectangle(result_with_contours, (x, y), (x + w, y + h), (0, 0, 255), 10)
            brick_boxes.append((x, y, w, h))

        cv2.drawContours(result_with_contours, filtered_contours, -1, (0, 255, 0), 2)

        resized_result = cv2.resize(
            np.hstack([result_with_keypoints, result_with_contours]), (800, 400)
        )  # Resize the image

        # Display the result
        print("Number of bricks: ", len(brick_boxes))
        
        cv2.imshow("Result with Keypoints and Contours", resized_result)
        cv2.waitKey(0)

cv2.destroyAllWindows()
