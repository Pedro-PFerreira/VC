import cv2
import os
import random


def detect_lego_bricks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    dilated = cv2.dilate(edges, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    brick_boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:
            x, y, w, h = cv2.boundingRect(contour)
            brick_boxes.append((x, y, w, h))
    for box in brick_boxes:
        x, y, w, h = box
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    brick_count = len(brick_boxes)
    return image, brick_count


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
        result_image, brick_count = detect_lego_bricks(image_resized)
        print(f"File: {filename}, Number of Lego Bricks: {brick_count}")
        cv2.imshow("Result", result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
