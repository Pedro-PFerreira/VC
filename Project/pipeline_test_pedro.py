import cv2
import os
import random
import numpy as np
from sklearn.cluster import DBSCAN

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

    return [equalized, corrected]

# FEATURES

# LOCAL FEATURES
def detect_keypoints(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors

def match_keypoints(descriptors1, descriptors2):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    return good_matches

def draw_matches(image1, keypoints1, image2, keypoints2, matches):
    result = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return result

# CLUSTERING

# Define a function to cluster keypoints into groups representing LEGO bricks
def cluster_keypoints(keypoints, matches_mask, min_samples=5, eps=10):
    # Extract the coordinates of keypoints that are inlier matches
    inlier_keypoints = np.array([keypoints[i].pt for i, match in enumerate(matches_mask) if match == 1])
    
    # Apply DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(inlier_keypoints)
    
    # Get the labels assigned to each cluster
    labels = dbscan.labels_
    
    # Count the number of unique labels, which represent individual LEGO bricks
    num_bricks = len(np.unique(labels)) - 1  # Subtract 1 to exclude noise points
    
    return num_bricks

# DECISIONS

# RANSAC ALGORITHM
def ransac(matches, keypoints1, keypoints2, threshold=5):
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, threshold)
    matches_mask = mask.ravel().tolist()
    return matches_mask

def main():
    folder_path = "samples"
    image_files = [
        f for f in os.listdir(folder_path) if f.endswith(".jpg") or f.endswith(".png")
    ]
    num_images = int(input("Enter the number of images to process: "))
    selected_images = random.sample(image_files, min(num_images, len(image_files)))
    for filename in selected_images:
        image = cv2.imread(os.path.join(folder_path, filename))
        image = cv2.resize(image, (0, 0), fx=0.2, fy=0.2)

        # Apply the reduce_brightness_and_shadows function
        results = reduce_brightness_and_shadows(image)

        cv2.imshow("Original", image)
        # cv2.imshow("Equalized", results[0])
        cv2.imshow("Corrected", results[1])

        # Detect keypoints and descriptors

        keypoints1, descriptors1 = detect_keypoints(image)
        keypoints2, descriptors2 = detect_keypoints(results[1])

        # Match keypoints

        matches = match_keypoints(descriptors1, descriptors2)

        # Draw matches

        result = draw_matches(image, keypoints1, results[1], keypoints2, matches)

         # RANSAC

        matches_mask = ransac(matches, keypoints1, keypoints2)

        # Cluster keypoints to detect LEGO bricks

        num_bricks = cluster_keypoints(keypoints1, matches_mask)

        print(f"Number of LEGO bricks detected: {num_bricks}")

        # Draw RANSAC matches and display the result

        draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, matchesMask=matches_mask, flags=2)
        result = cv2.drawMatches(image, keypoints1, results[1], keypoints2, matches, None, **draw_params)

        cv2.imshow("Matches", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()