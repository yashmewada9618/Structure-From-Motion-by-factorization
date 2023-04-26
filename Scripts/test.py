import cv2
import os
import glob
import numpy as np

# Load the first image
img_path = '/home/yash/Documents/Computer_VIsion/CV_OptionalProject/hotel/hotel/'
os.chdir(img_path)
images = []
for file in list(glob.glob("*.png")):
    images.append(os.getcwd() + str('/') + file)
images = sorted(images)
img = cv2.imread(images[0], 0)

# Define the feature detector and tracker
feature_detector = cv2.FastFeatureDetector_create(threshold=25)
feature_tracker = cv2.TrackerKCF_create()

# Detect the feature points in the first image
keypoints = feature_detector.detect(img)
prev_pts = np.array([kp.pt for kp in keypoints], dtype=np.float32).reshape(-1, 1, 2)

# Initialize the W matrix
num_frames = 100
W = np.zeros((num_frames*2, prev_pts.shape[0]))

# Loop over the remaining frames and track the feature points
for i in range(1, num_frames):
    # Load the current image
    img_path = f'path/to/image/number/{i}'
    img = cv2.imread(images[i], 0)

    # Track the feature points from the previous frame to the current frame
    ok, bbox = feature_tracker.update(img, prev_pts[-1])
    if ok:
        curr_pt = np.array(bbox, dtype=np.float32).reshape(-1, 1, 2)
        prev_pts = np.concatenate([prev_pts, curr_pt], axis=0)
    else:
        print("Failed to track feature point")

    # Add the current frame's feature points to the W matrix
    W[2*i:2*i+2, :] = curr_pt.reshape(-1, 2).T

# Subtract the mean in each frame
W -= np.mean(W, axis=1, keepdims=True)

# Compute the SVD of W
U, D, VT = np.linalg.svd(W, full_matrices=True)
