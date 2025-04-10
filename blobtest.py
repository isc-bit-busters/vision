import cv2
import numpy as np

# Camera intrinsic parameters (you would get these from camera calibration)
focal_length = 800  # Example focal length (in pixels)
real_width = 0.1  # Real-world width of the object in cm

# Load image
image = cv2.imread('img/robot_camera.jpg')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply threshold to get binary image (black and white)
_, binary_image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Create blob detector
detector = cv2.SimpleBlobDetector_create()

# Detect blobs
keypoints = detector.detect(binary_image)

# Get the area or width of the largest blob
if keypoints:
    # Find the largest blob (in case there are multiple blobs)
    largest_blob = max(keypoints, key=lambda x: x.size)
    blob_width = largest_blob.size  # Size of the blob in pixels

    print(f"Blob width in pixels: {blob_width}")

    # Use the formula to estimate distance
    # Distance (d) = (focal length * real width) / width in image
    distance = (focal_length * real_width) / blob_width
    print(f"Estimated distance to the object: {distance} cm")

    # Visualize the blob (optional)
    output_image = cv2.drawKeypoints(image, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow("Blob Detection", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No blobs detected")
