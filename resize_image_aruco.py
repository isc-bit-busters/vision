import cv2
import numpy as np

# path = "test_aruco1.jpg"
path = "test_corner_aruco.jpg"

image = cv2.imread(path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters()

detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
corners, ids, rejectedImgPoints = detector.detectMarkers(gray)

if ids is None or len(ids) < 4:
    print("Not enough markers detected.")
    exit(1)

ids = ids.flatten()

marker_order = {3 : "top_left", 0 : "top_right", 2 : "bottom_left", 1 : "bottom_right"}

marker_corners = {}
# Extract marker corners based on ID
for i, id in enumerate(ids):
    if id in marker_order:
        # Each corner[i] is 4 points (the corners of the marker)
        # We'll take the average for simplicity, or specific points if needed
        marker_corners[marker_order[id]] = corners[i][0]

# Ensure we have all 4 required corners
if len(marker_corners) != 4:
    print("Not all required marker IDs were found.")
    exit(1)

# Define source points in the order: top-left, top-right, bottom-right, bottom-left
src_pts = np.array([
    marker_corners["top_left"][0],       # top-left marker: top-left corner
    marker_corners["top_right"][1],      # top-right marker: top-right corner
    marker_corners["bottom_right"][2],   # bottom-right marker: bottom-right corner
    marker_corners["bottom_left"][3]     # bottom-left marker: bottom-left corner
], dtype=np.float32)

# Define the desired width and height of the output image
width = 500
height = 300

dst_pts = np.array([
    [0, 0],
    [width - 1, 0],
    [width - 1, height - 1],
    [0, height - 1]
], dtype=np.float32)

# Compute the perspective transform matrix
M = cv2.getPerspectiveTransform(src_pts, dst_pts)

# Apply the warp transformation
warped = cv2.warpPerspective(image, M, (width, height))

# Display result
cv2.imshow("Warped Image", warped)
cv2.waitKey(0)
cv2.destroyAllWindows()