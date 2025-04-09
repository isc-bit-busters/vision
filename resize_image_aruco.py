import cv2
import numpy as np

path = "image_more_aruco_playground_marta.jpg"

margin_x = 0
margin_y = 0

marker_order = {2 : "top_left", 1 : "top_right", 3 : "bottom_left", 0 : "bottom_right"}

image = cv2.imread(path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
parameters = cv2.aruco.DetectorParameters()

detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
corners, ids, rejectedImgPoints = detector.detectMarkers(gray)

if ids is None or len(ids) < 4:
    print("Not enough markers detected.")
    exit(1)

ids = ids.flatten()

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

src_pts = np.array([
    marker_corners["top_left"][0] - [margin_x, margin_y],
    marker_corners["top_right"][1] + [margin_x, -margin_y],
    marker_corners["bottom_right"][2] + [margin_x, margin_y],
    marker_corners["bottom_left"][3] - [margin_x, -margin_y]
], dtype=np.float32)

# Compute width as the average of top and bottom edge lengths
width_top = np.linalg.norm(marker_corners["top_right"][1] - marker_corners["top_left"][0])
width_bottom = np.linalg.norm(marker_corners["bottom_right"][2] - marker_corners["bottom_left"][3])
width = int(max(width_top, width_bottom))

# Compute height as the average of left and right edge lengths
height_left = np.linalg.norm(marker_corners["bottom_left"][3] - marker_corners["top_left"][0])
height_right = np.linalg.norm(marker_corners["bottom_right"][2] - marker_corners["top_right"][1])
height = int(max(height_left, height_right))

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

# Save the warped image
cv2.imwrite("warped_image.jpg", warped)