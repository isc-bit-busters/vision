import cv2

# path = "test_aruco1.jpg"
path = "test_corner_aruco.jpg"

image = cv2.imread(path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters()

detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
corners, ids, rejectedImgPoints = detector.detectMarkers(gray)
if ids is not None:
    cv2.aruco.drawDetectedMarkers(image, corners, ids)

cv2.imshow("Detected Markers", image)
cv2.waitKey(0)
cv2.destroyAllWindows()