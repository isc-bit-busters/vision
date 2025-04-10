import cv2 as cv
from camera import Camera
from aruco import Aruco, processAruco
from position import Position

# Initialize the camera with its intrinsic parameters and distortion coefficients
camera = Camera("Logitec_robot", 1, focus=10, resolution=(1920, 1080))

# Define fixed ArUco markers with known positions
fixedArucos = [
    Aruco(id=0, size=53, topLeft=Position(0, 0, 0)),
    Aruco(id=1, size=53, topLeft=Position(134, 0, 0)),
    Aruco(id=2, size=53, topLeft=Position(0, -215, 0)), 
    Aruco(id=3, size=53, topLeft=Position(134, -215, 0)),
]

# Define moving ArUco markers whose positions need to be determined
movingArucos = [
    Aruco(id=5, size=53),
    Aruco(id=6, size=53),
]

while True:
    ret, frame = camera.captureStream.read()
    if not ret:
        print("Failed to grab frame")
        break

    try:
        # Detect and process ArUco markers in the current frame
        rvec, tvec, arucosPosition, metrics_collected = processAruco(
            fixedArucos=fixedArucos,
            movingArucos=movingArucos,
            camera=camera,
            img=frame,
            accept_none=True,
            directUpdate=True,
            metrics=True,
        )

        for aruco_id, transform in arucosPosition.items():
            print(f"ArUco ID {aruco_id} Position: {transform.tvec}, Rotation: {transform.rvec}")
        
            pass

        # Draw camera axes
        if rvec is not None and tvec is not None:
            # print(f"Camera Position: {tvec} and rotation : {rvec}")
            cv.drawFrameAxes(frame, camera.mtx, camera.dist, rvec, tvec, 50)  # length in mm

    except Exception as e:
        print("Error in processing:", e)

    # Show frame
    cv.imshow("Live ArUco Tracking", frame)
    key = cv.waitKey(1)
    if key == 27:  # ESC to quit
        break

cv.destroyAllWindows()