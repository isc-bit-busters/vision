import cv2 as cv
from cv2.typing import Vec3f
import numpy as np
from itertools import combinations


from referential import Transform
from camera import Camera
from position import Point, Position


class Aruco:
    corners: list[Position] = None

    # TODO: This method needs refinment, I think we need to generate it using a Transform taht represent the center
    @staticmethod
    def getCornersFromTopLeft(
        topLeft: np.ndarray[float], size
    ) -> np.ndarray[np.ndarray[float]]:
        topLeft
        topRight = topLeft.copy()
        topRight[0] += size
        bottomRight = topLeft.copy()
        bottomRight[0] += size
        bottomRight[1] -= size
        bottomLeft = topLeft.copy()
        bottomLeft[1] -= size
        return np.array([topLeft, topRight, bottomRight, bottomLeft])

    @staticmethod
    def getCornersFromPose(pose: Transform, size: float):
        halfSize = size / 2
        corners = [
            [-halfSize, -halfSize, 0.0],
            [halfSize, -halfSize, 0.0],
            [halfSize, halfSize, 0.0],
            [-halfSize, halfSize, 0.0],
        ]
        final = []
        for c in corners:
            final.append(pose.apply(c))

        return final

    @property
    def isLocated(self) -> bool:
        return self.corners is not None

    def __init__(self, id: int, size: float, topLeft: Position = None):
        self.id = id
        self.size = size
        if topLeft is not None:
            self.corners = [
                Position(*c)
                for c in Aruco.getCornersFromTopLeft(topLeft.coords, self.size)
            ]

    def getCornersAsList(self) -> list[list[float]]:
        return np.array(
            [
                self.corners[0].coords,
                self.corners[1].coords,
                self.corners[2].coords,
                self.corners[3].coords,
            ],
            dtype=np.float32,
        )

    def updatePose(self, aruco2world: Transform):
        newCorners = Aruco.getCornersFromPose(aruco2world, self.size)
        self.corners = newCorners
        pass

    def getCenter(self) -> Position:
        return Position(
            self.corners[0].x + self.size / 2,
            self.corners[0].y - self.size / 2,
            self.corners[0].z,
        )


def estimate_pose_single_markers(corners, marker_size, mtx, distortion):
    """
    Replacement for cv2.aruco.estimatePoseSingleMarkers

    Parameters:
        corners: List of detected marker corners
        marker_size: Physical size of the marker in your desired unit (usually meters)
        mtx: Camera calibration matrix
        distortion: Camera distortion coefficients

    Returns:
        rvecs: List of rotation vectors
        tvecs: List of translation vectors
    """
    # Define the marker points in 3D space at origin
    marker_points = np.array(
        [
            [-marker_size / 2, marker_size / 2, 0],
            [marker_size / 2, marker_size / 2, 0],
            [marker_size / 2, -marker_size / 2, 0],
            [-marker_size / 2, -marker_size / 2, 0],
        ],
        dtype=np.float32,
    )

    # Get 2D coordinates from the corner
    corner_points = corners.astype(np.float32)

    # Use solvePnP to get the rotation and translation vectors
    success, rvec, tvec = cv.solvePnP(
        marker_points,
        corner_points,
        mtx,
        distortion,
        flags=cv.SOLVEPNP_IPPE_SQUARE,
    )

    return rvec, tvec


def getArucosFromPaper(pap_v: int = 1) -> dict[int, Aruco]:
    dic = {}

    if pap_v == 1:
        dic = {
            1: Aruco(1, size=27, topLeft=Position(0, 0, 0)),
            2: Aruco(2, size=27, topLeft=Position(0, 0, 0)),
            3: Aruco(3, size=27, topLeft=Position(0, 0, 0)),
            0: Aruco(0, size=27, topLeft=Position(0, 0, 0)),
            10: Aruco(10, size=80, topLeft=Position(0, 0, 0)),
            11: Aruco(11, size=80, topLeft=Position(110, 0, 0)),
            12: Aruco(12, size=80, topLeft=Position(0, -95, 0)),
            13: Aruco(13, size=80, topLeft=Position(110, -95, 0)),
            14: Aruco(14, size=80, topLeft=Position(0, -190, 0)),
            15: Aruco(15, size=80, topLeft=Position(110, -190, 0)),
        }
    elif pap_v == 2:
        x_start = 0
        y_start = 0
        id = 20
        x_step = 35
        y_step = -35
        size = 30
        for i in range(0, 5):
            for j in range(0, 5):
                dic[id] = Aruco(
                    id,
                    size=size,
                    topLeft=Position(x_start + j * x_step, y_start + i * y_step, 0),
                )
                id += 1
    elif pap_v == 4:
        positions = [
            (0, 0),  # Top left corner
            (150.14, 0),  # Top right corner
            (0, -220.14),  # Bottom left corner
            (150.14, -220.14),  # Bottom right corner
        ]
        id = 32
        x_step = 40
        y_step = -40
        for x_start, y_start in positions:
            for i in range(0, 2):
                for j in range(0, 2):
                    x = x_start + j * x_step
                    y = y_start + i * y_step
                    dic[id] = Aruco(id, size=35, topLeft=Position(x=x, y=y, z=0))
                    id += 1
                    pass
    elif pap_v == 3:
        dic[44] = Aruco(44, size=77, topLeft=Position(0, 0, 0))
        dic[45] = Aruco(45, size=77, topLeft=Position(150.4, 0, 0))
        dic[46] = Aruco(46, size=77, topLeft=Position(0, -220, 0))
        dic[47] = Aruco(47, size=77, topLeft=Position(150.4, -220, 0))

    return dic


def generate_aruco_marker(
    marker_id,
    dictionary_id=cv.aruco.DICT_4X4_50,
    marker_size=200,
    save_path=None,
):
    """
    Generates an ArUco marker and saves it as an image.

    :param marker_id: The ID of the marker to generate (should be within the dictionary range)
    :param dictionary_id: The ArUco dictionary to use (default: DICT_4X4_50)
    :param marker_size: The size of the output image in pixels
    :param save_path: The file path to save the marker image
    :return: The generated marker image
    """
    if save_path is None:
        save_path = f"./aruco/aruco_{marker_id}.png"

    # Load the predefined dictionary
    aruco_dict = cv.aruco.getPredefinedDictionary(dictionary_id)

    # Create an empty image to store the marker
    marker_image = np.zeros((marker_size, marker_size), dtype=np.uint8)

    # Generate the marker
    marker_image = cv.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)

    # Save the marker image
    cv.imwrite(save_path, marker_image)
    print(f"ArUco marker (ID: {marker_id}) saved as {save_path}")
    return marker_image


def drawArucoCorners(frame, img_points: list[Point]):
    colors = [
        (0, (255, 0, 0)),
        (1, (0, 255, 0)),
        (2, (0, 0, 255)),
        (3, (255, 255, 0)),
    ]

    for a in img_points:
        for c in colors:
            cv.circle(
                frame,
                (int(a[c[0]][0]), int(a[c[0]][1])),
                10,
                c[1],
                -1,
            )


def detectAruco(
    img, dictionary_id=cv.aruco.DICT_4X4_50, debug=False
) -> tuple[dict, list]:
    """
    Detect ArUco markers in an image.

    :param img: The image to detect markers in
    :param dictionary_id: The ArUco dictionnary to detect
    :param debug: Whether to display debug information
    :return: A dictionary of detected markers and their corners, and a list of rejected markers
    """

    detected = {}

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    arucoDict = cv.aruco.getPredefinedDictionary(dictionary_id)

    arucoParams = cv.aruco.DetectorParameters()
    # TODO: We may need to test different methods here to find the most accurate for our case !
    arucoParams.cornerRefinementMethod = cv.aruco.CORNER_REFINE_SUBPIX
    arucoParams.cornerRefinementWinSize = 20
    arucoParams.cornerRefinementMinAccuracy = 0.01
    arucoParams.cornerRefinementMaxIterations = 100

    if hasattr(cv.aruco, "detectMarkers"):
        (corners, ids, rejected) = cv.aruco.detectMarkers(
            gray, arucoDict, parameters=arucoParams
        )

    else:
        detector = cv.aruco.ArucoDetector(arucoDict, arucoParams)

        (corners, ids, rejected) = detector.detectMarkers(gray)
    if ids is None:
        if debug:
            print("No markers found")
        return detected, rejected

    for i, id in enumerate(ids):
        detected[id[0]] = corners[i]
    if debug:
        print("Detected aruco markers : ", detected.keys())
    return detected, rejected


def locateAruco(
    aruco: Aruco, img_positions: list, camera: Camera, metrics: bool = False
) -> tuple[Transform, dict[str, any]]:
    metrics = {}

    assert len(img_positions[0]) == 4

    rvec, tvec = estimate_pose_single_markers(
        img_positions, aruco.size, camera.mtx, camera.dist
    )

    arcuo2cam = Transform.fromRodrigues(rvec=rvec, tvec=tvec)

    aruco2world = arcuo2cam.combine(camera.world2cam.invert)

    return aruco2world, metrics


def processAruco(
    fixedArucos: list[Aruco],
    movingArucos: list[Aruco],
    camera: Camera,
    img,
    accept_none=False,
    directUpdate=True,
    metrics=False,
):
    metrics_collected = {}
    corners_position, rejected = detectAruco(img, debug=False)
    final_image_points = []
    final_obj_points = []
    for aruco in fixedArucos:
        if corners_position.get(aruco.id) is None:
            continue
        else:
            for c in aruco.getCornersAsList():
                final_obj_points.append(c)
            for c in corners_position[aruco.id][0]:
                final_image_points.append(c)

    if len(final_image_points) < 4 or len(final_obj_points) < 4:
        if accept_none:
            return False
        raise Exception("Could not find any ArUco markers in the image")

    final_image_points = np.array(final_image_points)
    final_obj_points = np.array(final_obj_points)

    rvec, tvec, met = PnP(
        np.array(final_image_points, dtype=np.float32),
        np.array(final_obj_points, dtype=np.float32),
        camera.mtx,
        camera.dist,
        metrics,
    )

    if metrics:
        metrics_collected["PnP"] = met

    camera.updateWorldPosition(rvec, tvec, noLowPassFilter=False)

    arucosPosition: dict[int, Transform] = {}

    for a in movingArucos:
        if corners_position.get(a.id) is None:
            continue
        else:
            aruco2world, ar_met = locateAruco(a, corners_position[a.id], camera)
            if directUpdate:
                a.updatePose(aruco2world)
            arucosPosition[a.id] = aruco2world

    return rvec, tvec, arucosPosition, metrics_collected


def PnP(image_points, object_points, mtx, dist, getMetrics=False) -> tuple[list, list]:
    """
    Estimate the pose of an the camera using PnP.

    :param image_points: The 2D points of the object in the image
    :param object_points: The 3D points of the object
    :param mtx: The camera matrix
    :param dist: The distortion coefficients
    :return: The rotation and translation vectors
    """

    image_points = np.array(image_points, dtype=np.float32)

    # Solve PnP to estimate rotation and translation

    try:
        success, rvec, tvec = cv.solvePnP(object_points, image_points, mtx, dist)
    except cv.error as e:
        print("An error occured while solving PnP")
        print(
            f"Object point n°: {len(object_points)}, Image point n°: {len(image_points)}"
        )
        raise e

    if success:
        cv.solvePnPRefineLM(
            object_points,
            image_points,
            mtx,
            dist,
            rvec,
            tvec,
        )
        metrics = {}
        if getMetrics:
            projected, _ = cv.projectPoints(
                object_points,
                rvec,
                tvec,
                mtx,
                dist,
            )
            projected = projected.squeeze(1)

            distances = np.linalg.norm(projected - image_points, axis=1)

            metrics["MAE"] = np.mean(np.abs(distances))

            metrics["RMSE"] = np.sqrt(np.mean(distances**2))

        return rvec, tvec, metrics
    else:
        raise Exception("Could not solve PnP")


def processArucoFromMultipleCameras(
    fixedArucos: list[Aruco],
    movingArucos: dict[int, Aruco],
    cameras: list[Camera],
    frame: list[list],
    getMetrics=False,
):
    assert len(cameras) == len(frame)
    # This dict will contains each "variant" of the positions of each arucos corner. One for each time a camera see them
    arucosPositions: dict[int, list[np.ndarray[np.ndarray[float]]]] = {}
    for c, f in zip(cameras, frame):
        res = processAruco(
            fixedArucos,
            movingArucos.values(),
            c,
            f,
            accept_none=True,
            directUpdate=False,
            metrics=getMetrics,
        )
        if not res:
            continue
        rvec, tvec, ap, met = res
        for key in ap.keys():
            if key not in arucosPositions:
                arucosPositions[key] = []

            corners = Aruco.getCornersFromTopLeft(
                np.array([-movingArucos[key].size / 2, movingArucos[key].size / 2, 0]),
                movingArucos[key].size,
            )
            finalCorners = []
            for c in corners:
                finalCorners.append(ap[key].apply(c))
            arucosPositions[key].append(finalCorners)

    for a in movingArucos.values():
        if a.id in arucosPositions:
            if getMetrics:
                nbrDims = 3  # We're in 3d
                nbrCorner = 4  # We have 4 corners for now

                arr = np.array(arucosPositions[a.id])
                # Get all unique pairs of indices
                indices = list(combinations(range(len(arr)), 2))

                # Compute differences for unique pairs
                diffs = np.array([arr[i] - arr[j] for i, j in indices])

                vals = np.square(diffs) / len(arr)  # val ** 2 / nbr_of_observed_values

                err = np.sum(
                    vals, axis=0
                )  # This gives us an error per coordinate per corner for this aruco marker

                err = np.sum(err, axis=0) / nbrCorner  # This gives us an error per axes

                RMSE_per_axes = np.sqrt(err)

                RMSE_total = np.sum(RMSE_per_axes) / nbrDims

            average_positions = np.mean(arucosPositions[a.id], axis=0)
            for i, newPos in enumerate(average_positions):
                a.corners[i].updatePos(newPos)


if __name__ == "__main__":
    # getArucosFromPaper()
    for i in range(5, 7):
        generate_aruco_marker(i, marker_size=200)
    pass