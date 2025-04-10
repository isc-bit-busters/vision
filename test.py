import cv2
import numpy as np


def get_walls(img_path):
    # Load the image
    polygons = []
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not open or read image at {img_path}")
        return []
    print(img.shape)
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(hsv, 150, 150, apertureSize=3)

    # Dilate the edges to join nearby edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Apply a mask to filter out unwanted areas (optional, based on specific needs)

    # Set a minimum threshold for contour area (to filter noise)
    min_area = 500

    # Filter out contours that are too small
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > min_area and cv2.contourArea(contour) < 10000]

    # Draw rectangles around the filtered contours
    for contour in filtered_contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw rectangles in blue
    lines = cv2.HoughLinesP(dilated_edges, 1, np.pi/180, threshold=100, minLineLength=130, maxLineGap=15)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            polygons.append([x1, y1, x2, y2])

            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    unique_polygons = []
    for p in polygons:
        if not any(abs(p[0] - up[0]) < 100 and abs(p[1] - up[1]) < 100 and abs(p[2] - up[2]) < 150 and abs(p[3] - up[3]) < 150 for up in unique_polygons):
            unique_polygons.append(p)

    # Resize the image for display purposes
    
    # Display the results
    cv2.imshow('Edges', cv2.resize(edges, (640, 480)))
    cv2.imshow('Dilated Edges', cv2.resize(dilated_edges, (640, 480)))
    cv2.imshow('Detected Walls', img)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    output_img = np.zeros(img.shape, dtype=np.uint8)

    # Make it so the first point is always the top left corner and the last point is always the bottom right corner
    for i in range(len(unique_polygons)):
        if unique_polygons[i][0] > unique_polygons[i][2]:
            unique_polygons[i][0], unique_polygons[i][2] = unique_polygons[i][2], unique_polygons[i][0]
        if unique_polygons[i][1] > unique_polygons[i][3]:
            unique_polygons[i][1], unique_polygons[i][3] = unique_polygons[i][3], unique_polygons[i][1]

    for p in unique_polygons:
        cv2.rectangle(output_img, (p[0], p[1]), (p[2], p[3]), (0, 0, 255), 2)  # Draw rectangles in red
        cv2.circle(output_img, (p[0], p[1]), 5, (255, 0, 0), -1)
        cv2.circle(output_img, (p[2], p[3]), 5, (0, 255, 0), -1)
        
    np.array(unique_polygons).tofile("polygons.txt")

    # Display the result
    cv2.imshow('Unique Polygons', output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
        
    return unique_polygons

def detect_cubes(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cube_positions = []

    def find_cubes(thresh_img, label_color):
        contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 150 < area < 3000:
                # Approximate shape
                epsilon = 0.04 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)

                if len(approx) == 4 and cv2.isContourConvex(approx):
                    # Compute center
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        cube_positions.append((cx, cy))

                        # Draw detected shape
                        cv2.drawContours(image, [approx], 0, label_color, 2)
                        cv2.circle(image, (cx, cy), 4, label_color, -1)

    # Threshold for black cubes
    _, thresh_black = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
    find_cubes(thresh_black, (0, 255, 0))

    # Threshold for white cubes
    _, thresh_white = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    find_cubes(thresh_white, (255, 0, 0))

    # Show result
    cv2.imshow("Detected Cubes (Rotation-Invariant)", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return cube_positions

# === Run detection ===
cube_positions = detect_cubes("img/warped_image3.jpg")
print("Cube positions (pixels):")
for i, (x, y) in enumerate(cube_positions):
    print(f"Cube {i+1}: x = {x}, y = {y}")

# Call the function with your image path
pol = get_walls("img/warped_image3.jpg")