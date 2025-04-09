import cv2
import numpy as np



def get_walls(img_path):
    # Load the image
    polygons = []
    img = cv2.imread(img_path)

    # Convert the image to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(hsv, 80, 150, apertureSize=3)

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
    for line in lines:
        x1, y1, x2, y2 = line[0]
        polygons.append([x1, y1, x2, y2])
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)


    # Resize the image for display purposes
    img = cv2.resize(img, (640, 480))
    
    # Display the results
    cv2.imshow('Edges', cv2.resize(edges, (640, 480)))
    cv2.imshow('Dilated Edges', cv2.resize(dilated_edges, (640, 480)))
    cv2.imshow('Detected Walls', img)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return polygons

# Call the function with your image path
get_walls("img/warped_image3.jpg")
