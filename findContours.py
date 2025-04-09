import cv2
import numpy as np

image = cv2.imread('img/noObstaclescut1.jpeg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, 70, 150)

contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Or for lines
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)
kernel = np.ones((5, 5), np.uint8)
dilated = cv2.dilate(edges, kernel, iterations=1)

contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"Image shape: {image.shape}")
# Draw a line on the image

points = []
for c in contours:
    area   = cv2.contourArea(c)
    if area < 600 and area>250:  # Adjust the threshold as needed
        # Draw the contour on the original image
        cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
        x, y, w, h = cv2.boundingRect(c)
        # print(f"Contour located at x: {x}, y: {y}, width: {w}, height: {h}")
        # cv2.circle(image, (x + w // 2, y + h // 2), 5, (0, 0, 255), -1)  # Draw a point at the center of the bounding box
        points.append((x + w // 2, y + h // 2))


def filter_points(points, end,start):
 
    selected_points = []


    for p in points:
        if p[0] >= end*(image.shape[1]//6) and p[0] <= start*(image.shape[1] // 6):
            
                selected_points.append(p)
    new_points = []
    for s in selected_points: 
        if s[1] < 850 and s[1] > 200:
            new_points.append(s)
    selected_points = new_points
                    
    mean_x = np.median([pt[0] for pt in selected_points])
    print(f"Median X: {mean_x}")
      
    selected_points = sorted(selected_points, key=lambda pt: abs(pt[0] - mean_x))
    print(f"Selected points: {selected_points}")

    filtered_points = []
    for pt in selected_points:
        if not any(abs(pt[1] - fp[1]) < 50 for fp in filtered_points):
            filtered_points.append(pt)
    selected_points = filtered_points[:9]
    print(f"Filtered points: {selected_points}")
        # Remove points that are too distant from the mean X value
    for p in selected_points:
        threshold_distance = 100  # Adjust this threshold as needed
    
    
        new_selected_points = [pt for pt in selected_points if abs(pt[0] - mean_x) <= threshold_distance]
        selected_points = new_selected_points
        selected_points = sorted(selected_points, key=lambda pt: pt[1])
    for s in selected_points:
        cv2.circle(image, s, 5, (255, 0, 0), -1)  # Draw a point at the center of the bounding box
        print(
            f"Selected point located at x: {s[0]}, y: {s[1]}")
    return selected_points

def filter_pointsteest(points, end,start):
 
    selected_points = []


    for p in points:
        if p[1] >= end*(image.shape[0]//2) and p[1] <= start*(image.shape[0] // 2):
            
                selected_points.append(p)
    new_points = []
    for s in selected_points: 
        if s[1] < 900 and s[1] > 350:
            new_points.append(s)
    selected_points = new_points
            
    mean_x = np.mean([pt[1] for pt in selected_points])
    print(f"Median X: {mean_x}")
      
    selected_points = sorted(selected_points, key=lambda pt: abs(pt[0] - mean_x))
    print(f"Selected points: {selected_points}")

    # filtered_points = []
    # for pt in selected_points:
    #     if not any(abs(pt[0] - fp[0]) < 10 for fp in filtered_points):
    #         filtered_points.append(pt)
    # selected_points = filtered_points[:9]
    # print(f"Filtered points: {selected_points}")
    #     # Remove points that are too distant from the mean X value
        # Group points by their y-coordinate


    for p in selected_points:
        threshold_distance = 30 # Adjust this threshold as needed
    
    
        new_selected_points = [pt for pt in selected_points if abs(pt[1] - mean_x) <= threshold_distance]
        selected_points = new_selected_points
        selected_points = sorted(selected_points, key=lambda pt: pt[1])
    for s in selected_points:
        cv2.circle(image, s, 5, (255, 0, 0), -1)  # Draw a point at the center of the bounding box
        print(
            f"Selected point located at x: {s[0]}, y: {s[1]}")
        
    return selected_points

def find_walls(selected_points):
    isFree = [True,True,True]
    for point_col in range(len(selected_points)):
        for point in range (1,len(selected_points[point_col])):
            if selected_points[point_col][point][1] - selected_points[point_col][point-1][1] > 100: 
                cv2.line(image, selected_points[point_col][point-1], selected_points[point_col][point], (0, 0, 255), 2)

                isFree.append(False)
            
            if 837 - selected_points[point_col][len(selected_points[point_col])-1][1] > 200:
                center = selected_points[point_col][len(selected_points[point_col])-1]
                cv2.circle(image, center, 6, (0, 0, 255), -1)
                isFree = [False,False,False]
            print(point_col,selected_points[point_col][0][1])
            if selected_points[point_col][0][1]- 200  > 150:
                center = selected_points[point_col][0]
                cv2.circle(image, center, 6, (0, 0, 255), -1)
                isFree = [False,False,False]

            if point % 3 == 0 and not any(isFree):
               cv2.line(image, selected_points[point_col][point-1], selected_points[point_col][point], (0, 0, 255), 2)
               print(f"Wall detected between points{point_col}: {selected_points[point_col][point-1]} and {selected_points[point_col][point]}")
               isFree=[True,True,True]
                

            

points_filtered = []
for i in range(1,3):
    points_filtered.append(filter_pointsteest(points, i-1,i))

    cv2.imshow("Detected Walls", image)
    cv2.waitKey(0)
print(points_filtered)
find_walls(points_filtered)
cv2.imshow("Detected Walls", image)
cv2.waitKey(0)