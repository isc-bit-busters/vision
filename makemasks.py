import cv2
import os

import numpy as np

hsv_ranges = [
    ((92, 48, 65), (112, 108, 125)),
    ((90, 0, 0), (110, 59, 56)),
    ((96, 49, 70), (116, 109, 130)),
    ((95, 65, 59), (115, 125, 119)),
    ((89, 51, 0), (109, 111, 52)),
    ((94, 78, 15), (114, 138, 75)),
    ((97, 70, 44), (117, 130, 104)),
    ((98, 77, 1), (118, 137, 61)),
    ((98, 53, 25), (118, 113, 85)),
    ((98, 29, 26), (118, 89, 86)),
    ((97, 61, 29), (117, 121, 89)),
    ((102, 38, 26), (122, 98, 86)),
    ((96, 56, 50), (116, 116, 110)),
    ((100, 38, 15), (120, 98, 75)),
    ((80, 0, 18), (100, 46, 78)),
    ((99, 74, 36), (119, 134, 96)),
    ((100, 63, 36), (120, 123, 96)),
    ((98, 56, 38), (118, 116, 98)),
    ((96, 82, 52), (116, 142, 112)),
    ((102, 61, 12), (122, 121, 72)),
    ((101, 73, 17), (121, 133, 77)),
    ((110, 27, 0), (130, 87, 39)),
    ((0, 0, 0), (10, 30, 39)),
    ((0, 0, 0), (10, 30, 35)),
    ((101, 50, 2), (121, 110, 62)),
    ((100, 21, 0), (120, 81, 45)),
    ((0, 0, 0), (10, 30, 44)),
    ((110, 6, 0), (130, 66, 44)),
    ((99, 51, 14), (119, 111, 74)),
    ((100, 0, 0), (120, 57, 58)),
    ((104, 28, 0), (124, 88, 52)),
    ((0, 0, 0), (10, 30, 46)),
    ((100, 0, 15), (120, 47, 75)),
    ((20, 0, 10), (40, 43, 70)),
    ((93, 6, 19), (113, 66, 79)),
    ((110, 34, 0), (130, 94, 38)),
    ((100, 34, 0), (120, 94, 42)),
    ((100, 47, 0), (120, 107, 40)),
    ((97, 36, 82), (117, 96, 142)),
    ((110, 55, 0), (130, 115, 36)),
    ((105, 4, 0), (125, 64, 52)),   
    ((120, 0, 0), (140, 40, 52)), 
 ]


def get_images(img_path, mask_path,contours_path):
    img = cv2.imread(img_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    final_mask = None
    for min_hsv, max_hsv in hsv_ranges:
        lower = np.array(min_hsv)
        upper = np.array(max_hsv)
        mask = cv2.inRange(hsv, lower, upper)
        if final_mask is None:
            final_mask = mask
        else:
            final_mask = cv2.bitwise_or(final_mask, mask)

    mask = final_mask

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        area = cv2.contourArea(c)

        epsilon = 0.05 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        
        rect = cv2.minAreaRect(approx)
        box = cv2.boxPoints(rect)
        box = np.int8(box)  # Convert float coordinates to int

        w, h = rect[1]
        if w == 0 or h == 0:
            continue  # Avoid division by zero

        aspect_ratio = float(min(w, h)) / max(w, h)
        size = 30
        
        if area > 1_000 and (w > size and h > size) and ((0.2 < aspect_ratio and min(w, h) < 200) or (0.6 < aspect_ratio)):
            print(area, w, h, aspect_ratio)
            cv2.drawContours(img, c, -1, (0, 255, 0), 10)
    cv2.imwrite(mask_path, mask)
    cv2.imwrite(contours_path, img)
    return img,mask

if __name__ == "__main__":
    dir_path = './images'
    i=0
    for filename in os.listdir(dir_path):
        print()
        print(filename)

        img_path = os.path.join(dir_path, filename)
        mask_path = f'masks/mask{i}.jpeg'
        contours_path = f'contours/contours{i}.jpeg'
        i+=1
        img, mask = get_images(img_path, mask_path, contours_path)

        cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("img", 800, 600)

        cv2.namedWindow("mask", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("mask", 800, 600)

        cv2.imshow('img', img)
        cv2.imshow('mask', mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()