from colorBounds import *
import cv2
import os

import numpy as np

# low_bound = ColorBound().min_black
# up_bound = ColorBound().max_black

# low_bound = ColorBound().min_black
# up_bound = ColorBound().max_black

min_black_1 = np.array([90, 0, 0])
max_black_1 = np.array([150, 255, 95])

min_black_2 = np.array([0, 0, 0])
max_black_2 = np.array([179, 30, 30])



def get_images(img_path, mask_path,contours_path):
    img = cv2.imread(img_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # mask = cv2.inRange(hsv, low_bound, up_bound)
    mask1 = cv2.inRange(hsv, min_black_1, max_black_1)
    mask2 = cv2.inRange(hsv, min_black_2, max_black_2)
    mask = cv2.bitwise_or(mask1, mask2)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        area = cv2.contourArea(c)
        if area > 6000:
        # if area > 1000 and area < 100_000:
            cv2.drawContours(img, c, -1, (0, 255, 0), 10)
    cv2.imwrite(mask_path, mask)
    cv2.imwrite(contours_path, img)
    return img,mask

if __name__ == "__main__":
    i=0
    for filename in os.listdir('./images/black'):
        img_path = os.path.join('./images/black', filename)
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