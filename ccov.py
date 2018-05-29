# -*- coding: UTF-8 -*-
import cv2
import numpy as np


def carcass_coverage(image_path):

    # Retrieving original image
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (480, 640))

    # HSV color model conversion
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Start: BACKGROUND REMOVAL
    lower_color_bg1 = np.array((10, 0, 0), dtype=np.uint8, ndmin=1)
    upper_color_bg1 = np.array((110, 255, 255), dtype=np.uint8, ndmin=1)
    mask_bg1 = cv2.inRange(img_hsv, lower_color_bg1, upper_color_bg1)
    # End: BACKGROUND REMOVAL

    # Start: HANG PIXELS RANGE
    lower_color_hang1 = np.array((0, 0, 0), dtype=np.uint8, ndmin=1)
    upper_color_hang1 = np.array((180, 255, 55), dtype=np.uint8, ndmin=1)
    mask_hang1 = cv2.inRange(img_hsv, lower_color_hang1, upper_color_hang1)
    mask_hang_hsv1 = cv2.cvtColor(mask_hang1, cv2.COLOR_GRAY2BGR)

    img_hang = img & mask_hang_hsv1
    _, img_hang = cv2.threshold(img_hang, 2, 255, cv2.THRESH_BINARY)
    img_hang = cv2.cvtColor(img_hang, cv2.COLOR_BGR2GRAY)
    structuring_el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilation = cv2.dilate(img_hang, structuring_el, iterations=3)
    # End: HANG PIXELS RANGE

    # Start: TOTAL PIXELS RANGE
    lower_color_total = np.array((0, 30, 30), dtype=np.uint8, ndmin=1)
    upper_color_total = np.array((255, 255, 255), dtype=np.uint8, ndmin=1)
    mask_total = cv2.inRange(img_hsv, lower_color_total, upper_color_total)
    mask_total_hsv = cv2.cvtColor(mask_total, cv2.COLOR_GRAY2BGR)

    img_total_range = img & mask_total_hsv

    img_total_gray = cv2.cvtColor(img_total_range, cv2.COLOR_BGR2GRAY)
    _, th_img_total_gray = cv2.threshold(img_total_gray, 20, 255, cv2.THRESH_BINARY)

    # Background and hang removal from the total area
    th_img_total_gray = th_img_total_gray - dilation - mask_bg1
    _, th_img_total_gray = cv2.threshold(th_img_total_gray, 20, 255, cv2.THRESH_BINARY)

    structuring_el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opening_img_total_gray = cv2.morphologyEx(th_img_total_gray, cv2.MORPH_OPEN, structuring_el, iterations=1)
    structuring_el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closing_img_total_gray = cv2.morphologyEx(opening_img_total_gray, cv2.MORPH_CLOSE, structuring_el, iterations=3)

    cntrs = []
    _, contours, hierarchy = cv2.findContours(closing_img_total_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > 20000:
            cntrs.append(contour)

    if len(cntrs) > 0:
        epsilon = 5
        approx = cv2.approxPolyDP(cntrs[0], epsilon, True)

        # Step-by-step image processing description. TODO: Remove after tests
        # cv2.drawContours(img_total_range, [approx], -1, (0, 255, 0), 3)

        img_total = np.zeros((640, 480), dtype=np.uint8)
        cv2.fillPoly(img_total, pts=[approx], color=255)

        # Quantifying total carcass area
        total_px = cv2.countNonZero(img_total)
        # End: TOTAL PIXELS RANGE

        # Start: RED PIXELS RANGE
        img_hsv = img_hsv & cv2.cvtColor(img_total, cv2.COLOR_GRAY2BGR)
        lower_color_red1 = np.array((0, 70, 0), dtype=np.uint8, ndmin=1)
        upper_color_red1 = np.array((15, 255, 255), dtype=np.uint8, ndmin=1)
        mask1 = cv2.inRange(img_hsv, lower_color_red1, upper_color_red1)
        lower_color_red2 = np.array((170, 70, 0), dtype=np.uint8, ndmin=1)
        upper_color_red2 = np.array((179, 255, 255), dtype=np.uint8, ndmin=1)
        mask2 = cv2.inRange(img_hsv, lower_color_red2, upper_color_red2)

        img_fat = img_total - mask1 - mask2

        # Quantifying red (non-fat) carcass area
        fat_px = cv2.countNonZero(img_fat)
        # End: RED PIXELS RANGE

        # Coverage percentage
        if total_px > 0:
            percentage_coverage = (fat_px / float(total_px)) * 100
        else:
            percentage_coverage = 0

        img_fat = cv2.cvtColor(img_fat, cv2.COLOR_GRAY2BGR)
        img_fat[np.where((img_fat != [0, 0, 0]).all(axis=2))] = [0, 255, 255]

        img_total_output = cv2.cvtColor(img_total, cv2.COLOR_GRAY2BGR)
        _, img_total_output = cv2.threshold(img_total_output, 50, 255, cv2.THRESH_BINARY)
        img_total_output = img & img_total_output
        overlay = cv2.addWeighted(img_total_output, 0.8, img_fat, 0.2, 0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cov_text = "Fat Coverage: %2.2f%%" % percentage_coverage
        cv2.putText(overlay, cov_text, (260, 630), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        return overlay

    else:
        return np.zeros((640, 480))
