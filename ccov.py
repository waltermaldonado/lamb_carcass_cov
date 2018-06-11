# -*- coding: UTF-8 -*-
import cv2
import numpy as np


def carcass_coverage(image_path, show_steps=False, save_steps=False):

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

    structuring_el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilation_hang = cv2.dilate(mask_hang1, structuring_el, iterations=3)
    # End: HANG PIXELS RANGE

    # Start: TOTAL PIXELS RANGE
    lower_color_total = np.array((0, 30, 30), dtype=np.uint8, ndmin=1)
    upper_color_total = np.array((255, 255, 255), dtype=np.uint8, ndmin=1)
    mask_total = cv2.inRange(img_hsv, lower_color_total, upper_color_total)

    # Background and hang removal from the total area
    img_total_gray = mask_total - dilation_hang - mask_bg1
    _, img_total_gray = cv2.threshold(img_total_gray, 20, 255, cv2.THRESH_BINARY)

    structuring_el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opening_img_total_gray = cv2.morphologyEx(img_total_gray, cv2.MORPH_OPEN, structuring_el, iterations=1)
    structuring_el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closing_img_total_gray = cv2.morphologyEx(opening_img_total_gray, cv2.MORPH_CLOSE, structuring_el, iterations=3)

    selected_contours = []
    _, contours, hierarchy = cv2.findContours(closing_img_total_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > 20000:
            selected_contours.append(contour)

    if len(selected_contours) > 0:
        epsilon = 5
        approx = cv2.approxPolyDP(selected_contours[0], epsilon, True)

        img_total = np.zeros((640, 480), dtype=np.uint8)
        cv2.fillPoly(img_total, pts=[approx], color=255)

        # Quantifying total carcass area
        total_px = cv2.countNonZero(img_total)
        # End: TOTAL PIXELS RANGE

        # Start: RED PIXELS RANGE
        img_hsv_roi = img_hsv & cv2.cvtColor(img_total, cv2.COLOR_GRAY2BGR)
        lower_color_red1 = np.array((0, 70, 0), dtype=np.uint8, ndmin=1)
        upper_color_red1 = np.array((15, 255, 255), dtype=np.uint8, ndmin=1)
        mask1 = cv2.inRange(img_hsv_roi, lower_color_red1, upper_color_red1)
        lower_color_red2 = np.array((170, 70, 0), dtype=np.uint8, ndmin=1)
        upper_color_red2 = np.array((179, 255, 255), dtype=np.uint8, ndmin=1)
        mask2 = cv2.inRange(img_hsv_roi, lower_color_red2, upper_color_red2)

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
        img_fat_overlay = img_fat.copy()
        img_fat_overlay[np.where((img_fat_overlay != [0, 0, 0]).all(axis=2))] = [0, 255, 255]

        img_total_output = cv2.cvtColor(img_total, cv2.COLOR_GRAY2BGR)
        _, img_total_output = cv2.threshold(img_total_output, 50, 255, cv2.THRESH_BINARY)
        img_total_output = img & img_total_output
        overlay = cv2.addWeighted(img_total_output, 0.8, img_fat_overlay, 0.2, 0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cov_text = "Fat Coverage: %2.2f%%" % percentage_coverage
        cv2.putText(overlay, cov_text, (260, 630), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # Output related stuff: Shows or stores the steps in the image processing
        if show_steps | save_steps:
            img_total_range = img & cv2.cvtColor(closing_img_total_gray, cv2.COLOR_GRAY2BGR)
            img_total_range_c1 = img_total_range.copy()
            img_total_range_c2 = img_total_range.copy()
            cv2.drawContours(img_total_range_c1, [selected_contours[0]], -1, (0, 255, 0), 3)
            cv2.drawContours(img_total_range_c2, [approx], -1, (0, 255, 0), 3)

            if show_steps:
                cv2.imshow("1-img.JPG", img)
                cv2.imshow("2-img_hsv.JPG", img_hsv)
                cv2.imshow("3-mask_bg1.JPG", mask_bg1)
                cv2.imshow("4-mask_total.JPG", mask_total)
                cv2.imshow("5-img_total_gray.JPG", img_total_gray)
                cv2.imshow("6-closing_img_total_gray.JPG", closing_img_total_gray)
                cv2.imshow("7-img_total_range.JPG", img_total_range)
                cv2.imshow("8-img_total_range_c1.JPG", img_total_range_c1)
                cv2.imshow("9-img_total_range_c2.JPG", img_total_range_c2)
                cv2.imshow("10-img_fat.JPG", img & img_fat)

            if save_steps:
                cv2.imwrite("1-img.JPG", img)
                cv2.imwrite("2-img_hsv.JPG", img_hsv)
                cv2.imwrite("3-mask_bg1.JPG", mask_bg1)
                cv2.imwrite("4-mask_total.JPG", mask_total)
                cv2.imwrite("5-img_total_gray.JPG", img_total_gray)
                cv2.imwrite("6-closing_img_total_gray.JPG", closing_img_total_gray)
                cv2.imwrite("7-img_total_range.JPG", img_total_range)
                cv2.imwrite("8-img_total_range_c1.JPG", img_total_range_c1)
                cv2.imwrite("9-img_total_range_c2.JPG", img_total_range_c2)
                cv2.imwrite("10-img_fat.JPG", img & img_fat)
                cv2.imwrite("11-overlay.JPG", overlay)

        return overlay

    else:
        return np.zeros((640, 480))
