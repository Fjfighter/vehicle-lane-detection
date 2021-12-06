import os
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
# from IPython.display import clear_output
import utils
import settings

# images location
im1 = './images/lane_sample_3.jpeg'

# Load an color image in grayscale
# im1 = cv2.imread(im1)

# plt.imshow(im1)
# plt.plot()

vid = cv2.VideoCapture("videos/vid_sample_1.mp4")

car_cascade = cv2.CascadeClassifier('cars.xml')

utils.create_coordinates.prev_line_parameters_left = (1,1)
utils.create_coordinates.prev_line_parameters_right = (1,1)

utils.display_lines.line_average = [None for i in range(200)]
utils.display_lines.counter = 0
utils.display_lines.prev_avg_lines_center = None
utils.display_lines.prev_avg_lines = None

cars_cache_max = 20
NUM_DETECTIONS_THRES = 3
CLOSE_DIST_THRES = 8
cars_cache = []
cars_cache_mod = []
avg_car_list = []
car_counter = 0

while vid.isOpened():
    _, im1 = vid.read()

    im_canny = utils.canny_edge_detector(im1)

    if car_counter % 2 == 0:
        gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)

        polygons = np.array([
            [[0, gray.shape[0]-gray.shape[0]/5], [gray.shape[1]/3, gray.shape[0]/3], [gray.shape[1]-gray.shape[1]/3, gray.shape[0]/3], [gray.shape[1], gray.shape[0]-gray.shape[0]/5]]
            ])
        mask = np.zeros_like(gray)
        cv2.fillPoly(mask, np.int32(polygons), 255) 
        masked_image = cv2.bitwise_and(gray, mask) 
        cars = car_cascade.detectMultiScale(masked_image, 1.15, 4)

        # populate car cache on first run
        # if car_counter == 0:
        #     for i in range(cars_cache_max):
        #         cars_cache[i] = cars

        # shift cars cache
        # for i in range(cars_cache_max-1):
        #     cars_cache[i+1] = cars_cache[i]

        # cars_cache[0] = cars

        cars_cache.append(cars)

        if len(cars_cache) > cars_cache_max:
            cars_cache.pop(0)

    if car_counter % 10 == 0:
        cars_cache_mod = []
        
    if car_counter % 10 == 0:
        avg_car_list = []

    for i in range(len(cars_cache)):            
        for j in range(len(cars_cache[i])):
            num_detections = 0
            avg_car_cache = []

            for x in range(len(cars_cache)):
                for y in range(len(cars_cache[x])):

                    if y == j:
                        continue

                    point1 = [cars_cache[i][j][0], cars_cache[i][j][1]]
                    point2 = [cars_cache[x][y][0], cars_cache[x][y][1]]
                    yw, yh = cars_cache[x][y][2], cars_cache[x][y][3]
                    iw, ih = cars_cache[i][j][2], cars_cache[i][j][3]
                    if utils.getDist(point1, point2) < CLOSE_DIST_THRES and abs(yw-iw) < CLOSE_DIST_THRES and abs(yh-ih) < CLOSE_DIST_THRES:
                        num_detections += 1
                        avg_car_cache.append(cars_cache[x][y])
        
            if num_detections > NUM_DETECTIONS_THRES:
                cars_cache_mod.append(cars_cache[i])
                avg_car_cache.append(cars_cache[i][j])

                calc_car_avg = np.average(avg_car_cache, axis=0).astype(int)
                if not any((np.array(calc_car_avg) == n).all() for n in np.array(avg_car_list)): # (np.array(calc_car_avg).all() == np.array(avg_car_list).any()):
                    # print(calc_car_avg)
                    avg_car_list.append(calc_car_avg)


    car_counter += 1

    if settings.modifiers.debug:
        cv2.imshow("car mask", masked_image)

    # print(np.array(polygon, dtype=np.int32))

    im_roi = utils.region_of_interest(im_canny)
    # print(im_roi.shape)
    # plt.imshow(im_roi)
    # plt.show()

    # print(im_canny.shape)

    # plt.imshow(im_canny)
    # plt.show()

    if settings.modifiers.debug:
        cv2.imshow("mask", im_roi)

    im_mod = im1.copy()

    lines = cv2.HoughLinesP(im_roi, 2, np.pi / 180, 100, 
                                np.array([]), minLineLength = 40, 
                                maxLineGap = 5) 

    # print(lines.shape)

    if lines is None:
        continue

    lines_reduced = lines.copy()
    lines_x = lines[:,:,0].reshape(lines.shape[0])
    line = 0

    # while True:
    #     if line >= lines_reduced.shape[0]:
    #         break
    # #     print("here")
    #     lines_x_rem = np.delete(lines_x, line)
    # #     print(min(abs(lines_x_rem - lines[line,0,0])))
    #     if min(abs(lines[:][0] - lines[line][0])).any() < 5:
    # #         lines_reduced = np.delete(lines_reduced, line, axis=0)
    # #         lines_reduced = np.delete(lines_reduced, np.argwhere(lines_reduced == lines[line]), axis=0)
    #         lines_reduced = np.delete(lines_reduced, line, axis=0)
    #     else: 
    #         line += 1

    # print(lines_reduced.shape)

    
    # plt.imshow(im_mod)
    # plt.show()

    averaged_lines, processed_lines = utils.average_slope_intercept(im1, lines) 
    
    for x in range(0, len(processed_lines)):
        for x1,y1,x2,y2 in processed_lines[x]:
            cv2.line(im_mod,(x1,y1),(x2,y2),(0,255,0),2)

    if settings.modifiers.debug:
        cv2.imshow('hough',im_mod)

    lines_bottom = []
    lines_top = []

    for i in range(len(lines)):
        x1,y1,x2,y2 = lines[i][0]
        if y1 > im1.shape[0]-im1.shape[0]/4 or y2 > im1.shape[0]-im1.shape[0]/4:
            lines_bottom.append(lines[i])
        else:
            lines_top.append(lines[i])

    averaged_lines_bottom, processed_lines_bottom = utils.average_slope_intercept(im1, lines_bottom) 
    averaged_lines_top, processed_lines_top = utils.average_slope_intercept(im1, lines_top) 
    
    # line_image_top = utils.display_lines_2(im1, averaged_lines_top, averaged_lines_bottom)
    # line_image_bottom = utils.display_lines(im1, averaged_lines_bottom)
    # cv2.imshow("top", line_image_top)
    # cv2.imshow("bottom", line_image_bottom)
    
    line_image = utils.display_lines(im1, averaged_lines)
    combo_image = cv2.addWeighted(im1, 0.8, line_image, 1, 1) 

    # for c in avg_car_list:
    for (x,y,w,h) in avg_car_list:
        cv2.rectangle(combo_image,(x,y),(x+w,y+h),(0,255,255),2)

    cv2.imshow("results", combo_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):      
        break
    # plt.imshow(combo_image)
    # plt.show()

# close the video file
vid.release() 
  
# destroy all the windows that is currently on
cv2.destroyAllWindows() 