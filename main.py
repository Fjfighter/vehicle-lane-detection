import os
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import utils
import settings

def main(source_path=None):
    
    # load in video capture for source
    if source_path is None:
        vid = cv2.VideoCapture("videos/vid_sample_1.mp4")
    else:
        vid = cv2.VideoCapture(source_path)

    # set default lane line parameters
    utils.create_coordinates.prev_line_parameters_left = (1,1)
    utils.create_coordinates.prev_line_parameters_right = (1,1)

    utils.display_lines.line_average = [None for i in range(200)]
    utils.display_lines.counter = 0
    utils.display_lines.prev_avg_lines_center = None
    utils.display_lines.prev_avg_lines = None

    # init car detection classifier 
    # (classifer source trained from: https://github.com/afzal442/Real-Time_Vehicle_Detection-as-Simple)
    car_cascade = cv2.CascadeClassifier('cars.xml')

    CARS_CACHE_MAX = 20             # car cache limit
    NUM_DETECTIONS_THRES = 3        # num car detections threshold
    CLOSE_DIST_THRES = 10           # car point euclidian distance threshold
    CLOSE_SIZE_THRES = 20           # car box size threshold
    CLOSE_DIST_THRES_DAMPED = 15    # car point euclidian distance threshold on damped cache
    CAR_UPDATE_THRES = 6
    MAX_CAR_UPDATE = 30

    cars_cache = []
    cars_cache_mod = []
    avg_car_list = []
    avg_car_list_damped = []
    car_counter = 0

    # loop to process each frame of video
    while vid.isOpened():
        _, im1 = vid.read()

        im_canny = utils.get_canny_image(im1)

        # run object detection to find car matches
        if car_counter % 2 == 0:
            gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)

            # mask input image
            polygons = np.array([
                [[0, gray.shape[0]-gray.shape[0]/5], [gray.shape[1]/8, gray.shape[0]/4], [gray.shape[1]-gray.shape[1]/8, gray.shape[0]/4], [gray.shape[1], gray.shape[0]-gray.shape[0]/5]]
                ])
            mask = np.zeros_like(gray)
            cv2.fillPoly(mask, np.int32(polygons), 255) 
            masked_image = cv2.bitwise_and(gray, mask) 
            
            # detect and append cars to car cache
            cars = car_cascade.detectMultiScale(masked_image, 1.1, 3)
            cars_cache.append(cars)

            # remove old cars from cache if exceed cache limit
            if len(cars_cache) > CARS_CACHE_MAX:
                cars_cache.pop(0)

        # clear modified car cache
        if car_counter % 10 == 0:
            cars_cache_mod = []
            
        # clear average car cache 
        if car_counter % 20 == 0:
            avg_car_list = []

        # filter and average cars that meet threshold critera
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
                        
                        # compare if within thresholds
                        if utils.getDist(point1, point2) < CLOSE_DIST_THRES and abs(yw-iw) < CLOSE_SIZE_THRES and abs(yh-ih) < CLOSE_SIZE_THRES:
                            num_detections += 1
                            avg_car_cache.append(cars_cache[x][y])
            
                # compare if detected enough times
                if num_detections > NUM_DETECTIONS_THRES:
                    cars_cache_mod.append(cars_cache[i])
                    avg_car_cache.append(cars_cache[i][j])

                    # calculate average box
                    calc_car_avg = np.average(avg_car_cache, axis=0).astype(int)

                    point1 = [calc_car_avg[0], calc_car_avg[1]]
                    aw, ah = calc_car_avg[2], calc_car_avg[3]
                    n = 0
                    update_value = 0
                    while n < len(avg_car_list):
                        point2 = [avg_car_list[n][0], avg_car_list[n][1]]
                        lw, lh = avg_car_list[n][2], avg_car_list[n][3]
                        
                        # remove similar and nearby detections already in cache
                        if utils.getDist(point1, point2) < CLOSE_DIST_THRES_DAMPED and abs(aw-lw) < CLOSE_DIST_THRES_DAMPED and abs(ah-lh) < CLOSE_DIST_THRES_DAMPED:
                            
                            # give lower update value for more similar detections found
                            if update_value > -MAX_CAR_UPDATE:
                                update_value -= (avg_car_list[n][4]+num_detections)
                            
                            avg_car_list.pop(n)
                        else:
                            avg_car_list[n][4] += 1
                            
                            # remove if box not been updated within update threshold
                            if avg_car_list[n][4] > CAR_UPDATE_THRES:
                                avg_car_list.pop(n)
                            else:
                                n += 1

                    # add to average car cache
                    avg_car_list.append(np.append(calc_car_avg, int(update_value)))

        car_counter += 1

        # debug flag to show real time car detections
        if settings.modifiers.debug:
            for (x,y,w,h) in cars:
                cv2.rectangle(masked_image,(x,y),(x+w,y+h),(0,255,255),2)
            cv2.imshow("car mask", masked_image)

        im_roi = utils.region_of_interest(im_canny)

        # debug flag to show masked lane detection image
        if settings.modifiers.debug:
            cv2.imshow("mask", im_roi)

        im_mod = im1.copy()

        # use houghlines to get all lines within image
        lines = cv2.HoughLinesP(im_roi, 2, np.pi / 180, 100, np.array([]), minLineLength = 40, maxLineGap = 5) 

        # skip over epoch if no lines detected 
        if lines is None:
            continue

        # get the average of all lines within slope threshold
        averaged_lines, processed_lines = utils.average_slope_intercept(im1, lines) 

        # debug flag to show all found lines
        if settings.modifiers.debug:
            for x in range(0, len(processed_lines)):
                for x1,y1,x2,y2 in processed_lines[x]:
                    cv2.line(im_mod,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.imshow('hough',im_mod)
        
        # add line markings to image
        line_image = utils.display_lines(im1, averaged_lines)
        processed_image = cv2.addWeighted(im1, 0.8, line_image, 1, 1) 

        # add boxes around detected cars
        for (x,y,w,h,u) in avg_car_list:
            cv2.rectangle(processed_image,(x,y),(x+w,y+h),(0,255,255),1)

        # display final results
        cv2.imshow("results", processed_image)

        # close window on q button
        if cv2.waitKey(1) & 0xFF == ord('q'):      
            break

    # close the video file
    vid.release() 
    cv2.destroyAllWindows() 

if __name__ == "__main__":
    main()