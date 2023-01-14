import os
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import project_utils as utils
import settings
import streamlink

import torch
import time
# import tensorflow as tf
# import core.utils as utilsYolo
# from core.yolov3 import YOLOv3, decode
# from PIL import Image

torch.cuda.is_available()

def yolo_results():
    return

def main(source_path=None):

    # load in video capture for source
    if source_path is None:
        vid = cv2.VideoCapture("videos/highway2.mov")
    else:
        vid = cv2.VideoCapture(source_path)

    # set default lane line parameters
    utils.create_coordinates.prev_line_parameters_left = (1,1)
    utils.create_coordinates.prev_line_parameters_right = (1,1)

    utils.display_lines.line_average = [None for i in range(200)]
    utils.display_lines.counter = 0
    utils.display_lines.detected_lines_counter = 0
    utils.display_lines.prev_avg_lines_center = None
    utils.display_lines.prev_avg_lines = None

    car_counter = 0

    lost_counter = 0
    LOST_COUNTER_THRES = 20

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    read_once = False

    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    yolo_results.results = None

    # loop to process each frame of video
    while vid.isOpened():
        _, im1 = vid.read()

        # im1 = cv2.rotate(im1, cv2.cv2.ROTATE_90_CLOCKWISE)
        im1 = cv2.rotate(im1, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)

        im_canny = utils.get_canny_image(im1)

        # YOLO
        if car_counter % 1 == 0:
            yolo_results.results = model(im1)
        results = yolo_results.results
        # print(results.pandas().xyxy[0])
        cv2.imshow("predicted image", results.render()[0])
        car_counter += 1

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
        averaged_lines, processed_lines, use_car = utils.average_slope_intercept(im1, lines) 

        # debug flag to show all found lines
        if settings.modifiers.debug:
            for x in range(0, len(processed_lines)):
                for x1,y1,x2,y2 in processed_lines[x]:
                    cv2.line(im_mod,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.imshow('hough',im_mod)
        
        use_car_detect = False
        if use_car:
            lost_counter += 1
        else:
            lost_counter = 0

        if lost_counter > LOST_COUNTER_THRES:
            use_car_detect = True

        # add line markings to image
        line_image = utils.display_lines(im1, averaged_lines, use_car_detect, results.xyxy[0].cpu())
        processed_image = cv2.addWeighted(im1, 0.8, line_image, 1, 1) 

        # add boxes around detected cars
        # for (x,y,w,h,u) in avg_car_list:
        #     cv2.rectangle(processed_image,(x,y),(x+w,y+h),(0,255,255),1)
        # for car in results.xyxy[0]:
        #     x1, y1, x2, y2 = int(car[0]), int(car[1]), int(car[2]), int(car[3])
        #     cv2.rectangle(processed_image,(x1,y1),(x2,y2),(0,255,255),1)

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