from skimage import draw
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import cv2
import math
import settings


def region_of_interest(image, top_offset=30, trap_width=300):
    '''
    Returns masked image with set parameters
    image: np array of image to be masked
    top_offset: y coord of top of mask
    trap_width: left and right offsets for top mask
    '''

    height, width = image.shape

    # create polygon mask
    polygons = np.array([
        [(0, height), (width, height), (int(image.shape[1]/2)+trap_width, int(image.shape[0]/2)+top_offset), (int(image.shape[1]/2)-trap_width, int(image.shape[0]/2)+top_offset)]
        ])
    mask = np.zeros_like(image)
      
    # use polygon to create mask and mask image
    cv2.fillPoly(mask, polygons, 255) 
    masked_image = cv2.bitwise_and(image, mask) 
    
    return masked_image

def get_canny_image(image):
    '''
    Returns canny outline of image
    image: np array of image to be modified
    '''

    # convert the image color to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) 
    
    # blur and convert to canny image
    blur = cv2.GaussianBlur(gray_image, (5, 5), cv2.BORDER_DEFAULT) 
    canny = cv2.Canny(blur, 50, 150)
    
    # re-merge image back into 3 channels
    im_final = np.zeros_like(image)
    im_final[:,:,0] = canny
    im_final[:,:,1] = canny
    im_final[:,:,2] = canny
    
    return canny

def create_coordinates(image, line_parameters, type):
    '''
    Returns coordinates of lane lines given line parameters
    image: np array of image to draw lane lines on
    line_parameters: intercept and slope of lane line
    type: which side lane marking (left or right)
    '''

    use_car = False

    # update prev lane marking with new marking
    try:
        slope, intercept = line_parameters
        if type == "left":
            create_coordinates.prev_line_parameters_left = line_parameters
        elif type == "right":
            create_coordinates.prev_line_parameters_right = line_parameters
    
    # if no new lane marking keep prev marking
    except TypeError:
        use_car = True
        if type == "left":
            slope, intercept = create_coordinates.prev_line_parameters_left
        elif type == "right":
            slope, intercept = create_coordinates.prev_line_parameters_right
    
    # calculate coordinates based on slope and intercept
    y1 = image.shape[0]
    y2 = int(y1 * (3 / 5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    
    return np.array([x1, y1, x2, y2]), use_car

def average_slope_intercept(image, lines, threshold_colors=None):
    '''
    Returns the average slope and intercept of all the slopes meeting threshold
    image: np array of image where lines are
    lines: all lines within image
    '''
    
    left_fit = []
    right_fit = []
    processed_lines = []
    
    # step through all lines
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
          
        # get slope and intercept of line
        parameters = np.polyfit((x1, x2), (y1, y2), 1) 
        slope = parameters[0]
        intercept = parameters[1]
        
        # ignore if slope too steep
        if abs(slope) < 0.40:
            continue

        # # ignore if line does not fall in color range
        # line_pixels = []
        # for y in range(y1, y2):
        #     for x in range(x1, x2):
        #         if y == slope * x + intercept:
        #             line_pixels.append([image[y][x]])
        # # print(np.array(line_pixels).shape)
        # if threshold_colors is not None:
        #     rgb = np.mean(line_pixels, axis=(0,1))
        #     if not (rgb > threshold_colors[0]).all() or not (rgb < threshold_colors[1]).all():
        #         continue

        processed_lines.append(line)
        
        # find center of image for weight modifiers
        im_center = int(image.shape[1]/2)
        im_center_height = int(image.shape[0]/3)

        x_center_weight_modifier = 0.5
        y_center_weight_modifier = 3

        # right line
        if slope < 0:
            
            # calculate and weight lines based on position
            weight = int(x1*x_center_weight_modifier+abs(y1-im_center_height)*y_center_weight_modifier)
            for i in range(weight):
                left_fit.append((slope, intercept))
        
        # left line
        else:

            # calculate and weight lines based on position
            weight = int((image.shape[1]-x1)*x_center_weight_modifier+abs(y1-im_center_height)*y_center_weight_modifier)
            for i in range(weight):
                right_fit.append((slope, intercept))
              
    # get average of left and right lane lines
    if len(left_fit) != 0:
        left_fit_average = np.mean(left_fit, axis = 0)
    else:
        left_fit_average = float('nan')

    if len(right_fit) != 0:
        right_fit_average = np.mean(right_fit, axis = 0)
    else:
        right_fit_average = float('nan')
    
    # create coordinates for lane lines
    left_line, use_car1 = create_coordinates(image, left_fit_average, "left")
    right_line, use_car2 = create_coordinates(image, right_fit_average, "right")
    
    return np.array([left_line, right_line]), processed_lines, use_car1 or use_car2

def display_lines(image, lines, use_car, detect_cars=None, car_confidence_thres=0.5, detected_lines_thres=8, car_size_thres=80):
    '''
    Returns calculated and drawn image of detected lane lines
    image: np array of image where lanes are
    lines: left and right lane markings
    '''
    display_lines.counter += 1
    line_image = np.zeros_like(image)

    use_car_detection = False
    closest_car = None
    closest_car_x_center = None
    closest_car_x = float('inf')
    largest_car = 0

    # display prev lane markings if no new lane markings detected
    if np.array(display_lines.prev_avg_lines).any() == None:
        use_car_detection = True
        display_lines.prev_avg_lines = lines

        print("no lines detected")
    
    # print(closest_car_x, closest_car_x_center)

    # calculate center offset for top and bottom parts of line
    left_offset_x1 = int((lines[0][0]-display_lines.prev_avg_lines[0][0])/3)
    left_offset_x2 = int((lines[0][2]-display_lines.prev_avg_lines[0][2])/3)
    right_offset_x1 = int((lines[1][0]-display_lines.prev_avg_lines[1][0])/3)
    right_offset_x2 = int((lines[1][2]-display_lines.prev_avg_lines[1][2])/3)
    
    # add offset to prev avg line
    display_lines.prev_avg_lines = [
        [display_lines.prev_avg_lines[0][0]+left_offset_x1, display_lines.prev_avg_lines[0][1], display_lines.prev_avg_lines[0][2]+left_offset_x2, display_lines.prev_avg_lines[0][3]],
        [display_lines.prev_avg_lines[1][0]+right_offset_x1, display_lines.prev_avg_lines[1][1], display_lines.prev_avg_lines[1][2]+right_offset_x2, display_lines.prev_avg_lines[1][3]] 
    ]

    # # draw the updated average line (green marking)
    # for x1, y1, x2, y2 in display_lines.prev_avg_lines:
    #     cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 10)

    # get average between average lane lines
    avg_x1 = int((display_lines.prev_avg_lines[0][0]+display_lines.prev_avg_lines[1][0])/2)
    avg_x2 = int((display_lines.prev_avg_lines[0][2]+display_lines.prev_avg_lines[1][2])/2)
    avg_y1 = int(display_lines.prev_avg_lines[0][1])
    avg_y2 = int(display_lines.prev_avg_lines[0][3])

    # fill cache on init
    if np.array(display_lines.line_average).any() == None:
        for i in range(len(display_lines.line_average)):
            display_lines.line_average[i] = np.array([avg_x1, avg_y1, avg_x2, avg_y2])

    if display_lines.counter % 5 == 0:

        # shift cache
        for i in range(len(display_lines.line_average)-1):
            display_lines.line_average[i+1] = display_lines.line_average[i]

        # insert new calculation into cache
        display_lines.line_average[0] = np.array([avg_x1, avg_y1, avg_x2, avg_y2])
    
    # get the average of cache
    average_lines = np.average(np.array(display_lines.line_average), axis=0)

    # init prev avg center line
    if np.array(display_lines.prev_avg_lines_center).any() == None:
        display_lines.prev_avg_lines_center = average_lines

    # init prev avg line
    if np.array(display_lines.prev_avg_lines_center).any() == None:
        display_lines.prev_avg_lines = lines

    if display_lines.prev_avg_lines[1][2]-display_lines.prev_avg_lines[0][2] < 30:
        display_lines.detected_lines_counter = 0

    if use_car or display_lines.prev_avg_lines[1][2]-display_lines.prev_avg_lines[0][2] < 0:
        # find car thats in center
        if detect_cars is not None:
            for car in detect_cars:
                if car[5] == 2 or car[5] == 5 or car[5] == 7:
                    car_center_x_coord = (car[2] - car[0]) / 2 + car[0]
                    if display_lines.prev_avg_lines_center is None:
                        continue
                    # car size
                    car_size = (car[2] - car[0]) * (car[3] - car[1])
                    center_dist = abs(display_lines.prev_avg_lines_center[2] - car_center_x_coord)
                    if car_size < car_size_thres**2:
                        continue
                    # check if car within lane
                    if center_dist > 100 and (car_center_x_coord < image.shape[1]*2/5 or car_center_x_coord > image.shape[1]*3/5):
                        continue
                    if car[4] > car_confidence_thres and center_dist < closest_car_x and car_size > largest_car:
                        closest_car_x = abs(display_lines.prev_avg_lines_center[2] - car_center_x_coord)
                        closest_car = car
                        closest_car_x_center = car_center_x_coord
                        largest_car = car_size

    if closest_car_x_center is None:
        display_lines.detected_lines_counter += 1

        if display_lines.detected_lines_counter > detected_lines_thres:
            # calculate center offset for top and bottom parts of line
            center_offset_x1 = int((average_lines[0]-display_lines.prev_avg_lines_center[0])/3)
            center_offset_x2 = int((average_lines[2]-display_lines.prev_avg_lines_center[2])/3)

            if abs(display_lines.prev_avg_lines_center[3] - avg_y2) > 40:
                if display_lines.prev_avg_lines_center[3] > avg_y2:
                    display_lines.prev_avg_lines_center[3] -= 10
                else:
                    display_lines.prev_avg_lines_center[3] += 10
            else:
                display_lines.prev_avg_lines_center[3] = avg_y2

            # add offset to prev avg line
            display_lines.prev_avg_lines_center = [display_lines.prev_avg_lines_center[0]+center_offset_x1, display_lines.prev_avg_lines_center[1], display_lines.prev_avg_lines_center[2]+center_offset_x2, display_lines.prev_avg_lines_center[3]] 
        

    # if closest_car_x_center is not None:
    #     display_lines.prev_avg_lines_center[2] = closest_car_x_center
    #     display_lines.prev_avg_lines_center[0] = image.shape[1] / 2
        # top_center = closest_car_x_center
        # bottom_center = image.shape[1] / 2
        # center_offset_x1 = int((top_center-display_lines.prev_avg_lines_center[0])/10)
        # center_offset_x2 = int((bottom_center-display_lines.prev_avg_lines_center[2])/10)

    # # add offset to prev avg line
    # display_lines.prev_avg_lines_center = [display_lines.prev_avg_lines_center[0]+center_offset_x1, display_lines.prev_avg_lines_center[1], display_lines.prev_avg_lines_center[2]+center_offset_x2, display_lines.prev_avg_lines_center[3]] 

    # flag to enable dynamic visual adjustment of center line thickness
    if settings.modifiers.dynamic_center_line: 
        poly_bottom_diff = int((display_lines.prev_avg_lines[0][0]-display_lines.prev_avg_lines[1][0])/15)
        poly_top_diff = int((display_lines.prev_avg_lines[0][2]-display_lines.prev_avg_lines[1][2])/12)
    else: 
        poly_bottom_diff = 60
        poly_top_diff = 10

    if closest_car_x_center is not None:
        if closest_car_x_center - display_lines.prev_avg_lines_center[2] > 100:
            # print(closest_car_x_center - display_lines.prev_avg_lines_center[2])
            display_lines.prev_avg_lines_center[2] = display_lines.prev_avg_lines_center[2] + 30
        elif closest_car_x_center - display_lines.prev_avg_lines_center[2] < -100:
            display_lines.prev_avg_lines_center[2] = display_lines.prev_avg_lines_center[2] - 30
        else:
            display_lines.prev_avg_lines_center[2] = closest_car_x_center

        if image.shape[1] / 2 - display_lines.prev_avg_lines_center[0] > 50:
            display_lines.prev_avg_lines_center[0] = display_lines.prev_avg_lines_center[0] + 30
        elif image.shape[1] / 2 - display_lines.prev_avg_lines_center[0] < -50:
            display_lines.prev_avg_lines_center[0] = display_lines.prev_avg_lines_center[0] - 30
        else:
            display_lines.prev_avg_lines_center[0] = image.shape[1] / 2
        
        # lock to bottom of objective vehicle
        max_bottom = int(((closest_car[3]-closest_car[1])/4).item())
        if closest_car[3] > display_lines.prev_avg_lines_center[3]:
            if closest_car[3] - display_lines.prev_avg_lines_center[3] > max_bottom + 20:
                display_lines.prev_avg_lines_center[3] = display_lines.prev_avg_lines_center[3] + 5
            elif closest_car[3] - display_lines.prev_avg_lines_center[3] > max_bottom + 5:
                display_lines.prev_avg_lines_center[3] = display_lines.prev_avg_lines_center[3] + 1
            else:
                display_lines.prev_avg_lines_center[3] = closest_car[3] - max_bottom
    
    # calculate the center line
    poly_x1_1 = int(display_lines.prev_avg_lines_center[0])+poly_bottom_diff
    poly_x1_2 = int(display_lines.prev_avg_lines_center[0])-poly_bottom_diff

    poly_x2_1 = int(display_lines.prev_avg_lines_center[2])+poly_top_diff
    poly_x2_2 = int(display_lines.prev_avg_lines_center[2])-poly_top_diff

    poly_points = np.array([[
        (poly_x1_1, int(display_lines.prev_avg_lines_center[1])),
        (poly_x1_2, int(display_lines.prev_avg_lines_center[1])),
        (poly_x2_2, int(display_lines.prev_avg_lines_center[3])),
        (poly_x2_1, int(display_lines.prev_avg_lines_center[3]))
    ]])

    # draw the center line (cyan)
    cv2.fillPoly(line_image, poly_points, color=(255, 255, 0))

    if closest_car_x_center is None and display_lines.detected_lines_counter > detected_lines_thres:
        # draw the updated average line (green marking)
        for x1, y1, x2, y2 in display_lines.prev_avg_lines:
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 10)

        # flag to overlay real time lane detections (red)
        if settings.modifiers.show_realtime_lines:
            for x1, y1, x2, y2 in lines:
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    return line_image

def getDist(point1, point2):
    '''
    Returns euclidian distance between 2 points
    '''
    return np.sqrt(np.sum(np.square(np.array(point1) - np.array(point2))))