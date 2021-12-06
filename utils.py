from skimage import draw
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import cv2
import math
import settings

def poly2mask(vertex_row_coords, vertex_col_coords, shape):
    fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, shape)
    mask = np.zeros(shape, dtype=np.bool)
    mask[fill_row_coords, fill_col_coords] = True
    return mask


def specify_bottom_center(img):
    print("If it doesn't get you to the drawing mode, then rerun this function again. Also, make sure the object fill fit into the background image. Otherwise it will crash")
    fig = plt.figure()
    plt.imshow(img, cmap='gray')
    fig.set_label('Choose target bottom-center location')
    plt.axis('off')
    target_loc = np.zeros(2, dtype=int)

    def on_mouse_pressed(event):
        target_loc[0] = int(event.xdata)
        target_loc[1] = int(event.ydata)
        
    fig.canvas.mpl_connect('button_press_event', on_mouse_pressed)
    return target_loc

def align_source(object_img, mask, background_img, bottom_center):
    ys, xs = np.where(mask == 1)
    (h,w,_) = object_img.shape
    y1 = x1 = 0
    y2, x2 = h, w
    object_img2 = np.zeros(background_img.shape)
    yind = np.arange(y1,y2)
    yind2 = yind - int(max(ys)) + bottom_center[1]
    xind = np.arange(x1,x2)
    xind2 = xind - int(round(np.mean(xs))) + bottom_center[0]

    ys = ys - int(max(ys)) + bottom_center[1]
    xs = xs - int(round(np.mean(xs))) + bottom_center[0]
    mask2 = np.zeros(background_img.shape[:2], dtype=bool)
    for i in range(len(xs)):
        mask2[int(ys[i]), int(xs[i])] = True
    for i in range(len(yind)):
        for j in range(len(xind)):
            object_img2[yind2[i], xind2[j], :] = object_img[yind[i], xind[j], :]
    mask3 = np.zeros([mask2.shape[0], mask2.shape[1], 3])
    for i in range(3):
        mask3[:,:, i] = mask2
    background_img  = object_img2 * mask3 + (1-mask3) * background_img
    plt.figure()
    plt.imshow(background_img)
    return object_img2, mask2

def upper_left_background_rc(object_mask, bottom_center):
    """ 
      Returns upper-left (row,col) coordinate in background image that corresponds to (0,0) in the object image
      object_mask: foreground mask in object image
      bottom_center: bottom-center (x=col, y=row) position of foreground object in background image
    """
    ys, xs = np.where(object_mask == 1)
    (h,w) = object_mask.shape[:2]
    upper_left_row = bottom_center[1]-int(max(ys)) 
    upper_left_col = bottom_center[0] - int(round(np.mean(xs)))
    return [upper_left_row, upper_left_col]

def crop_object_img(object_img, object_mask):
    ys, xs = np.where(object_mask == 1)
    (h,w) = object_mask.shape[:2]
    x1 = min(xs)-1
    x2 = max(xs)+1
    y1 = min(ys)-1
    y2 = max(ys)+1
    object_mask = object_mask[y1:y2, x1:x2]
    object_img = object_img[y1:y2, x1:x2, :]
    return object_img, object_mask

def get_combined_img(bg_img, object_img, object_mask, bg_ul):
    combined_img = bg_img.copy()
    (nr, nc) = object_img.shape[:2]

    for b in np.arange(object_img.shape[2]):
      combined_patch = combined_img[bg_ul[0]:bg_ul[0]+nr, bg_ul[1]:bg_ul[1]+nc, b]
      combined_patch = combined_patch*(1-object_mask) + object_img[:,:,b]*object_mask
      combined_img[bg_ul[0]:bg_ul[0]+nr, bg_ul[1]:bg_ul[1]+nc, b] =  combined_patch

    return combined_img 


def specify_mask(img):
    # get mask
    print("If it doesn't get you to the drawing mode, then rerun this function again.")
    fig = plt.figure()
    fig.set_label('Draw polygon around source object')
    plt.axis('off')
    plt.imshow(img, cmap='gray')
    xs = []
    ys = []
    clicked = []

    def on_mouse_pressed(event):
        x = event.xdata
        y = event.ydata
        xs.append(x)
        ys.append(y)
        plt.plot(x, y, 'r+')

    def onclose(event):
        clicked.append(xs)
        clicked.append(ys)
    # Create an hard reference to the callback not to be cleared by the garbage
    # collector
    fig.canvas.mpl_connect('button_press_event', on_mouse_pressed)
    fig.canvas.mpl_connect('close_event', onclose)
    return clicked

def get_mask(ys, xs, img):
    mask = poly2mask(ys, xs, img.shape[:2]).astype(int)
    fig = plt.figure()
    plt.imshow(mask, cmap='gray')
    return mask


def region_of_interest(image):

#     # create a zero array
#     stencil = np.zeros_like(image[:,:])

#     # specify coordinates of the polygon
#     # polygon = np.array([[50,270], [220,160], [360,160], [480,270]])
#     %matplotlib notebook
#     polygon = specify_mask(image)
    
#     print(polygon)

#     # fill polygon with ones
#     cv2.fillConvexPoly(stencil, polygon, 1)

#     return stencil

    height, width = image.shape
#     polygons = np.array([
#         [(200, height), (1100, height), (550, 250)]
#         ])

    top_offset = 30
    trap_width = 100
    polygons = np.array([
        [(0, height), (width, height), (int(image.shape[1]/2)+trap_width, int(image.shape[0]/2)+top_offset), (int(image.shape[1]/2)-trap_width, int(image.shape[0]/2)+top_offset)]
        ])
    mask = np.zeros_like(image)
      
    # Fill poly-function deals with multiple polygon
    cv2.fillPoly(mask, polygons, 255) 
    
    # plt.imshow(mask)
    # plt.show()
      
    # Bitwise operation between canny image and mask image
    masked_image = cv2.bitwise_and(image, mask) 
    return masked_image

def canny_edge_detector(image):
    
    # print(image.shape)
      
    # Convert the image color to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) 
    
#     print(gray_image.shape)
      
    # Reduce noise from the image
    blur = cv2.GaussianBlur(gray_image, (5, 5), 0) 
    canny = cv2.Canny(blur, 50, 150)
    
    im_final = np.zeros_like(image)
    im_final[:,:,0] = canny
    im_final[:,:,1] = canny
    im_final[:,:,2] = canny
    
#     print(im_final.shape)
    return canny

def create_coordinates(image, line_parameters, type):
    # print(line_parameters)
    # slope, intercept = line_parameters

    try:
        slope, intercept = line_parameters
        if type == "left":
            create_coordinates.prev_line_parameters_left = line_parameters
        elif type == "right":
            create_coordinates.prev_line_parameters_right = line_parameters
    except TypeError:
        if type == "left":
            slope, intercept = create_coordinates.prev_line_parameters_left
        elif type == "right":
            slope, intercept = create_coordinates.prev_line_parameters_right
    
    y1 = image.shape[0]
    y2 = int(y1 * (3 / 5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    processed_lines = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
          
        # It will fit the polynomial and the intercept and slope
        parameters = np.polyfit((x1, x2), (y1, y2), 1) 
        slope = parameters[0]
        intercept = parameters[1]
        
        # ignore if slope too steep
        if abs(slope) < 0.40:
            continue

        processed_lines.append(line)
        
        im_center = int(image.shape[1]/2)
        im_center_height = int(image.shape[0]/3)

        x_center_weight_modifier = 0.5
        y_center_weight_modifier = 3

        if slope < 0:
            weight = int(x1*x_center_weight_modifier+abs(y1-im_center_height)*y_center_weight_modifier)
            for i in range(weight):
                left_fit.append((slope, intercept))
        else:
            weight = int((image.shape[1]-x1)*x_center_weight_modifier+abs(y1-im_center_height)*y_center_weight_modifier)
            for i in range(weight):
                right_fit.append((slope, intercept))
              
    left_fit_average = np.average(left_fit, axis = 0)
    right_fit_average = np.average(right_fit, axis = 0)
    # print(left_fit, right_fit)
    # print(left_fit_average, right_fit_average)
    left_line = create_coordinates(image, left_fit_average, "left")
    right_line = create_coordinates(image, right_fit_average, "right")
    return np.array([left_line, right_line]), processed_lines

def display_lines(image, lines):
    display_lines.counter += 1

    line_image = np.zeros_like(image)

    # if display_lines.counter % 5 == 0:

    #     for i in range(len(display_lines.line_average_sides)-1):
    #         display_lines.line_average_sides[i+1] = display_lines.line_average_sides[i]

    #     display_lines.line_average_sides[0] = lines

    if np.array(display_lines.prev_avg_lines).any() == None:
        display_lines.prev_avg_lines = lines

    # side_offset_x1_left = int((lines[0][0]-display_lines.prev_avg_lines[0][0])/1)
    # side_offset_x2_left = int((lines[0][0]-display_lines.prev_avg_lines[0][0])/1)
    # side_offset_x1_right = int((lines[0][0]-display_lines.prev_avg_lines[0][0])/1)
    # side_offset_x2_right = int((lines[0][0]-display_lines.prev_avg_lines[0][0])/1)

    # calculate center offset for top and bottom parts of line
    left_offset_x1 = int((lines[0][0]-display_lines.prev_avg_lines[0][0])/10)
    left_offset_x2 = int((lines[0][2]-display_lines.prev_avg_lines[0][2])/10)
    right_offset_x1 = int((lines[1][0]-display_lines.prev_avg_lines[1][0])/10)
    right_offset_x2 = int((lines[1][2]-display_lines.prev_avg_lines[1][2])/10)
    
    # add offset to prev avg line
    display_lines.prev_avg_lines = [
        [display_lines.prev_avg_lines[0][0]+left_offset_x1, display_lines.prev_avg_lines[0][1], display_lines.prev_avg_lines[0][2]+left_offset_x2, display_lines.prev_avg_lines[0][3]],
        [display_lines.prev_avg_lines[1][0]+right_offset_x1, display_lines.prev_avg_lines[1][1], display_lines.prev_avg_lines[1][2]+right_offset_x2, display_lines.prev_avg_lines[1][3]] 
    ]

    for x1, y1, x2, y2 in display_lines.prev_avg_lines:
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 10)

    avg_x1 = int((display_lines.prev_avg_lines[0][0]+display_lines.prev_avg_lines[1][0])/2)
    avg_x2 = int((display_lines.prev_avg_lines[0][2]+display_lines.prev_avg_lines[1][2])/2)
    avg_y1 = int(display_lines.prev_avg_lines[0][1])
    avg_y2 = int(display_lines.prev_avg_lines[0][3])

    if np.array(display_lines.line_average).any() == None:
        # display_lines.line_average = np.array(display_lines.line_average)
        for i in range(len(display_lines.line_average)):
            display_lines.line_average[i] = np.array([avg_x1, avg_y1, avg_x2, avg_y2])

    if display_lines.counter % 5 == 0:

        for i in range(len(display_lines.line_average)-1):
            display_lines.line_average[i+1] = display_lines.line_average[i]

        display_lines.line_average[0] = np.array([avg_x1, avg_y1, avg_x2, avg_y2])
    
    average_lines = np.average(np.array(display_lines.line_average), axis=0)

    # init prev avg center line
    if np.array(display_lines.prev_avg_lines_center).any() == None:
        display_lines.prev_avg_lines_center = average_lines

    # init prev avg line
    if np.array(display_lines.prev_avg_lines_center).any() == None:
        display_lines.prev_avg_lines = lines

    # calculate center offset for top and bottom parts of line
    center_offset_x1 = int((average_lines[0]-display_lines.prev_avg_lines_center[0])/10)
    center_offset_x2 = int((average_lines[2]-display_lines.prev_avg_lines_center[2])/10)

    # add offset to prev avg line
    display_lines.prev_avg_lines_center = [display_lines.prev_avg_lines_center[0]+center_offset_x1, display_lines.prev_avg_lines_center[1], display_lines.prev_avg_lines_center[2]+center_offset_x2, display_lines.prev_avg_lines_center[3]] 

    # cv2.line(line_image, (int(average_lines[0]+center_offset), int(average_lines[1])), (int(average_lines[2]+center_offset), int(average_lines[3])), (255, 255, 0), 10)
    poly_x1_1 = int(display_lines.prev_avg_lines_center[0])+60
    poly_x1_2 = int(display_lines.prev_avg_lines_center[0])-60

    poly_x2_1 = int(display_lines.prev_avg_lines_center[2])+10
    poly_x2_2 = int(display_lines.prev_avg_lines_center[2])-10

    poly_points = np.array([[
        (poly_x1_1, int(display_lines.prev_avg_lines_center[1])),
        (poly_x1_2, int(display_lines.prev_avg_lines_center[1])),
        (poly_x2_2, int(display_lines.prev_avg_lines_center[3])),
        (poly_x2_1, int(display_lines.prev_avg_lines_center[3]))
    ]])

    cv2.fillPoly(line_image, poly_points, color=(255, 255, 0))
    # cv2.line(line_image, (avg_x1, avg_y1), (avg_x2, avg_y2), (255, 0, 0), 10)

    # print(average_lines)

    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return line_image

def display_lines_2(image, lines, lines_bottom):
    display_lines.counter += 1

    line_image = np.zeros_like(image)
    
    avg_x1 = int((lines[0][0]+lines[1][0])/2)
    avg_x2 = int((lines[0][2]+lines[1][2])/2)
    avg_y1 = int(lines[0][1])
    avg_y2 = int(lines[0][3])

    if np.array(display_lines.line_average).any() == None:
        # display_lines.line_average = np.array(display_lines.line_average)
        for i in range(len(display_lines.line_average)):
            display_lines.line_average[i] = np.array([avg_x1, avg_y1, avg_x2, avg_y2])

    if display_lines.counter % 5 == 0:

        for i in range(len(display_lines.line_average)-1):
            display_lines.line_average[i+1] = display_lines.line_average[i]

        display_lines.line_average[0] = np.array([avg_x1, avg_y1, avg_x2, avg_y2])
    
    average_lines = np.average(np.array(display_lines.line_average), axis=0)

    # init prev avg center line
    if np.array(display_lines.prev_avg_lines_center).any() == None:
        display_lines.prev_avg_lines_center = average_lines

    # init prev avg line
    if np.array(display_lines.prev_avg_lines_center).any() == None:
        display_lines.prev_avg_lines = lines

    # calculate center offset for top and bottom parts of line
    center_offset_x1 = int((average_lines[0]-display_lines.prev_avg_lines_center[0])/10)
    center_offset_x2 = int((average_lines[2]-display_lines.prev_avg_lines_center[2])/10)

    # add offset to prev avg line
    display_lines.prev_avg_lines_center = [display_lines.prev_avg_lines_center[0]+center_offset_x1, display_lines.prev_avg_lines_center[1], display_lines.prev_avg_lines_center[2]+center_offset_x2, display_lines.prev_avg_lines_center[3]] 

    # cv2.line(line_image, (int(average_lines[0]+center_offset), int(average_lines[1])), (int(average_lines[2]+center_offset), int(average_lines[3])), (255, 255, 0), 10)
    poly_x1_1 = int(display_lines.prev_avg_lines_center[0])+60
    poly_x1_2 = int(display_lines.prev_avg_lines_center[0])-60

    poly_x2_1 = int(display_lines.prev_avg_lines_center[2])+10
    poly_x2_2 = int(display_lines.prev_avg_lines_center[2])-10

    poly_points = np.array([[
        (poly_x1_1, int(display_lines.prev_avg_lines_center[1])),
        (poly_x1_2, int(display_lines.prev_avg_lines_center[1])),
        (poly_x2_2, int(display_lines.prev_avg_lines_center[3])),
        (poly_x2_1, int(display_lines.prev_avg_lines_center[3]))
    ]])

    cv2.fillPoly(line_image, poly_points, color=(255, 255, 0))
    # cv2.line(line_image, (avg_x1, avg_y1), (avg_x2, avg_y2), (255, 0, 0), 10)

    # print(average_lines)

    if lines is not None:
        for x1, y1, x2, y2 in lines:
            print(x1,y1,x2,y2)
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 10)
        for x1, y1, x2, y2 in lines_bottom:
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 10)
    return line_image

def getDist(point1, point2):
    return np.sqrt(np.sum(np.square(np.array(point1) - np.array(point2))))