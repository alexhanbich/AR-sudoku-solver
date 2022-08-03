from os import remove
import cv2
import numpy as np


def undo_transformation(img_crop, M, original_dimension):
    w,h = original_dimension
    undo_img = cv2.warpPerspective(img_crop, M, (w, h), flags=cv2.WARP_INVERSE_MAP)
    return undo_img


def remove_question_digits(original_grid, solved_grid):
    for i in range(9):
        for j in range(9):
            if original_grid[i][j] != 0:
                solved_grid[i][j] = 0
    return solved_grid


def get_digit_centroid(x, y, w, h):
    x_centroid = int(((x+0.5)/9*w).round())
    y_centroid = int(((h+0.5)/9*w).round())
    return x_centroid, y_centroid


def get_font_size(text, shape):
    w = int(shape[1]/9*0.4)
    for scale in reversed(range(0, 60, 1)):
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, scale/10, 2)
        font_w = text_size[0][0]
        font_h = text_size[0][1]
        if font_w <= w:
            return scale/10, font_w, font_h 
    return 1
    

def insert_digits(img, s_grid, u_grid):
    c_i = 0
    s_grid = s_grid.T
    u_grid = u_grid.T
    for i in range(9):
        for j in range(9):
            if u_grid[i][j] == 0:
                font = cv2.FONT_HERSHEY_SIMPLEX
                text = str(s_grid[i][j])
                font_size, font_w, font_h = get_font_size(text, img.shape)
                x_coor = int(round((i+0.5)/9*img.shape[1])) - font_w//2
                y_coor = int(round((j+0.5)/9*img.shape[0])) +  font_h//2
                cv2.putText(img, text, (int(x_coor), int(y_coor)), font, font_size, (255, 0, 0), 2, cv2.LINE_AA)
                c_i += 1
    return img
    
def merge_images(img1, img2):
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    thresh2 = cv2.threshold(gray2, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    result1 = cv2.bitwise_and(img1, img1, mask=thresh2)
    result2 = cv2.bitwise_and(img2, img2, mask=255-thresh2)
    result = cv2.add(result1, result2)
    return result