
import cv2
import numpy as np

from image_read.preprocess import threshold_image


def undo_transformation(img_crop, M, sudoku_w, sudoku_h):
    undo_img = cv2.warpPerspective(img_crop, M, (sudoku_w, sudoku_h), flags=cv2.WARP_INVERSE_MAP)
    return undo_img


def remove_question_digits(original_grid, solved_grid):
    for i in range(9):
        for j in range(9):
            if original_grid[i][j] != 0:
                solved_grid[i][j] = 0
    return solved_grid


# def get_digit_centroid(x, y, w, h):
#     x_centroid = int(((x+0.5)/9*w).round())
#     y_centroid = int(((y+0.5)/9*h).round())
#     return x_centroid, y_centroid


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
    

def create_img_from_solution_grid(sudoku_w, sudoku_h, s_grid, u_grid):
    blank = np.zeros((sudoku_w, sudoku_h,3), dtype=np.uint8)
    c_i = 0
    s_grid = s_grid.T
    u_grid = u_grid.T
    for i in range(9):
        for j in range(9):
            if u_grid[i][j] == 0:
                font = cv2.FONT_HERSHEY_SIMPLEX
                text = str(s_grid[i][j])
                font_size, font_w, font_h = get_font_size(text, blank.shape)
                x_coor = int(round((i+0.5)/9*blank.shape[1])) - font_w//2
                y_coor = int(round((j+0.5)/9*blank.shape[0])) +  font_h//2
                cv2.putText(blank, text, (int(x_coor), int(y_coor)), font, font_size, (0, 255, 0), 2, cv2.LINE_AA)
                c_i += 1
    return blank
    
def overlay_images(img, overlay):
    # threshold overlay
    
    # thresholded_overlay = threshold_image(overlay)
    inv_overlay = cv2.bitwise_not(overlay)
    binary_overlay = cv2.cvtColor(inv_overlay, cv2.COLOR_RGB2GRAY)
    _, binary_threshold = cv2.threshold(binary_overlay, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # create maked_img
    masked_img = cv2.bitwise_and(img, img, mask=binary_threshold)
    # add masked_img with overlay
    overlayed_img = cv2.add(masked_img, overlay)
    return overlayed_img