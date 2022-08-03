import cv2
import numpy as np
from scipy.fft import dst

def find_contours(img):
    edges = cv2.Canny(img, 100, 200)
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    return contours

def find_corners(contour):
    epsilon = cv2.arcLength(contour, True)*0.04
    approx = cv2.approxPolyDP(contour, epsilon, True)
    coords = np.squeeze(approx)
    if len(coords) != 4:
        raise Exception("Not a quadrilateral!")
    return coords


def find_sudoku_contour(contours, thresh_area, thresh_ratio):
    for contour in contours:
        (_,_),(w,h), _ = cv2.minAreaRect(contour)
        # 3 heuristics to check contour is sudoku
        # 1. has to cover 1/3 of the screen
        # 2. has to be a quadrilateral
        # 3. has to be somewhat close to a square
        if w*h < thresh_area:
            continue
        try:
            coords = find_corners(contour)
        except:
            continue
        if abs((w-h)/h) > thresh_ratio:
            continue
        # if passes heuristic, contour is sudoku
        return contour
    raise Exception("Sudoku not in contours.")


# order: top_left, top_right, bottom_left, bottom_right
# also the order for dest_points in getPerspectiveTransform()
def sort_points(coords):
    sorted_by_y = sorted(coords, key=lambda x:x[1])
    sorted_top_by_x = sorted(sorted_by_y[2:], key=lambda x:x[0])
    rev_sorted_bottom_by_x = sorted(sorted_by_y[:2], key=lambda x:x[0], reverse=True)
    sorted_points = np.concatenate((sorted_top_by_x, rev_sorted_bottom_by_x))
    return np.float32(sorted_points)


def get_dimention_of_contour(contour):
    (_,_),(w,h), _ = cv2.minAreaRect(contour)
    return w,h


def get_transformation_matrix(contour, w, h):
    # src points 
    src_points = find_corners(contour)
    src_points = sort_points(src_points)
    dst_points = np.array([(0,h),(w,h),(w,0),(0,0)], dtype = "float32")
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    return M


def crop_image(binary_img, contour):
    w,h = get_dimention_of_contour(contour)
    M = get_transformation_matrix(contour, w, h)
    cropped_binary_img = cv2.warpPerspective(binary_img, M, (int(w), int(h)))
    return cropped_binary_img, M


def extract_sudoku(binary_img):
    contours = find_contours(binary_img)
    try:
        thresh_area = int(binary_img.shape[1]*binary_img.shape[0]/6)
        thresh_ratio = 0.15
        sudoku_contour = find_sudoku_contour(contours, thresh_area, thresh_ratio)
    except Exception as e:
        raise e
    cropped_binary_img, M = crop_image(binary_img, sudoku_contour)
    return cropped_binary_img, M