import cv2
import numpy as np


def undo_transformation(img_crop, M, original_dimension):
    w,h = original_dimension
    undo_img = cv2.warpPerspective(img_crop, M, (w, h), flags=cv2.WARP_INVERSE_MAP)
    return undo_img

def insert_sudoku_to_image():
    pass