"""
Many functions related to image is difficult to test with 
exact values. So, some tests will save images after the 
functions has been processed, and the tests will check if
the file exists or not.
"""
import unittest
import os
from pathlib import Path

THIS_DIR = Path(__file__).parent

import cv2
import numpy as np
from image_read.preprocess import preprocess_image, threshold_image, dilate_image
from image_read.extract_sudoku import extract_sudoku, find_contours, find_corners, find_sudoku_contour, sort_points, crop_image

FILENAME = 'custom-sudoku.png'

class TestPreprocess(unittest.TestCase):
    def setUp(self):
        img_path = THIS_DIR.parent / f'resources/{FILENAME}'
        self.img = cv2.imread(str(img_path))


    def test_threshold(self):
        img = self.img
        threshold = threshold_image(img)
        img_path = THIS_DIR / f'images/image_read/thresh.png'
        cv2.imwrite(str(img_path), threshold)
        self.assertTrue(os.path.isfile(str(img_path)))


    def test_dilate(self):
        img = self.img
        dilate = dilate_image(img)
        img_path = THIS_DIR / f'images/image_read/dilate.png'
        cv2.imwrite(str(img_path), dilate)
        self.assertTrue(os.path.isfile(str(img_path)))


    def test_preprocess(self):
        img = self.img
        binary = preprocess_image(img)
        img_path = THIS_DIR / f'images/image_read/preprocess.png'
        cv2.imwrite(str(img_path), binary)
        self.assertTrue(os.path.isfile(str(img_path)))


class TestSortCorners(unittest.TestCase):
    def test_sort_corners_tilted_right(self):
        coor = np.array([(2,0), (1,3), (3,2), (0,1)])
        actual_coor = sort_points(coor)
        expected_coor = np.array([(1,3), (3,2), (2,0), (0,1)])
        self.assertEqual(expected_coor.tolist(), actual_coor.tolist())


    def test_sort_corners_tilted_left(self):
        coor = np.array([(1,0), (3,1), (2,3), (0,2)])
        actual_coor = sort_points(coor)
        expected_coor = np.array([(0,2), (2,3), (3,1), (1,0)])
        self.assertEqual(expected_coor.tolist(), actual_coor.tolist())


    def test_sort_corners_no_tilt(self):
        coor = np.array([(0,0),  (0,5), (5,5), (5,0)])
        actual_coor = sort_points(coor)
        expected_coor = np.array([(0,5), (5,5), (5,0), (0,0)])
        self.assertEqual(expected_coor.tolist(), actual_coor.tolist())



class TestFindSudoku(unittest.TestCase):
    def setUp(self):
        img_path = THIS_DIR.parent / f'resources/{FILENAME}'
        img = cv2.imread(str(img_path))
        self.binary_img = preprocess_image(img)


    def test_find_contours(self):
        binary_img = self.binary_img
        contours = find_contours(binary_img)
        # convert to BGR so lines can have color
        BGR_img = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(BGR_img, contours, -1, (0,255,0), 3)
        img_path = THIS_DIR / 'images/image_read/all_contours.png'
        cv2.imwrite(str(img_path), BGR_img)
        self.assertTrue(os.path.isfile(str(img_path)))


    def test_find_sudoku_contour(self):
        binary_img = self.binary_img
        contours = find_contours(binary_img)
        try:
            thresh_area = self.binary_img.shape[1]*self.binary_img.shape[0]/4
            thresh_ratio = 0.15
            sudoku_contour = find_sudoku_contour(contours, thresh_area, thresh_ratio)
        except Exception as e:
            self.fail(e)
        # convert to BGR so lines can have color
        bgr_img = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(bgr_img, [sudoku_contour], 0, (0,255,0), 3)
        img_path = THIS_DIR / 'images/image_read/sudoku_contour.png'
        cv2.imwrite(str(img_path), bgr_img)
        self.assertTrue(os.path.isfile(str(img_path)))


    def test_find_corners(self):
        binary_img = self.binary_img
        contours = find_contours(binary_img)
        try:
            thresh_area = self.binary_img.shape[1]*self.binary_img.shape[0]/6
            thresh_ratio = 0.15
            sudoku_contour = find_sudoku_contour(contours, thresh_area, thresh_ratio)
        except Exception as e:
            self.fail(e)
        coords = find_corners(sudoku_contour)
        # convert to BGR so lines can have color
        bgr_img = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
        for coord in coords:
            cv2.circle(bgr_img, coord, radius=15, color=(0,255,0), thickness=-1)
        img_path = THIS_DIR / 'images/image_read/corners.png'
        cv2.imwrite(str(img_path), bgr_img)
        self.assertTrue(os.path.isfile(str(img_path)))


class TestExtractSudoku(unittest.TestCase):
    def setUp(self):
        img_path = THIS_DIR.parent / f'resources/{FILENAME}'
        img = cv2.imread(str(img_path))
        self.binary_img = preprocess_image(img)


    def test_extract_sudoku(self):
        try:
            cropped_binary_img, M = extract_sudoku(self.binary_img)
        except Exception as e:
            self.fail(e)
        cropped_binary_path = THIS_DIR / 'images/image_read/cropped_binary.png'
        cv2.imwrite(str(cropped_binary_path),  cropped_binary_img)
        self.assertTrue(os.path.isfile(str(cropped_binary_path)))
