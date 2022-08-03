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
from image_read.preprocess import threshold_image, dilate_image
from image_read.extract_sudoku import extract_sudoku, find_contours, find_corners, find_sudoku_contour, sort_points, crop_image

class TestPreprocess(unittest.TestCase):
    def setUp(self):
        img1_path = THIS_DIR.parent / 'resources/sudoku1.png'
        img2_path = THIS_DIR.parent / 'resources/sudoku2.png'
        img3_path = THIS_DIR.parent / 'resources/sudoku3.png'
        img4_path = THIS_DIR.parent / 'resources/sudoku4.png'
        img5_path = THIS_DIR.parent / 'resources/sudoku5.png'
        self.img_list = []
        self.img_list.append(cv2.imread(str(img1_path)))
        self.img_list.append(cv2.imread(str(img2_path)))
        self.img_list.append(cv2.imread(str(img3_path)))
        self.img_list.append(cv2.imread(str(img4_path)))
        self.img_list.append(cv2.imread(str(img5_path)))


    def test_threshold(self):
        img_list = self.img_list
        for i in range(len(img_list)):
            threshold = threshold_image(img_list[i])
            imgi_path = THIS_DIR / f'preprocess_img/threshold_img/thresh{i+1}.png'
            cv2.imwrite(str(imgi_path), threshold)
            self.assertTrue(os.path.isfile(str(imgi_path)))


    def test_dilate(self):
        img_list = self.img_list
        for i in range(len(img_list)):
            dilate = dilate_image(img_list[i])
            imgi_path = THIS_DIR / f'preprocess_img/dilate_img/dilate{i+1}.png'
            cv2.imwrite(str(imgi_path), dilate)
            self.assertTrue(os.path.isfile(str(imgi_path)))


    def test_thresh_dilate(self):
        img_list = self.img_list
        for i in range(len(img_list)):
            threshold = threshold_image(img_list[i])
            thresh_dilate = dilate_image(threshold)
            imgi_path = THIS_DIR / f'preprocess_img/thresh_dilate_img/thresh_dilate{i+1}.png'
            cv2.imwrite(str(imgi_path), thresh_dilate)
            self.assertTrue(os.path.isfile(str(imgi_path)))


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
        img_path = THIS_DIR.parent / 'resources/sudoku2.png'
        img = cv2.imread(str(img_path))
        self.color_img = img
        thresh = threshold_image(img)
        thresh_dilate = dilate_image(thresh)
        self.binary_img = thresh_dilate


    def test_find_contours(self):
        binary_img = self.binary_img
        contours = find_contours(binary_img)
        # convert to BGR so lines can have color
        BGR_img = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(BGR_img, contours, -1, (0,255,0), 3)
        img_path = THIS_DIR / 'contours_img/contours.png'
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
        img_path = THIS_DIR / 'contours_img/sudoku_contour.png'
        cv2.imwrite(str(img_path), bgr_img)
        self.assertTrue(os.path.isfile(str(img_path)))


    def test_find_corners(self):
        binary_img = self.binary_img
        contours = find_contours(binary_img)
        try:
            thresh_area = self.binary_img.shape[1]*self.binary_img.shape[0]/4
            thresh_ratio = 0.15
            sudoku_contour = find_sudoku_contour(contours, thresh_area, thresh_ratio)
        except Exception as e:
            self.fail(e)
        coords = find_corners(sudoku_contour)
        # convert to BGR so lines can have color
        bgr_img = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
        for coord in coords:
            cv2.circle(bgr_img, coord, radius=15, color=(0,255,0), thickness=-1)
        img_path = THIS_DIR / 'contours_img/corners.png'
        cv2.imwrite(str(img_path), bgr_img)
        self.assertTrue(os.path.isfile(str(img_path)))


class TestExtractSudoku(unittest.TestCase):
    def setUp(self):
        img_path = THIS_DIR.parent / 'resources/sudoku2.png'
        img = cv2.imread(str(img_path))
        self.color_img = img
        thresh = threshold_image(img)
        thresh_dilate = dilate_image(thresh)
        self.binary_img = thresh_dilate


    def test_crop_image(self):
        binary_img = self.binary_img
        contours = find_contours(binary_img)
        try:
            thresh_area = self.binary_img.shape[1]*self.binary_img.shape[0]/4
            thresh_ratio = 0.15
            sudoku_contour = find_sudoku_contour(contours, thresh_area, thresh_ratio)
        except Exception as e:
            self.fail(e)
        cropped_binary_img, M = crop_image(self.binary_img, sudoku_contour)
        cropped_binary_path = THIS_DIR / 'contours_img/cropped_binary1.png'
        cv2.imwrite(str(cropped_binary_path),  cropped_binary_img)
        self.assertTrue(os.path.isfile(str(cropped_binary_path)))


    def test_extract_sudoku(self):
        cropped_binary_img, M = extract_sudoku(self.binary_img)
        cropped_binary_path = THIS_DIR / 'contours_img/cropped_binary2.png'
        cv2.imwrite(str(cropped_binary_path),  cropped_binary_img)
        self.assertTrue(os.path.isfile(str(cropped_binary_path)))
