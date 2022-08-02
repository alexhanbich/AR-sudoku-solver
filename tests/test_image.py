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
from image.preprocess import threshold_image, dilate_image
from image.find_sudoku import find_contours, find_corners, find_sudoku_contour, get_dimention_of_contour, get_transformation_matrix, sort_points, crop_image

class TestPreprocess(unittest.TestCase):
    def setUp(self):
        # get absolute path
        img1_path = THIS_DIR.parent / 'resources/sudoku1.png'
        img2_path = THIS_DIR.parent / 'resources/sudoku2.png'
        img3_path = THIS_DIR.parent / 'resources/sudoku3.png'
        img4_path = THIS_DIR.parent / 'resources/sudoku4.png'
        img5_path = THIS_DIR.parent / 'resources/sudoku5.png'
        # load image into img_list
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
        original_path = THIS_DIR.parent / 'resources/sudoku2.png'
        original_img = cv2.imread(str(original_path))
        self.original_img = original_img
        thresh_dilate_path = THIS_DIR.parent / 'resources/thresh_dilate2.png'
        thresh_dilate_img = cv2.imread(str(thresh_dilate_path))
        self.img = thresh_dilate_img
        self.contours = find_contours(self.img)
        try:
            self.thresh_area = self.img.shape[1]*self.img.shape[0]/4
            self.thresh_ratio = 0.15
            self.sudoku_contour = find_sudoku_contour(self.contours, self.thresh_area, self.thresh_ratio)
        except Exception as e:
            self.fail(e)
        


    def test_find_contours(self):
        img = self.img
        contours = find_contours(img)
        cv2.drawContours(img, contours, -1, (0,255,0), 3)
        img_path = THIS_DIR / 'contours_img/contours.png'
        cv2.imwrite(str(img_path), img)
        self.assertTrue(os.path.isfile(str(img_path)))

    def test_find_sudoku_contour(self):
        img = self.img
        contours = self.contours
        
        try:
            sudoku_contour = find_sudoku_contour(contours, self.thresh_area, self.thresh_ratio)
        except Exception as e:
            self.fail(e)
        cv2.drawContours(img, [sudoku_contour], 0, (0,255,0), 3)
        img_path = THIS_DIR / 'contours_img/sudoku_contour.png'
        cv2.imwrite(str(img_path), img)
        self.assertTrue(os.path.isfile(str(img_path)))


    def test_find_corners(self):
        img = self.img
        sudoku_contour = self.sudoku_contour
        coords = find_corners(sudoku_contour)
        for coord in coords:
            cv2.circle(img, coord, radius=15, color=(0,255,0), thickness=-1)
        img_path = THIS_DIR / 'contours_img/corners.png'
        cv2.imwrite(str(img_path), img)
        self.assertTrue(os.path.isfile(str(img_path)))


    def test_get_dimention_of_contour(self):
        w,h = get_dimention_of_contour(self.sudoku_contour)
        self.assertIsNotNone(w)
        self.assertIsNotNone(h)

    def test_get_transformation_matrix(self):
        w,h = get_dimention_of_contour(self.sudoku_contour)
        M = get_transformation_matrix(self.sudoku_contour, w, h)
        self.assertIsNotNone(M)


    def test_crop_image(self):
        original_crop, preprocess_crop, M = crop_image(self.original_img, self.img, self.sudoku_contour)
        ori_crop_path = THIS_DIR / 'contours_img/ori_crop.png'
        pre_crop_path = THIS_DIR / 'contours_img/pre_crop.png'
        cv2.imwrite(str(ori_crop_path), original_crop)
        cv2.imwrite(str(pre_crop_path), preprocess_crop)
        self.assertTrue(os.path.isfile(str(ori_crop_path)))
        self.assertTrue(os.path.isfile(str(pre_crop_path)))