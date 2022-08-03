import unittest
import os
import cv2
import numpy as np
from pathlib import Path
from image_processing.extract_digits import extract_digits, is_digit_component
from image_processing.preprocess import threshold_image, dilate_image
from image_processing.insert_sudoku import insert_digits, undo_transformation
from image_processing.find_sudoku import extract_sudoku
THIS_DIR = Path(__file__).parent


class TestExtractDigits(unittest.TestCase):
    def setUp(self):
        img_path = THIS_DIR.parent / 'resources/ori_crop.png'
        self.img = cv2.imread(str(img_path))
        thresh = threshold_image(self.img)
        self.preprocessed_img = dilate_image(thresh)
        self.original_w, self.original_h = self.img.shape[1], self.img.shape[0]


    def test_digit_connected_components(self):
        original_img = self.img
        preprocessed_img = self.preprocessed_img
        cnt, _, stats, centroids = cv2.connectedComponentsWithStats(preprocessed_img)
        self.assertTrue(cnt > 10)
        for i in range(0, cnt):
            if not is_digit_component(stats[i], self.original_w, self.original_h):
                continue
            (x, y, w, h, _) = stats[i]
            cv2.rectangle(original_img, (x, y, w, h), (0, 255, 0), 3)
        img_path = THIS_DIR / f'contours_img/connected_components.png'
        cv2.imwrite(str(img_path), original_img)
        self.assertTrue(os.path.isfile(str(img_path)))


    def test_extract_digits(self):
        preprocessed_img = self.preprocessed_img
        digits, _ = extract_digits(preprocessed_img)
        self.assertTrue(len(digits) > 0)
        for i in range(len(digits)):
            imgi_path = THIS_DIR / f'digits_img/digit{i+1}.png'
            cv2.imwrite(str(imgi_path), digits[i])
            self.assertTrue(os.path.isfile(str(imgi_path)))


class TestInsertSudoku(unittest.TestCase):
    def setUp(self):
        original_path = THIS_DIR.parent / 'resources/sudoku2.png'
        original_img = cv2.imread(str(original_path))
        thresh_img = threshold_image(original_img)
        preprocessed_img = dilate_image(thresh_img)
        ori_crop, _, M = extract_sudoku(original_img, preprocessed_img)
        self.ori_crop = ori_crop
        self.M = M
        self.dimensions = (int(original_img.shape[1]), int(original_img.shape[0]))


    def test_insert_digits(self):
        w, h = self.ori_crop[1], self.ori_crop[0]
        img = np.zeros((w,h,3), dtype=np.uint8)
        solution_img = insert_digits(img, self.)

    def test_undo_transformation(self):
        undo_img = undo_transformation(self.ori_crop, self.M, self.dimensions)
        undo_path = THIS_DIR / 'transformation_img/undo_img.png'
        cv2.imwrite(str(undo_path), undo_img)
        self.assertTrue(os.path.isfile(str(undo_path)))