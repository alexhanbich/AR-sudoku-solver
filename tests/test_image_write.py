import unittest
import os
import cv2
from pathlib import Path
from image.preprocess import threshold_image, dilate_image
from image.insert_sudoku import undo_transformation
from image.find_sudoku import extract_sudoku
THIS_DIR = Path(__file__).parent


class TestInsertSudoku(unittest.TestCase):
    def setUp(self):
        original_path = THIS_DIR.parent / 'resources/sudoku2.png'
        original_img = cv2.imread(str(original_path))
        thresh_img = threshold_image(original_img)
        preprocessed_img = dilate_image(thresh_img)
        ori_crop, pre_crop, M = extract_sudoku(original_img, preprocessed_img)
        self.ori_crop = ori_crop
        self.M = M
        self.dimensions = (int(original_img.shape[1]), int(original_img.shape[0]))


    def test_undo_transformation(self):
        undo_img = undo_transformation(self.ori_crop, self.M, self.dimensions)
        undo_path = THIS_DIR / 'transformation_img/undo_img.png'
        cv2.imwrite(str(undo_path), undo_img)
        self.assertTrue(os.path.isfile(str(undo_path)))