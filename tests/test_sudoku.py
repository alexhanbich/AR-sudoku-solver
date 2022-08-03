import unittest
import numpy as np
import os
import cv2
from pathlib import Path
from image_processing.predict_digits import load_model, predict_digits
from image_processing.preprocess import threshold_image, dilate_image
from image_processing.extract_digits import extract_digits
from sudoku.solve import SolveSudoku

THIS_DIR = Path(__file__).parent

class TestSudoku(unittest.TestCase):
    def setUp(self):
        img_path = THIS_DIR.parent / 'resources/ori_crop.png'
        img = cv2.imread(str(img_path))
        thresh = threshold_image(img)
        preprocessed_img = dilate_image(thresh)
        self.digits, self.coords = extract_digits(preprocessed_img)
        self.model = load_model()
        self.vals = predict_digits(self.digits, self.model)


    def test_create_board(self):
        solver = SolveSudoku(self.vals, self.coords)

        grid = solver.get_grid()
        actual = np.array([ [0,0,9,3,0,6,0,0,4],
                            [0,0,0,0,0,8,6,0,1],
                            [0,8,0,0,9,7,0,2,0],
                            [7,9,0,5,2,4,0,0,8],
                            [0,0,0,7,0,9,0,0,0],
                            [2,0,0,6,8,1,0,7,9],
                            [0,7,0,4,6,0,0,1,0],
                            [6,0,4,9,0,0,0,0,0],
                            [9,0,0,8,0,3,4,0,0]])
        self.assertTrue(np.array_equal(grid, actual))


    def test_solve_sudoku(self):
        solver = SolveSudoku(self.vals, self.coords)
        solver.solve()
        grid = solver.get_grid()
        actual = np.array([ [1,2,9,3,5,6,7,8,4],
                            [5,3,7,2,4,8,6,9,1],
                            [4,8,6,1,9,7,5,2,3],
                            [7,9,3,5,2,4,1,6,8],
                            [8,6,1,7,3,9,2,4,5],
                            [2,4,5,6,8,1,3,7,9],
                            [3,7,8,4,6,5,9,1,2],
                            [6,5,4,9,1,2,8,3,7],
                            [9,1,2,8,7,3,4,5,6]])
        self.assertTrue(np.array_equal(grid, actual))
