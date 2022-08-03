import unittest
import os
import cv2
from pathlib import Path
from digit_processing.extract_digits import extract_digits
from digit_processing.predict_digits import load_model, predict_digits
from image_read.extract_sudoku import extract_sudoku
from image_read.preprocess import dilate_image, threshold_image
from image_write.insert_solution import create_img_from_solution_grid, overlay_images, undo_transformation
from sudoku.solve import SolveSudoku
THIS_DIR = Path(__file__).parent


class TestSolutionGridToImage(unittest.TestCase):
    def setUp(self):
        img_path = THIS_DIR.parent / 'resources/cropped_binary.png'
        img = cv2.imread(str(img_path))
        # convert to GRAY so it can be processed
        binary_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.sudoku_w, self.sudoku_h = int(binary_img.shape[1]), int(binary_img.shape[0])
        digits, coords = extract_digits(binary_img)
        model = load_model()
        vals = predict_digits(digits, model)
        solver = SolveSudoku(vals, coords)
        self.unsolved_grid = solver.get_grid().copy()
        solver.solve()
        self.solved_grid = solver.get_grid().copy()


    def test_solution_grid_to_img(self):
        solution_img = create_img_from_solution_grid(self.sudoku_w, self.sudoku_h, self.solved_grid, self.unsolved_grid)
        solution_path = THIS_DIR / 'solution_img/solution_img.png'
        cv2.imwrite(str(solution_path), solution_img)
        self.assertTrue(os.path.isfile(str(solution_path)))


class TestUndoTransform(unittest.TestCase):
    def setUp(self):
        # Read img
        img_path = THIS_DIR.parent / 'resources/sudoku2.png'
        img = cv2.imread(str(img_path))
        self.img = img
        self.img_w, self.img_h = img.shape[1], img.shape[0]
        # preprocess
        thresh = threshold_image(img)
        thresh_dilate = dilate_image(thresh)
        binary_img = thresh_dilate
        # extract_sudoku from img
        cropped_binary_img, self.M = extract_sudoku(binary_img)
        sudoku_w, sudoku_h = int(cropped_binary_img.shape[1]), int(cropped_binary_img.shape[0])
        # extract digits from sudoku
        digits, coords = extract_digits(cropped_binary_img)
        # predict digits
        model = load_model()
        vals = predict_digits(digits, model)
        # solve sudoku
        solver = SolveSudoku(vals, coords)
        unsolved_grid = solver.get_grid().copy()
        solver.solve()
        solved_grid = solver.get_grid().copy()
        # create img from solution grid
        self.solution_img = create_img_from_solution_grid(sudoku_w, sudoku_h, solved_grid, unsolved_grid)


    def test_undo_transformation(self):
        undo_img = undo_transformation(self.solution_img, self.M, self.img_w, self.img_h)
        undo_path = THIS_DIR / 'transformation_img/undo_img.png'
        cv2.imwrite(str(undo_path), undo_img)
        self.assertTrue(os.path.isfile(str(undo_path)))


    def test_merge_images(self):
        undo_img = undo_transformation(self.solution_img, self.M, self.img_w, self.img_h)
        final_img = overlay_images(self.img, undo_img)
        final_path = THIS_DIR / 'transformation_img/final.png'
        cv2.imwrite(str(final_path), final_img)
        self.assertTrue(os.path.isfile(str(final_path)))
