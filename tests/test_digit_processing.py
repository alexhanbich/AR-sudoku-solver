import unittest
import os
import cv2
import numpy as np
from pathlib import Path
from digit_processing.extract_digits import extract_digits, is_digit_component
from digit_processing.predict_digits import load_model, predict_digits
from image_read.extract_sudoku import extract_sudoku
from image_read.preprocess import preprocess_image
THIS_DIR = Path(__file__).parent
FILENAME = 'custom-sudoku.png'


class TestExtractDigits(unittest.TestCase):
    def setUp(self):
        img_path = THIS_DIR.parent / f'resources/{FILENAME}'
        img = cv2.imread(str(img_path))
        binary_img = preprocess_image(img)
        cropped_binary_img, _ = extract_sudoku(binary_img)
        self.cropped_binary_img = cropped_binary_img


    def test_draw_digit_connected_components(self):
        binary_img = self.cropped_binary_img
        sudoku_w, sudoku_h = int(binary_img.shape[1]), int(binary_img.shape[0])
        # convert to BGR so lines can have color
        rgb_img = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
        cnt, _, stats, _ = cv2.connectedComponentsWithStats(binary_img)
        for i in range(0, cnt):
            if not is_digit_component(stats[i], sudoku_w, sudoku_h):
                continue
            (digit_x, digit_y, digit_w, digit_h, _) = stats[i]
            # draw to BGR image so that lines have color
            cv2.rectangle(rgb_img, (digit_x, digit_y, digit_w, digit_h), (0, 255, 0), 3)
        img_path = THIS_DIR / f'images/digit_processing/digit_component.png'
        cv2.imwrite(str(img_path), rgb_img)
        self.assertTrue(os.path.isfile(str(img_path)))


    def test_draw_extract_digits(self):
        binary_img = self.cropped_binary_img
        digits, _ = extract_digits(binary_img)
        for i in range(len(digits)):
            imgi_path = THIS_DIR / f'images/digit_processing/digits/digit{i+1}.png'
            cv2.imwrite(str(imgi_path), digits[i])
            self.assertTrue(os.path.isfile(str(imgi_path)))


# ONLY WORKS IF YOU TYPE IN THE EXPECTED VALUES
class TestPredictDigits(unittest.TestCase):
    def setUp(self):
        img_path = THIS_DIR.parent / f'resources/{FILENAME}'
        img = cv2.imread(str(img_path))
        binary_img = preprocess_image(img)
        cropped_binary_img, _ = extract_sudoku(binary_img)
        self.digits, _ = extract_digits(cropped_binary_img)
        self.model = load_model()

    def test_predict_digits(self):
        actual = predict_digits(self.digits, self.model)
        actual = np.sort(actual)
        expected = [2,8,3,5,9,2,6,5,1,2,3,5,4,2,1,8,8,1,9,7,3,1,9,8,5,3,7]
        expected = np.sort(expected)
        self.assertTrue(np.array_equal(expected, actual))