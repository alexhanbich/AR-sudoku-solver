import unittest
import os
import cv2
import numpy as np
from pathlib import Path
from digit_processing.extract_digits import extract_digits, is_digit_component
from digit_processing.predict_digits import load_model, predict_digits
THIS_DIR = Path(__file__).parent


class TestExtractDigits(unittest.TestCase):
    def setUp(self):
        img_path = THIS_DIR.parent / 'resources/cropped_binary.png'
        img = cv2.imread(str(img_path))
        # convert to GRAY so it can be processed
        self.binary_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        self.sudoku_w, self.sudoku_h = self.binary_img.shape[1], self.binary_img.shape[0]


    def test_draw_digit_connected_components(self):
        binary_img = self.binary_img
        # convert to RGB so lines can have color
        rgb_img = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
        cnt, _, stats, centroids = cv2.connectedComponentsWithStats(binary_img)
        for i in range(0, cnt):
            if not is_digit_component(stats[i], self.sudoku_w, self.sudoku_h):
                continue
            (digit_x, digit_y, digit_w, digit_h, _) = stats[i]
            # draw to RGB image so that lines have color
            cv2.rectangle(rgb_img, (digit_x, digit_y, digit_w, digit_h), (0, 255, 0), 3)
        img_path = THIS_DIR / f'contours_img/connected_components.png'
        cv2.imwrite(str(img_path), rgb_img)
        self.assertTrue(os.path.isfile(str(img_path)))


    def test_draw_extract_digits(self):
        binary_img = self.binary_img
        digits, _ = extract_digits(binary_img)
        for i in range(len(digits)):
            imgi_path = THIS_DIR / f'digits_img/digit{i+1}.png'
            cv2.imwrite(str(imgi_path), digits[i])
            self.assertTrue(os.path.isfile(str(imgi_path)))


class TestPredictDigits(unittest.TestCase):
    def setUp(self):
        img_path = THIS_DIR.parent / 'resources/cropped_binary.png'
        cropped_binary_img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        self.digits, _ = extract_digits(cropped_binary_img)
        self.model = load_model()

    def test_predict_digits(self):
        actual = predict_digits(self.digits, self.model)
        actual = np.sort(actual)
        expected = [6,9,3,4,6,8,1,8,9,2,7,9,5,4,8,7,2,9,7,6,2,8,1,7,9,6,7,4,1,6,4,9,9,8,3,4]
        expected = np.sort(expected)
        self.assertTrue(np.array_equal(expected, actual))