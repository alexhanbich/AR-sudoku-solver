import unittest
import cv2
from pathlib import Path
from image_processing.predict_digits import load_model, predict_digits
from image_processing.preprocess import threshold_image, dilate_image
from image_processing.extract_digits import extract_digits

THIS_DIR = Path(__file__).parent

class TestPredictDigits(unittest.TestCase):
    def setUp(self):
        img_path = THIS_DIR.parent / 'resources/ori_crop.png'
        img = cv2.imread(str(img_path))
        thresh = threshold_image(img)
        preprocessed_img = dilate_image(thresh)
        self.digits, _ = extract_digits(preprocessed_img)
        self.model = load_model()

    def test_predict_digits(self):
        vals = predict_digits(self.digits, self.model)
        actual = [6,9,3,4,6,8,1,8,9,2,7,9,5,4,8,7,2,9,7,6,2,8,1,7,9,6,7,4,1,6,4,9,9,8,3,4]
        print(vals)
        self.assertEqual(vals, actual)