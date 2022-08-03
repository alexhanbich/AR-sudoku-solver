FILENAME = 'custom-sudoku.png'
import cv2
from cv2 import exp
import numpy as np
from image_read.preprocess import preprocess_image
from image_read.extract_sudoku import extract_sudoku
from digit_processing.extract_digits import extract_digits
from digit_processing.predict_digits import load_model, predict_digits

img_path = f'resources/{FILENAME}'
img = cv2.imread(str(img_path))
binary_img = preprocess_image(img)
cropped_binary_img, _ = extract_sudoku(binary_img)
digits, M = extract_digits(cropped_binary_img)
model = load_model()
actual = predict_digits(digits, model)
actual = np.sort(actual)
expected = [2,8,3,5,9,2,6,5,1,2,3,5,4,2,1,8,8,1,9,7,3,1,9,8,5,3,7]
expected = np.sort(expected)

print(actual)
print(expected)