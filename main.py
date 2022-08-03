from image_processing.insert_sudoku import solution_to_image
from sudoku.solve import SolveSudoku
import cv2
from image_processing.preprocess import threshold_image, dilate_image
from image_processing.extract_digits import extract_digits
from image_processing.predict_digits import load_model, predict_digits

img = cv2.imread('resources/ori_crop.png')
thresh = threshold_image(img)
preprocessed_img = dilate_image(thresh)
digits, coords = extract_digits(preprocessed_img)
model = load_model()
vals = predict_digits(digits, model)


solver = SolveSudoku(vals, coords)
ori_grid = solver.get_grid().copy()
solver.solve()
grid = solver.get_grid()
real_grid = solution_to_image(ori_grid, grid)
print(real_grid)