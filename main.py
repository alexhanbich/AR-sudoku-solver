import cv2
from digit_processing.extract_digits import extract_digits
from digit_processing.predict_digits import load_model, predict_digits
from image_read.extract_sudoku import extract_sudoku

from image_read.preprocess import dilate_image, threshold_image
from image_write.insert_solution import create_img_from_solution_grid, overlay_images, undo_transformation
from sudoku.solve import SolveSudoku


# cv2.imshow(' ', img)
# Read img
img = cv2.imread(str('resources/custom-sudoku2.png'))


# resize to 720p
dim = (1280, 720)
img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
cv2.imshow(' ', img)
cv2.waitKey(0)
img_w, img_h = img.shape[1], img.shape[0]

# preprocess
thresh = threshold_image(img)
# thresh_dilate = dilate_image(thresh)
binary_img = thresh
cv2.imshow(' ', binary_img)
cv2.waitKey(0)

# extract_sudoku from img
cropped_binary_img, M = extract_sudoku(binary_img)
cv2.imshow(' ', cropped_binary_img)
cv2.imwrite('custom-binary.png', cropped_binary_img)
cv2.waitKey(0)
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
solution_img = create_img_from_solution_grid(sudoku_w, sudoku_h, solved_grid, unsolved_grid)

# unwarp sudoku to img
undo_img = undo_transformation(solution_img, M, img_w, img_h)

# overlay images
final = overlay_images(img, undo_img)
cv2.imwrite('test-final.png', final)