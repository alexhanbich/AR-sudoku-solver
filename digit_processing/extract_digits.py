import cv2

def crop_digit(img, stat):
    (x, y, w, h, _) = stat
    return img[y:y+h, x:x+w]


def is_digit_component(stat, original_w, original_h):
    cell_w = original_w // 9
    cell_h = original_h // 9
    thresh_min_width = cell_w * 0.12
    thresh_max_width = cell_w * 0.6
    thresh_min_height = cell_h * 0.4
    thresh_max_height = cell_h * 0.9
    (_, _, w, h, _) = stat
    if w < thresh_min_width or w > thresh_max_width:
        return False
    if h < thresh_min_height or h > thresh_max_height:
        return False
    return True


def pad_digit(digit, original_w, original_h):
    top_pad = (original_h//9 - digit.shape[0]) // 2
    bottom_pad = (original_h//9 - digit.shape[0]) // 2
    left_pad = (original_w//9 - digit.shape[1]) // 2
    right_pad = (original_w//9 - digit.shape[1]) // 2
    return cv2.copyMakeBorder(digit, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, None, 0)


def preprocess_digit(img):
    pre = cv2.resize(img, (28, 28))
    pre = pre.reshape(28, 28, 1)
    return pre


def get_digit_coords(centroid, original_w, original_h):
    x_coord = int((centroid[0]/original_w*9-0.5).round())
    y_coord = int((centroid[1]/original_h*9-0.5).round())
    return x_coord, y_coord


def extract_digits(img):
    original_w, original_h = img.shape[1], img.shape[0]
    # get all connected components from image
    cnt, _, stats, centroids = cv2.connectedComponentsWithStats(img)
    digits = []
    coords = []
    for i in range(0, cnt):
        if not is_digit_component(stats[i], original_w, original_h):
            continue
        cropped_digit = crop_digit(img, stats[i])
        padded_digit = pad_digit(cropped_digit, original_w, original_h)
        # inv_padded_digit = cv2.bitwise_not(padded_digit)
        digits.append(preprocess_digit(padded_digit))
        x_coord, y_coord = get_digit_coords(centroids[i], img.shape[1], img.shape[0])
        coords.append((x_coord, y_coord))
    return digits, coords