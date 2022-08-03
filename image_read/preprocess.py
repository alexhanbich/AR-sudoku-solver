import cv2

def threshold_image(img):
    # grayscale, blurr image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gb = gray
    gb = cv2.GaussianBlur(gray, (9, 9), 0)
    # binarize & invert image
    binary = cv2.adaptiveThreshold(gb , 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 3, 2)
    return binary

def dilate_image(img):
    kernal = cv2.getStructuringElement(cv2.MORPH_RECT , (5,5))
    dl = cv2.dilate(img ,kernal, iterations=1)
    return dl


def preprocess_image(img):
    dilate = dilate_image(img)
    dilate_thresh = threshold_image(dilate)
    return dilate_image(dilate_thresh)