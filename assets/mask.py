import cv2

def to_binary_mask(image):
    ret, bool_array = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    return bool_array