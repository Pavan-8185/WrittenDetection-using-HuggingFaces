from PIL import Image
import numpy as np
import cv2

def preprocess_image(img: Image.Image, resize_height=384):
    img = img.convert("L")
    w, h = img.size
    aspect_ratio = w / h
    new_w = int(aspect_ratio * resize_height)
    img = img.resize((new_w, resize_height))
    return img

def segment_lines(pil_image):
    gray = pil_image.convert("L")
    img = np.array(gray)
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (img.shape[1] // 2, 5))
    dilated = cv2.dilate(thresh, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    line_images = []
    for cnt in sorted(contours, key=lambda x: cv2.boundingRect(x)[1]):
        x, y, w, h = cv2.boundingRect(cnt)
        if h > 15:  
            line_crop = pil_image.crop((x, y, x + w, y + h))
            line_images.append(line_crop)

    return line_images
