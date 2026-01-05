import numpy as np
from PIL import Image
from scipy.signal import convolve2d

def load_image(image_path):
    """
    Load a color image and return it as a NumPy array.
    """
    img = Image.open(image_path).convert("RGB")
    return np.array(img)

def edge_detection(image):
    """
    Perform edge detection on a color image array.
    """
    # המרה לגרייסקייל
    if len(image.shape) == 3:
        gray = np.mean(image, axis=2)
    else:
        gray = image

    # תיקון הקרנלים לפי תקן Sobel
    # Sobel X - מזהה קווים אנכיים (שינוי אופקי)
    kernelX = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])

    # Sobel Y - מזהה קווים אופקיים (שינוי אנכי)
    kernelY = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ])

    # ביצוע הקונבולוציה
    edgeX = convolve2d(gray, kernelX, mode="same", boundary="fill", fillvalue=0)
    edgeY = convolve2d(gray, kernelY, mode="same", boundary="fill", fillvalue=0)

    # חישוב עוצמת הגרדיאנט
    edgeMAG = np.sqrt(edgeX**2 + edgeY**2)

    return edgeMAG


