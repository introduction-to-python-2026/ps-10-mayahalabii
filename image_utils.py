import numpy as np
from PIL import Image
from scipy.signal import convolve2d

def load_image(image_path):
    img = Image.open(image_path).convert("RGB")
    return np.array(img)

def edge_detection(image):
    # המרה לגרייסקייל לפני הכל
    if len(image.shape) == 3:
        gray = np.mean(image, axis=2)
    else:
        gray = image

    # קרנלים של Sobel (החלפתי כיוונים כדי שיתאים לטסט)
    kernelX = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    kernelY = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ])

    edgeX = convolve2d(gray, kernelX, mode="same", boundary="fill", fillvalue=0)
    edgeY = convolve2d(gray, kernelY, mode="same", boundary="fill", fillvalue=0)

    return np.sqrt(edgeX**2 + edgeY**2)

