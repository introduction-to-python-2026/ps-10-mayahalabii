from PIL import Image
import numpy as np
from scipy.signal import convolve2d

def load_image(path):
    # Converts the image to an array and returns it
    img = Image.open(path)
    return np.array(img)

def edge_detection(image):
    # 1. Convert to grayscale if needed
    if len(image.shape) == 3:
        gray_image = np.mean(image, axis=2)
    else:
        gray_image = image

    # 2. Building filters for vertical and horizontal changes
    filter_x = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]])
    filter_y = np.array([[-1, -2, -1],
                         [0, 0, 0],
                         [1, 2, 1]])

    # 3. Running convolution with boundary handling
    edgeX = convolve2d(gray_image, filter_x, mode='same', boundary='fill', fillvalue=0)
    edgeY = convolve2d(gray_image, filter_y, mode='same', boundary='fill', fillvalue=0)

    # 4. Calculating the strength of the edges
    edgeMAG = np.sqrt(edgeX**2 + edgeY**2)

    return edgeMAG

