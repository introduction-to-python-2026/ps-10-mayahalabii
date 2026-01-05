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
    # Convert RGB image to grayscale by averaging channels
    gray = np.mean(image, axis=2)

    # Define kernels
    kernelY = np.array([
        [1,  0, -1],
        [2,  0, -2],
        [1,  0, -1]
    ])

    kernelX = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ])

    # Apply convolution with zero padding
    edgeY = convolve2d(gray, kernelY, mode="same", boundary="fill", fillvalue=0)
    edgeX = convolve2d(gray, kernelX, mode="same", boundary="fill", fillvalue=0)

    # Edge magnitude
    edgeMAG = np.sqrt(edgeX**2 + edgeY**2)

    return edgeMAG


