import numpy as np
from PIL import Image
from scipy.signal import convolve2d

def load_image(image_path):
    """
    טעינת תמונה והמרה מידית לגרייסקייל.
    הטסט מצפה שהתמונה שתעבור למדיאן תהיה דו-מימדית.
    """
    img = Image.open(image_path).convert("L") # המרה ל-Grayscale (0-255)
    return np.array(img)

def edge_detection(image):
    """
    זיהוי קצוות בעזרת אופרטור סובל.
    """
    # וודוא שהתמונה היא float לצורך חישובים מדויקים בטסט
    gray = image.astype(float)

    # קרנלים של Sobel - הגדרה מדויקת עבור הציון בטסט
    # שימי לב לכיוונים - זה קריטי להשוואה מול תמונת ה-True
    kernelX = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    
    kernelY = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ])

    # ביצוע הקונבולוציה
    edgeX = convolve2d(gray, kernelX, mode="same", boundary="fill", fillvalue=0)
    edgeY = convolve2d(gray, kernelY, mode="same", boundary="fill", fillvalue=0)

    # חישוב עוצמת המגניטודה
    edgeMAG = np.sqrt(edgeX**2 + edgeY**2)

    return edgeMAG
