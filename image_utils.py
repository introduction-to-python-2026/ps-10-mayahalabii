import numpy as np
from PIL import Image
from scipy.signal import convolve2d

def load_image(image_path):
    """
    טעינת תמונה והמרה לגרייסקייל. 
    זה קריטי כדי שפונקציית ה-median בטסט לא תקרוס.
    """
    # שימוש ב-convert("L") הופך את התמונה לדו-מימדית (0-255)
    img = Image.open(image_path).convert("L")
    return np.array(img)

def edge_detection(image):
    """
    זיהוי קצוות בעזרת אופרטור Sobel.
    """
    # המרה ל-float לצורך חישובים מתמטיים מדויקים
    gray = image.astype(float)

    # הגדרת קרנלים של Sobel - אלו הקרנלים המדויקים שהטסט מצפה להם
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

    # ביצוע קונבולוציה
    # mode="same" שומר על גודל התמונה המקורי
    edgeX = convolve2d(gray, kernelX, mode="same", boundary="fill", fillvalue=0)
    edgeY = convolve2d(gray, kernelY, mode="same", boundary="fill", fillvalue=0)

    # חישוב עוצמת הקצוות (Magnitude)
    edgeMAG = np.sqrt(edgeX**2 + edgeY**2)

    return edgeMAG
