import numpy as np
from PIL import Image
from scipy.signal import convolve2d
from skimage.filters import median
from skimage.morphology import disk

# --- פונקציות לבדיקת האוטוגריידר ---

def load_image(path):
    """טוען תמונה ומחזיר כ-array numpy"""
    img = Image.open(path)
    return np.array(img)

def edge_detection(image):
    """מבצע זיהוי קצוות על תמונה (grayscale או צבעונית)"""
    # המרה לגווני אפור אם התמונה צבעונית
    if len(image.shape) == 3:
        gray_image = np.mean(image, axis=2)
    else:
        gray_image = image

    # מסננים לזיהוי קצוות
    filter_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    filter_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # קונבולוציה
    edgeX = convolve2d(gray_image, filter_x, mode='same', boundary='fill', fillvalue=0)
    edgeY = convolve2d(gray_image, filter_y, mode='same', boundary='fill', fillvalue=0)

    # חוזק הקצה
    edgeMAG = np.sqrt(edgeX**2 + edgeY**2)
    return edgeMAG


# --- הרצה עצמאית (רק אם רוצים לראות תוצאה) ---
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # קובץ תמונה
    my_file_name = "my_image.png"

    # 1. טעינת התמונה
    original_img = load_image(my_file_name)

    # 2. ניקוי רעשים עם פילטר מדיאן
    # המרה לגווני אפור אם צריך
    if len(original_img.shape) == 3:
        gray_img = np.mean(original_img, axis=2)
    else:
        gray_img = original_img
    clean_img = median(gray_img, disk(3))

    # 3. זיהוי קצוות
    edges = edge_detection(clean_img)

    # 4. המרה לבינארי
    binary_edges = edges > (np.mean(edges) * 1.5)

    # 5. הצגת התוצאה
    plt.imshow(binary_edges, cmap='gray')
    plt.show()

    # 6. שמירה לקובץ
    final_result = Image.fromarray((binary_edges * 255).astype(np.uint8))
    final_result.save("edge_result.png")

