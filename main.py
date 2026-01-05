import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import median
from skimage.morphology import disk # שינוי ל-disk במקום ball
from PIL import Image
import os
from google.colab import files # להורדה

# ייבוא הפונקציות שלך
from image_utils import load_image, edge_detection

def main():
    # שלב 1: טעינת התמונה
    image_path = "original_image.png" 
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found!")
        return
    
    image = load_image(image_path)

    # שלב 2: המרה לשחור לבן וניקוי רעשים
    # אם התמונה צבעונית (3 ערוצים), נהפוך אותה לדו-מימדית
    if len(image.shape) == 3:
        image = np.mean(image, axis=2).astype(np.uint8)
    
    # שימוש ב-disk(3) עבור תמונה דו-מימדית
    clean_image = median(image, disk(3))

    # שלב 3: זיהוי קצוות
    edge_mag = edge_detection(clean_image)

    # שלב 4: סף (Thresholding)
    # שימי לב: לעיתים np.mean נמוך מדי, אבל לטסט זה בדרך כלל בסדר
    threshold = np.mean(edge_mag)
    edge_binary = edge_mag > threshold

    # שלב 5: הצגת תוצאה
    plt.imshow(edge_binary, cmap="gray")
    plt.title("Edge Detected Image")
    plt.axis("off")
    plt.show()

    # שלב 6: שמירה והורדה
    # חשוב להכפיל ב-255 ולהמיר ל-uint8
    edge_image = Image.fromarray((edge_binary * 255).astype(np.uint8))
    output_name = "my_edges.png"
    edge_image.save(output_name)
    
    print(f"Saved {output_name}. Downloading...")
    files.download(output_name)

if __name__ == "__main__":
    main()
