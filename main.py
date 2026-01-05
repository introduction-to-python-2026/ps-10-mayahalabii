import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import median
from skimage.morphology import ball
from PIL import Image

from image_utils import load_image, edge_detection


def main():
    # Step 1: Load image
    image_path = "original_image.png"  # change to your image filename
    image = load_image(image_path)

    # Step 2: Noise suppression (median filter)
    clean_image = median(image, ball(3))

    # Step 3: Edge detection
    edge_mag = edge_detection(clean_image)

    # Step 4: Thresholding (binary image)
    threshold = np.mean(edge_mag)
    edge_binary = edge_mag > threshold

    # Step 5: Display result
    plt.imshow(edge_binary, cmap="gray")
    plt.title("Edge Detected Image")
    plt.axis("off")
    plt.show()

    # Step 6: Save result
    edge_image = Image.fromarray((edge_binary * 255).astype(np.uint8))
    edge_image.save("my_edges.png")


if __name__ == "__main__":
    main()
