import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.signal import convolve2d
from skimage.filters import median
from skimage.morphology import ball

# --- The utility functions (what was requested in image_utils) ---

def load_image(path):
img = Image.open(path)
return np.array(img)

def edge_detection(image):
# Turning gray
if len(image.shape) == 3:
gray_image = np.mean(image, axis=2)
else:
gray_image = image

# Edge detection filters
filter_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
filter_y = np.array([[-1, -2, -1], [ 0, 0, 0], [ 1, 2, 1]])

# Convolution
edgeX = convolve2d(gray_image, filter_x, mode='same', boundary='fill', fillvalue=0)
edgeY = convolve2d(gray_image, filter_y, mode='same', boundary='fill', fillvalue=0)

# Edge strength
edgeMAG = np.sqrt(edgeX**2 + edgeY**2)
return edgeMAG

# --- The run itself ---

# Here you link the image! Make sure the name in quotes is the same as the file name you uploaded on the side.
my_file_name = "my_image.png"

#1. Charging
original_img = load_image(my_file_name)

#2. Noise Clearing
clean_img = median(original_img, ball(3))

#3. Edge detection
edges = edge_detection(clean_img)

#4. Converting to black and white (binary)
binary_edges = edges > (np.mean(edges) * 1.5)

# 5. Displaying the result
plt.imshow(binary_edges, cmap='gray')
plt.show()

# 6. Saving the image to your computer (so you can upload it to GitHub)
final_result = Image.fromarray((binary_edges * 255).astype(np.uint8))
final_result.save("edge_result.png")
