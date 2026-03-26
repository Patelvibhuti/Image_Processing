#11. Sobel Operator.ipynb


import cv2
import numpy as np
from matplotlib import pyplot as plt


# -----------------------------------------
# Load Image (COLOR, not grayscale)
# -----------------------------------------
path = r'C:\Users\LENOVO\Downloads\image processing\img5.jpg'

originalImage = cv2.imread(path)

if originalImage is None:
    raise Exception("❌ Image not found. Check path!")

# Convert BGR → RGB
img = cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB)


# -----------------------------------------
# Sobel Kernels (FIXED)
# -----------------------------------------
vertical_kernel = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])

horizontal_kernel = np.array([[1, 2, 1],
                              [0, 0, 0],
                              [-1, -2, -1]])


# -----------------------------------------
# Apply Filters
# -----------------------------------------
vertical_output = cv2.filter2D(img, -1, vertical_kernel)
horizontal_output = cv2.filter2D(img, -1, horizontal_kernel)


# -----------------------------------------
# First Plot
# -----------------------------------------
plt.figure(figsize=(10,8))

plt.subplot(2,2,1)
plt.imshow(img)
plt.title("Original Image")

plt.subplot(2,2,2)
plt.imshow(horizontal_output)
plt.title("Horizontal Features")

plt.subplot(2,2,3)
plt.imshow(vertical_output)
plt.title("Vertical Features")


# Convert to absolute
horizontal_output = np.abs(horizontal_output)
vertical_output = np.abs(vertical_output)

sobel = horizontal_output + vertical_output


# -----------------------------------------
# Second Plot
# -----------------------------------------
plt.figure(figsize=(10,8))

plt.subplot(2,2,1)
plt.imshow(img, cmap='gray')
plt.title('Original')
plt.xticks([]), plt.yticks([])

plt.subplot(2,2,2)
plt.imshow(horizontal_output, cmap='gray')
plt.title('Sobel-X')
plt.xticks([]), plt.yticks([])

plt.subplot(2,2,3)
plt.imshow(vertical_output, cmap='gray')
plt.title('Sobel-Y')
plt.xticks([]), plt.yticks([])

plt.subplot(2,2,4)
plt.imshow(sobel, cmap='gray')
plt.title('Sobel X + Y')
plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()