# Mean (Average) & Weighted Average Filter - Clean Version

import numpy as np
import cv2
from matplotlib import pyplot as plt
import os

# -----------------------------
# Step 1: Load Image Safely
# -----------------------------
image_path = "img7.jpg"   # 🔁 Change if needed

if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image not found: {image_path}")

img = cv2.imread(image_path)

if img is None:
    raise ValueError("Error loading image!")

# Convert to RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

print("Original Shape:", img.shape)

# -----------------------------
# Step 2: Resize Image
# -----------------------------
resize = cv2.resize(img, (0, 0), fx=0.2, fy=0.2)
m, n, o = resize.shape

print("Resized Shape:", resize.shape)

# -----------------------------
# Step 3: Display Images
# -----------------------------
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(img)
plt.title("Original Image")

plt.subplot(1,2,2)
plt.imshow(resize)
plt.title("Resized Image")

plt.show()

# -----------------------------
# Step 4: Mean Filter (Manual)
# -----------------------------
mask = np.ones((3,3)) / 9
print("Mean Kernel:\n", mask)

img_mean_manual = np.zeros((m, n, o))

for i in range(1, m-1):
    for j in range(1, n-1):
        temp = (
            resize[i-1, j-1]*mask[0,0] + resize[i-1, j]*mask[0,1] + resize[i-1, j+1]*mask[0,2] +
            resize[i, j-1]*mask[1,0]   + resize[i, j]*mask[1,1]   + resize[i, j+1]*mask[1,2] +
            resize[i+1, j-1]*mask[2,0] + resize[i+1, j]*mask[2,1] + resize[i+1, j+1]*mask[2,2]
        )
        img_mean_manual[i, j] = temp

img_mean_manual = img_mean_manual.astype(np.uint8)

# -----------------------------
# Step 5: Mean Filter (OpenCV)
# -----------------------------
mean_cv = cv2.blur(resize, (3,3))

# -----------------------------
# Step 6: Weighted Filter (Manual)
# -----------------------------
kernel = np.array([[1,2,1],
                   [2,4,2],
                   [1,2,1]], dtype=float)

kernel = kernel / kernel.sum()
print("Weighted Kernel:\n", kernel)

img_weight_manual = np.zeros((m, n, o))

for i in range(1, m-1):
    for j in range(1, n-1):
        temp = (
            resize[i-1, j-1]*kernel[0,0] + resize[i-1, j]*kernel[0,1] + resize[i-1, j+1]*kernel[0,2] +
            resize[i, j-1]*kernel[1,0]   + resize[i, j]*kernel[1,1]   + resize[i, j+1]*kernel[1,2] +
            resize[i+1, j-1]*kernel[2,0] + resize[i+1, j]*kernel[2,1] + resize[i+1, j+1]*kernel[2,2]
        )
        img_weight_manual[i, j] = temp

img_weight_manual = img_weight_manual.astype(np.uint8)

# -----------------------------
# Step 7: Weighted Filter (OpenCV)
# -----------------------------
weight_cv = cv2.filter2D(resize, -1, kernel)

# -----------------------------
# Step 8: Display Results
# -----------------------------
plt.figure(figsize=(15,10))

plt.subplot(2,3,1)
plt.imshow(resize)
plt.title("Original (Resized)")

plt.subplot(2,3,2)
plt.imshow(img_mean_manual)
plt.title("Mean Filter (Manual)")

plt.subplot(2,3,3)
plt.imshow(mean_cv)
plt.title("Mean Filter (OpenCV)")

plt.subplot(2,3,4)
plt.imshow(img_weight_manual)
plt.title("Weighted (Manual)")

plt.subplot(2,3,5)
plt.imshow(weight_cv)
plt.title("Weighted (OpenCV)")

plt.tight_layout()
plt.show()