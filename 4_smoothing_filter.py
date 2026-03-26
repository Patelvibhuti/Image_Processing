
# """4. Smoothing Filter.ipynb
import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image, ImageFilter
import os

# Load image
img_path = os.path.join(os.getcwd(), "img5.jpg")
img = cv2.imread(img_path)
if img is None:
    print("Image not found at:", img_path)
    exit()

# Convert to RGB for visualization
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# -------------------------------
# Gaussian Blur using OpenCV
gaussian_blur = cv2.GaussianBlur(img_rgb, (5, 5), 0)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title("Original Image")

plt.subplot(1, 2, 2)
plt.imshow(gaussian_blur)
plt.title("Gaussian Blur (OpenCV)")
plt.show()

# -------------------------------
# Gaussian Blur using PIL
image_pil = Image.open(img_path)
blurred_pil = image_pil.filter(ImageFilter.GaussianBlur(radius=2))

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image_pil)
plt.title("Original (PIL)")

plt.subplot(1, 2, 2)
plt.imshow(blurred_pil)
plt.title("Gaussian Blur (PIL)")
plt.show()

# -------------------------------
# Gaussian Blur to a Portion
portion_blur_img = Image.open(img_path)
cropped = portion_blur_img.crop((0, 0, 150, 150))
blurred_crop = cropped.filter(ImageFilter.GaussianBlur(radius=3))
portion_blur_img.paste(blurred_crop, (0, 0))

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(Image.open(img_path))
plt.title("Original Image")

plt.subplot(1, 2, 2)
plt.imshow(portion_blur_img)
plt.title("Partial Gaussian Blur")
plt.show()

# -------------------------------
# Median Filter using OpenCV
median_filtered = cv2.medianBlur(img_rgb, 5)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title("Original Image")

plt.subplot(1, 2, 2)
plt.imshow(median_filtered)
plt.title("Median Filter (OpenCV)")
plt.show()

# -------------------------------
# Manual Median Filter using NumPy
gray = cv2.imread(img_path, 0)
m, n = gray.shape
img_manual = np.zeros((m, n))

for i in range(1, m-1):
    for j in range(1, n-1):
        neighbors = gray[i-1:i+2, j-1:j+2].flatten()
        img_manual[i, j] = np.median(neighbors)

img_manual = img_manual.astype(np.uint8)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(gray, cmap='gray')
plt.title("Original Grayscale")

plt.subplot(1, 2, 2)
plt.imshow(img_manual, cmap='gray')
plt.title("Manual Median Filter")
plt.show()
