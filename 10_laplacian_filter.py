# 10. Laplacian Filter.ipynb


import cv2
import numpy as np
from matplotlib import pyplot as plt


# -----------------------------------------
# ✅ Utility Function (IMPORTANT)
# -----------------------------------------
def load_img(path, gray=False):
    img = cv2.imread(path, 0 if gray else 1)
    if img is None:
        raise Exception(f"❌ Image not found: {path}")
    if not gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


# -----------------------------------------
# Kernels
# -----------------------------------------
kernel_1 = np.array([[0, 1, 0],
                     [1, -4, 1],
                     [0, 1, 0]])

kernel_2 = np.array([[0, -1, 0],
                     [-1, 4, -1],
                     [0, -1, 0]])

kernel_3 = np.array([[0, 1, 0],
                     [1, -8, 1],
                     [0, 1, 0]])

kernel_4 = np.array([[0, -1, 0],
                     [-1, 8, -1],
                     [0, -1, 0]])

sharpen_kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])


# -----------------------------------------
# Example 1
# -----------------------------------------
img = load_img('img5.jpg')

laplacian_1 = cv2.filter2D(img, -1, kernel_1)
laplacian_2 = cv2.filter2D(img, -1, kernel_2)
laplacian_3 = cv2.filter2D(img, -1, kernel_3)
laplacian_4 = cv2.filter2D(img, -1, kernel_4)
laplacian_5 = cv2.filter2D(img, -1, sharpen_kernel)

plt.figure(figsize=(12,8))

plt.subplot(2,3,1)
plt.imshow(img)
plt.title("Original")

plt.subplot(2,3,2)
plt.imshow(laplacian_1)
plt.title("Kernel 1")

plt.subplot(2,3,3)
plt.imshow(laplacian_2)
plt.title("Kernel 2")

plt.subplot(2,3,4)
plt.imshow(laplacian_3)
plt.title("Kernel 3")

plt.subplot(2,3,5)
plt.imshow(laplacian_4)
plt.title("Kernel 4")

plt.subplot(2,3,6)
plt.imshow(laplacian_5)
plt.title("Sharpen")

plt.tight_layout()
plt.show()


# -----------------------------------------
# ✅ Example 2
# -----------------------------------------
# If image not available, it will throw clear error
img2 = load_img(r'C:\Users\LENOVO\Downloads\image processing\img5.jpg')

laplacian_1 = cv2.filter2D(img2, -1, kernel_1)
laplacian_2 = cv2.filter2D(img2, -1, kernel_2)
laplacian_3 = cv2.filter2D(img2, -1, kernel_3)
laplacian_4 = cv2.filter2D(img2, -1, kernel_4)
laplacian_5 = cv2.filter2D(img2, -1, sharpen_kernel)

plt.figure(figsize=(12,8))

plt.subplot(2,3,1)
plt.imshow(img2)
plt.title("Original")

plt.subplot(2,3,2)
plt.imshow(laplacian_1)
plt.title("Kernel 1")

plt.subplot(2,3,3)
plt.imshow(laplacian_2)
plt.title("Kernel 2")

plt.subplot(2,3,4)
plt.imshow(laplacian_3)
plt.title("Kernel 3")

plt.subplot(2,3,5)
plt.imshow(laplacian_4)
plt.title("Kernel 4")

plt.subplot(2,3,6)
plt.imshow(laplacian_5)
plt.title("Sharpen")

plt.tight_layout()
plt.show()