#7. Minimum & Maximum Filter.ipynb
import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image, ImageFilter

img = Image.open(r"img6.jpg")

min_img = img.filter(ImageFilter.MinFilter(size = 3))

plt.figure(figsize=(15,15))

plt.subplot(221),
plt.imshow(img),
plt.title("Original Image")

plt.subplot(222),
plt.imshow(min_img),
plt.title("Minmum Filtered Image")

max_img = img.filter(ImageFilter.MaxFilter(size = 3))

plt.figure(figsize=(15,15))

plt.subplot(221),
plt.imshow(img),
plt.title("Original Image")

plt.subplot(222),
plt.imshow(max_img),
plt.title("Maximum Filtered Image")

