# 6. Median Filter.ipynb


import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image

img = cv2.imread(r'C:\Users\LENOVO\Downloads\image processing\img3.jpg')

img.shape

median = cv2.medianBlur(img,5) # 5 window size in image => 5*5

plt.figure(figsize=(15,15))

plt.subplot(221),
plt.imshow(img),
plt.title("Original Image")

plt.subplot(222),
plt.imshow(median),
plt.title("Median Filtered Image")

