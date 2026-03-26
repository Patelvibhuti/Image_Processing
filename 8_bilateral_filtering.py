#8. Bilateral Filtering.ipynb

import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread(r'C:\Users\LENOVO\Downloads\image processing\img7.jpg')

output_1 = cv2.bilateralFilter(img, 5,  50, 50) 
output_2 = cv2.bilateralFilter(img, 7,  60, 60) 
output_3 = cv2.bilateralFilter(img, 10, 70, 70) 
output_4 = cv2.bilateralFilter(img, 12, 75, 75) 
output_5 = cv2.bilateralFilter(img, 15, 85, 80)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


plt.figure(figsize=(15,15))

plt.subplot(231),
plt.imshow(img),
plt.title("Original Image")

plt.subplot(232),
plt.imshow(output_1),
plt.title("Output - 1")

plt.subplot(233),
plt.imshow(output_2),
plt.title("Output - 2")

plt.subplot(234),
plt.imshow(output_3),
plt.title("Output - 3")

plt.subplot(235),
plt.imshow(output_4),
plt.title("Output - 4")

plt.subplot(236),
plt.imshow(output_5),
plt.title("Output - 5")

