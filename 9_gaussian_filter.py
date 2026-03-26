# 9. Gaussian Filter.ipynb


import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread(r'C:\Users\LENOVO\Downloads\image processing\img5.jpg')

"""![image.png](attachment:image.png)"""

mask = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
print(mask)

weight = mask.sum()
print(weight)

kernel = mask/weight
print(kernel)

"""![image.png](attachment:image.png)"""

mask = np.array([[1, 4, 7, 4, 1], [4, 16, 26, 16, 4], [7, 26, 41, 26, 7],[4, 16, 26, 16, 4],[1, 4, 7, 4, 1]])
print(mask)

weight = mask.sum()
print(weight)

kernel_1 = mask/weight
print(kernel_1)

mask = np.array([[1, 4, 7, 9, 7, 4, 1], [4, 16, 26, 36, 26, 16, 4], [7, 26, 41, 55, 41, 26, 7],[4, 16, 26, 36, 26, 16, 4],[1, 4, 7, 9, 7, 4, 1]])
print(mask)

weight = mask.sum()
print(weight)

kernel_2 = mask/weight
print(kernel_2)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

gaussignBlur = cv2.GaussianBlur(img,kernel.shape,0)  # applying gaussian blur with kernel size 5 x 5 and standard deviation as 0
gaussignBlur_1 = cv2.GaussianBlur(img,kernel_1.shape,0)  # applying gaussian blur with kernel size 5 x 5 and standard deviation as 0
gaussignBlur_2 = cv2.GaussianBlur(img,kernel_2.shape,0)  # applying gaussian blur with kernel size 7 x 7 and standard deviation as 0

plt.figure(figsize=(15,15))

plt.subplot(221),
plt.imshow(img),
plt.title("Original Image")

plt.subplot(222),
plt.imshow(gaussignBlur),
plt.title("Gaussian Filter 3x3")

plt.subplot(223),
plt.imshow(gaussignBlur_1),
plt.title("Gaussian Filter 5x5")

plt.subplot(224),
plt.imshow(gaussignBlur_2),
plt.title("Gaussian Filter 7x7")



