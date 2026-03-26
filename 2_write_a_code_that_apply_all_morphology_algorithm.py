#2. Write a code that apply all morphology algorithm.ipynb

import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("opencv_morphological_ops_pyimagesearch_logo.png")

img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img1

grayimage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

grayimage

kernel = np.ones((5,5),np.uint8)

kernel

erosion = cv2.erode(img1, kernel, iterations=1)
dilation = cv2.dilate(img1, kernel, iterations=1)
opening = cv2.morphologyEx(img1, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(img1, cv2.MORPH_CLOSE, kernel)
gradiant = cv2.morphologyEx(img1, cv2.MORPH_GRADIENT, kernel)
tophat  = cv2.morphologyEx(img1, cv2.MORPH_TOPHAT, kernel)
blackhat = cv2.morphologyEx(img1, cv2.MORPH_BLACKHAT, kernel)

plt.subplot(241),plt.imshow(img1),plt.title("Original Image")
plt.xticks([]), plt.yticks([])

plt.subplot(242),plt.imshow(grayimage, cmap="gray"),plt.title("Gray Image")
plt.xticks([]), plt.yticks([])

plt.subplot(243),plt.imshow(erosion, cmap="gray"),plt.title("Erosion")
plt.xticks([]), plt.yticks([])

plt.subplot(244),plt.imshow(dilation, cmap="gray"),plt.title("Dilation")
plt.xticks([]), plt.yticks([])

plt.subplot(245),plt.imshow(opening),plt.title("Opening")
plt.xticks([]), plt.yticks([])

plt.subplot(246),plt.imshow(closing),plt.title("Closing")
plt.xticks([]), plt.yticks([])

plt.subplot(247),plt.imshow(tophat, cmap="gray"),plt.title("Tophat")
plt.xticks([]), plt.yticks([])

plt.subplot(248),plt.imshow(blackhat, cmap="gray"),plt.title("Blackhat")
plt.xticks([]), plt.yticks([])

plt.show()



