#1. Write a code that RGB image to gray and also negative.ipynb

import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt

print(sys.path[0])
# img = cv2.imread(sys.path[0]+"/img.jpg", 1)
img = cv2.imread('img7.jpg')

#we need to transform this in order that Matplotlib reads it correctly
fix_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(fix_img)
# fix_img

"""### 1. Average Method

Average method is the most simple one. You just have to take the average of three colors. Since its an RGB image, so it means that you have add R with G with B and then divide it by 3 to get your desired grayscale image.
"""

#Let's extract the three channels
R, G, B = fix_img[:,:,0], fix_img[:,:,1],fix_img[:,:,2]

grayscale_average_img = np.mean(fix_img, axis=2) # axis=2 means 2 dimension - 2D
# grayscale_average_img =  R/3 + G/3 +  B/3
# print(grayscale_average_img)

plt.imshow(grayscale_average_img, cmap='gray')
plt.savefig('image_average_method.png')

"""### 2. Weighted average

The weighted method, also called the luminosity method, weighs red, green, and blue according to their wavelengths.

#### Grayscale = 0.299R + 0.587G + 0.114B

above weighted average means 29& red color, 58% of green color & 11% of blue color participate in the image
"""

grayscale_weighted_img = 0.299 * R + 0.587 * G + 0.114 * B
# print(grayscale_weighted_img.shape)

plt.imshow(grayscale_average_img, cmap='gray')
# plt.savefig('image_average_method.png')



"""### Using OpenCV"""

grayimage = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

nagativeimg = 255 - fix_img

plt.subplot(231),plt.imshow(fix_img),plt.title("RGB Image")
plt.xticks([]), plt.yticks([])

plt.subplot(232),plt.imshow(grayimage, cmap="gray"),plt.title("Gray Image")
plt.xticks([]), plt.yticks([])

plt.subplot(233),plt.imshow(nagativeimg),plt.title("Negative Image")
plt.xticks([]), plt.yticks([])

plt.show()

cv2.imshow('image',img)
cv2.imshow('image1',grayimage)
cv2.imshow('image2',nagativeimg)

cv2.waitKey(0)
cv2.destroyAllWindows()







