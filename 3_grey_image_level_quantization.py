#3. Grey Image Level - Quantization.ipynb

import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt

print(sys.path[0])
img = cv2.imread('img3.jpg')

#we need to transform this in order that Matplotlib reads it correctly
fix_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(fix_img)
# fix_img

#Let's extract the three channels
R, G, B = fix_img[:,:,0], fix_img[:,:,1],fix_img[:,:,2]

grayscale_weighted_img = 0.299 * R + 0.587 * G + 0.114 * B
# print(grayscale_weighted_img.shape)

plt.imshow(grayscale_weighted_img, cmap='gray')
plt.savefig('image_average_method.png')



import cv2
im_gray = cv2.imread('image_average_method.png', cv2.IMREAD_GRAYSCALE)

thresh = 200
img_1 = cv2.threshold(im_gray, thresh, 100, cv2.THRESH_BINARY)[1]
# cv2.imwrite('img_1.png', im_bw)

thresh = 100
img_2 = cv2.threshold(im_gray, thresh, 150, cv2.THRESH_BINARY)[1]
# cv2.imwrite('img_1.png', im_bw)

thresh = 200
img_3 = cv2.threshold(im_gray, thresh, 256, cv2.THRESH_BINARY)[1]
# cv2.imwrite('img_1.png', im_bw)

thresh = 50
img_4 = cv2.threshold(im_gray, thresh, 500, cv2.THRESH_BINARY)[1]
# cv2.imwrite('img_1.png', im_bw)

plt.subplot(221),plt.imshow(img_1, cmap="gray"),plt.title("Grey Image - 1")
plt.xticks([]), plt.yticks([])

plt.subplot(222),plt.imshow(img_2, cmap="gray"),plt.title("Grey Image - 2")
plt.xticks([]), plt.yticks([])

plt.subplot(223),plt.imshow(img_3, cmap="gray"),plt.title("Grey Image - 3")
plt.xticks([]), plt.yticks([])

plt.subplot(224),plt.imshow(img_4, cmap="gray"),plt.title("Grey Image - 4")
plt.xticks([]), plt.yticks([])


plt.show()







