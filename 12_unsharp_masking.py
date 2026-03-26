# 12. Unsharp Masking.ipynb
import numpy as np
import cv2
from matplotlib import pyplot as plt

"""### Example : 1"""

original_image = cv2.imread(r"C:\Users\LENOVO\Downloads\image processing\img5.jpg")

img = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

plt.imshow(img),
plt.title("Original Image")


gaussignBlur = cv2.GaussianBlur(img,(5,5),0)  # applying gaussian blur with kernel size 5 x 5 and standard deviation as 0

UnsharpMask = img - gaussignBlur

plt.figure(figsize=(15,15))

plt.subplot(221),
plt.imshow(gaussignBlur),
plt.title("Gaussian Filter")

plt.subplot(222),
plt.imshow(UnsharpMask),
plt.title("Unsharp Filter")

"""### Example : 2"""

original_image = cv2.imread(r"C:\Users\LENOVO\Downloads\image processing\img7.jpg")

img = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

plt.imshow(img),
plt.title("Original Image")


gaussignBlur = cv2.GaussianBlur(img,(5,5),0)  # applying gaussian blur with kernel size 5 x 5 and standard deviation as 0

UnsharpMask = img - gaussignBlur

plt.figure(figsize=(15,15))

plt.subplot(221),
plt.imshow(gaussignBlur),
plt.title("Gaussian Filter")

plt.subplot(222),
plt.imshow(UnsharpMask),
plt.title("Unsharp Filter")

"""### Example : 3 : Using Library"""

from PIL import Image, ImageFilter

image = cv2.imread('img6.jpg')
image = Image.fromarray(image.astype('uint8'))

new_image = image.filter(ImageFilter.UnsharpMask(radius=2, percent=150))


plt.figure(figsize=(15,15))

plt.subplot(221),
plt.imshow(image),
plt.title("Original Image")

plt.subplot(222),
plt.imshow(new_image),
plt.title("Unsharp Filter")


plt.show()

