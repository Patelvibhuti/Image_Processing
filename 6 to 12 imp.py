

# # 6. Median Filter

import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
img = cv2.imread(r'C:\Users\LENOVO\Downloads\image processing\img6.jpg')
print(img.shape)
median = cv2.medianBlur(img,5) # 5 window size in image => 5*5
plt.figure(figsize=(15,15))

plt.subplot(221),
plt.imshow(img),
plt.title("Original Image")

plt.subplot(222),
plt.imshow(median),
plt.title("Median Filtered Image")




# # 7. Minimum & Maximum Filter
import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image, ImageFilter
img = Image.open(r"C:\Users\LENOVO\Downloads\image processing\img6.jpg")
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


# # 8. Bilateral Filtering

import numpy as np
import cv2
from matplotlib import pyplot as plt
img = cv2.imread(r"C:\Users\LENOVO\Downloads\image processing\img5.jpg")
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


# # 9. Gaussian Filter

import numpy as np
import cv2
from matplotlib import pyplot as plt
img = cv2.imread(r"C:\Users\LENOVO\Downloads\image processing\lena_color_256.jpg")
mask = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
print(mask)

weight = mask.sum()
print(weight)

kernel = mask/weight
print(kernel)
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


# # 10. Laplacian Filter

import cv2
import numpy as np
from matplotlib import pyplot as plt
originalImage = cv2.imread(r'C:\Users\LENOVO\Downloads\image processing\img5.jpg',0)
img = cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB)
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

### Example : 1
laplacian_1 = cv2.filter2D(src=img,ddepth=-1,kernel = kernel_1)  
laplacian_2 = cv2.filter2D(src=img,ddepth=-1,kernel = kernel_2)
laplacian_3 = cv2.filter2D(src=img,ddepth=-1,kernel = kernel_3)
laplacian_4 = cv2.filter2D(src=img,ddepth=-1,kernel = kernel_4)
laplacian_5 = cv2.filter2D(src=img,ddepth=-1,kernel = sharpen_kernel)

plt.figure(figsize=(15,15))

plt.subplot(421),
plt.imshow(originalImage),
plt.title("Original Image")

plt.subplot(422),
plt.imshow(img),
plt.title("GreyScale Image")

plt.subplot(423),
plt.imshow(laplacian_1),
plt.title("Laplacian Filter 3x3")

plt.subplot(424),
plt.imshow(laplacian_2),
plt.title("Laplacian Filter 3x3")

plt.subplot(425),
plt.imshow(laplacian_3),
plt.title("Laplacian Filter 3x3")


plt.subplot(426),
plt.imshow(laplacian_4),
plt.title("Laplacian Filter 3x3")

plt.subplot(427),
plt.imshow(laplacian_4),
plt.title("Laplacian Filter 3x3")


# In[23]:


### Example : 2

originalImage = cv2.imread(r'C:\Users\LENOVO\Downloads\image processing\img4.jpg',0)
img = cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB)

laplacian_1 = cv2.filter2D(src=img,ddepth=-1,kernel = kernel_1)  
laplacian_2 = cv2.filter2D(src=img,ddepth=-1,kernel = kernel_2)
laplacian_3 = cv2.filter2D(src=img,ddepth=-1,kernel = kernel_3)
laplacian_4 = cv2.filter2D(src=img,ddepth=-1,kernel = kernel_4)
laplacian_5 = cv2.filter2D(src=img,ddepth=-1,kernel = sharpen_kernel)

plt.figure(figsize=(15,15))

plt.subplot(421),
plt.imshow(originalImage),
plt.title("Original Image")

plt.subplot(422),
plt.imshow(img),
plt.title("GreyScale Image")

plt.subplot(423),
plt.imshow(laplacian_1),
plt.title("Laplacian Filter 3x3")

plt.subplot(424),
plt.imshow(laplacian_2),
plt.title("Laplacian Filter 3x3")

plt.subplot(425),
plt.imshow(laplacian_3),
plt.title("Laplacian Filter 3x3")


plt.subplot(426),
plt.imshow(laplacian_4),
plt.title("Laplacian Filter 3x3")

plt.subplot(427),
plt.imshow(laplacian_4),
plt.title("Laplacian Filter 3x3")


# # 11. Sobel Operator

import cv2
import numpy as np
from matplotlib import pyplot as plt
originalImage = cv2.imread(r'C:\Users\LENOVO\Downloads\image processing\img7.jpg',0)
img = cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB)
vertical_kernel = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])

horizontal_kernel = np.array([[1, 2, 1],
                              [0, 0, 0],
                              [-1, -2, -1]])
vertical_output = cv2.filter2D(src=img,ddepth=-1,kernel = vertical_kernel)
horizontal_output = cv2.filter2D(src=img,ddepth=-1,kernel = horizontal_kernel)  


plt.figure(figsize=(15,15))

plt.subplot(321),
plt.imshow(img),
plt.title("Original Image")

plt.subplot(322),
plt.imshow(horizontal_output),
plt.title("Horizontal Features")

plt.subplot(323),
plt.imshow(vertical_output),
plt.title("Vertical Features")



# Remove the negative values taking the absolute
horizontal_output = np.absolute(horizontal_output)
vertical_output = np.absolute(vertical_output)

sobel = horizontal_output + vertical_output

plt.figure(figsize=(25,15))


plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(horizontal_output,cmap = 'gray')
plt.title('horizontal features - sobel-x'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(vertical_output,cmap = 'gray')
plt.title('vertical features - sobel-y'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(sobel,cmap = 'gray')
plt.title('vertical + vertical features - sobel= x+y'), plt.xticks([]), plt.yticks([])

plt.show()




####################################333
originalImage = cv2.imread(r"C:\Users\LENOVO\Downloads\image processing\img3.jpg",0)
img = cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB)

vertical_output = cv2.filter2D(src=img,ddepth=-1,kernel = vertical_kernel)
horizontal_output = cv2.filter2D(src=img,ddepth=-1,kernel = horizontal_kernel)  


plt.figure(figsize=(15,15))

plt.subplot(321),
plt.imshow(img),
plt.title("Original Image")

plt.subplot(322),
plt.imshow(horizontal_output),
plt.title("Horizontal Features")

plt.subplot(323),
plt.imshow(vertical_output),
plt.title("Vertical Features")


# Remove the negative values taking the absolute
horizontal_output = np.absolute(horizontal_output)
vertical_output = np.absolute(vertical_output)

sobel = horizontal_output + vertical_output

plt.figure(figsize=(15,15))

plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(horizontal_output,cmap = 'gray')
plt.title('horizontal features - sobel-x'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(vertical_output,cmap = 'gray')
plt.title('vertical features - sobel-y'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(sobel,cmap = 'gray')
plt.title('vertical + vertical features - sobel= x+y'), plt.xticks([]), plt.yticks([])

plt.show()


# # 12. Unsharp Masking

import numpy as np
import cv2
from matplotlib import pyplot as plt
### Example : 1

original_image = cv2.imread(r"C:\Users\LENOVO\Downloads\image processing\img4.jpg")

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

### Example : 2
original_image = cv2.imread(r"C:\Users\LENOVO\Downloads\image processing\img4.jpg")

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



image = cv2.imread(r"C:\Users\LENOVO\Downloads\image processing\img6.jpg")
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





