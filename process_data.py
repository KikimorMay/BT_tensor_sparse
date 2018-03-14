from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

img=Image.open('Dataset/street/4.jpg')  #打开图像
plt.figure("beauty")
plt.subplot(1,2,1)
plt.title('origin')
plt.imshow(img)
plt.axis('off')

box=(100,100,1200,500)
roi=img.crop(box)
plt.subplot(1,2,2), plt.title('roi')
plt.imshow(roi),plt.axis('off')

plt.show()