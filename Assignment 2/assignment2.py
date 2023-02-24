"""
    CS20B1097 HIMANSHU

    ASSIGNMENT 2
    - Get good quality image from noise images fi where 1 <= i <= n.
    - Find g = 1/n ( ∑𝑛 fi(x,y) )
           = 1/n ( ∑𝑛 (f(x,y) + ni(x,y)) )
                                where ni(x,y) is the Gaussian noise with
                                mean=0 and variance=1

    - Display f, g and f1, f2, f3, ... fn
    - Do it for n=5, n=10, n=20, and n=30
"""

import cv2
import numpy as np
from skimage.util import random_noise

img = cv2.imread("Lena.png")
cv2.imshow('Original Image', img)
img_height = img.shape[0]
img_width = img.shape[1]

def process_image(img, n):
    final_image = np.empty([img_height, img_width, 3])
    for i in range(n):
        noise_img = random_noise(img, mode='gaussian', mean=0, var=1)
        noise_img = np.array(255*noise_img, dtype='uint8')
        # cv2.imshow(f"(n={n}), Image {i}", noise_img)
        final_image += noise_img

    final_image /= n

    cv2.imshow(f'Final Image (n={n})', final_image.astype(np.uint8))

process_image(img, n=5)
process_image(img, n=10)
process_image(img, n=20)
process_image(img, n=30)

cv2.waitKey(0)
cv2.destroyAllWindows()