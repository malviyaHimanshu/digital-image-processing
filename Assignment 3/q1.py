"""
    CS20B1097 HIMANSHU

    1) Convert the given Lena image to grayscale image. Use the cv2.resize() to down sample
    the image with 4 sizes (128*128, 64*64, 32*32, and 16*16). Display the original image,
    and down sampled images with the same display size. Observe what happens.
"""

import cv2

img = cv2.imread('Lena.png', 0)
cv2.imshow('Original Image', img)

height, width = img.shape[:2]

for i in range(4):
    res_img = cv2.resize(img, (16*(2**i), (16*(2**i))), interpolation=cv2.INTER_CUBIC)
    cv2.imshow(f'({16*(2**i)}, {(16*(2**i))})', res_img)

cv2.waitKey(0)
cv2.destroyAllWindows()