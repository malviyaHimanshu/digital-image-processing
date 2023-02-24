"""
    CS20B1097 HIMANSHU

    2) Down sample the grayscale Lena image with 8 different intensity ranges of values (0-
    255, 0-128, 0-64, 0-32, 0-16, 0-8, 0-4, and 0-2). (Note: Size of images are the same).
    And display all those 8 downsampled images in the same size display area on the
    screen. Observe what happens.
"""

import cv2
import numpy as np

img = cv2.imread('Lena.png', 0)
cv2.imshow("Original Image", img)

max = np.max(img)
min = np.min(img)

img = (img - min)/(max - min)
new_img = np.uint8(img*255)
cv2.imshow(f"Image with Intensity range (0-255)", new_img)
for i in range(7):
    range = int(128/(2**i))
    new_img = np.uint8(img*range)
    cv2.imshow(f"Image with Intensity range (0-{range})", new_img)

cv2.waitKey(0)
cv2.destroyAllWindows()