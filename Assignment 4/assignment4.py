"""
    CS20B1097 HIMANSHU

    Download the leaning tower of the PISA image and 
    find the angle of inclination using appropriate 
    rotations with bilinear interpolation.
"""

import cv2
import numpy as np

def image_rotation(image, angle):
    angle = np.radians(angle)

    height = image.shape[0]
    width = image.shape[1]
    final_image = np.uint8(np.zeros(image.shape))
    x0, y0 = (width//2), (height//2)

    for x in range(height):
        for y in range(width):
            x_new = (x-x0) * np.cos(angle) + (y-y0) * np.sin(angle)
            y_new = -(x-x0) * np.sin(angle) + (y-y0) * np.cos(angle)
            x_new = round(x_new) + x0
            y_new = round(y_new) + y0

            if (x_new >= 0 and y_new >= 0 and x_new < image.shape[0] and y_new < image.shape[1]):
                final_image[x, y, :] = image[x_new, y_new, :]

    # for x in range(height):
    #     for y in range(width):
    #         if final_image[x, y, :].all() == 0:
    #             if x<height-1 and y<width-1 and x>1 and y>1:
    #                 for z in range(3):
    #                     final_image[x, y, z] = (final_image[x+1, y, z] + final_image[x-1, y, z] + final_image[x, y+1, z] + final_image[x, y-1, z])/4

    return final_image


image = cv2.imread("PISA.jpg")
cv2.imshow("Original Image", image)

# angle in degree
angle = 8

# Image Rotation using user defined function
rotated_image_userdef = image_rotation(image, angle)
cv2.imshow("Rotated Image using User defined function", rotated_image_userdef)

# Image Rotation using Built-in function
matrix = cv2.getRotationMatrix2D((image.shape[0]/2, image.shape[1]/2), angle, 1)
rotated_image_buitin = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))
cv2.imshow("Rotated Image using Buit-in function", rotated_image_buitin)

cv2.waitKey(0)
cv2.destroyAllWindows()