"""
    CS20B1097 HIMANSHU

    ASSIGNMENT 1
    - Take the Lena image and convert it into gray scale image. 
    - Scale it by factors of 1, 2, 0.5 using bilinear interpolation methods. 
    - Display the scaled images. Also, display the output of built-in functions for doing scaling by factors of 0.5, 1 and 2. 
    - Compare the results.
"""

import cv2
import math
import numpy as np

# Taking the lena image and converting it into gray scale image.
img = cv2.imread("Lena.png", 0)
cv2.imshow("Grayscale Image", img)
img_height = img.shape[0]
img_width = img.shape[1]

"""
            X1
    A •-----•--------• B
            |
            • Y 
            |
            |
    C •-----•--------• D
            X2
"""

# Scaling using bilinear interpolation
def bilinear_interpolation(img, scale):
    img_height = img.shape[0]
    img_width = img.shape[1]

    height = int(scale * img_height)
    width = int(scale * img_width)

    # scaled image which will be the output
    scaled_img = np.empty([height, width])

    # ratio in x and y direction
    x_ratio = 1/scale
    y_ratio = 1/scale
    # x_ratio = float(img_width - 1)/(width - 1) if width > 1 else 0
    # y_ratio = float(img_height - 1)/(height - 1) if height > 1 else 0

    for i in range(1, height+1):
        for j in range(1, width+1):
            # first finding the horizontal points using linear interpolation (X1, X2)
            x_low = math.floor(x_ratio*j)
            x_high = math.ceil(x_ratio*j)
            x_weight = (x_ratio*j) - x_low

            # then wrt those points (X1, X2) finding vertical point using linear interpolation (Y)
            y_low = math.floor(y_ratio*i)
            y_high = math.ceil(y_ratio*i)
            y_weight = (y_ratio*i) - y_low

            # getting value of all four points A, B, C, D
            a = img[y_low-1, x_low-1]
            b = img[y_low-1, x_high-1]
            c = img[y_high-1, x_low-1]
            d = img[y_high-1, x_high-1]

            """
            Explanation:
                X1 = a(1-wx) + b(wx)
                X2 = c(1-wx) + d(wx)
                Y = X1(1-wy) + X2(wy)
                by putting the values of X1 and X2, the following equation will be derived
            """
            pixel_value = ((a*(1-x_weight) + b*x_weight) * (1-y_weight)) + ((c*(1-x_weight) + d*x_weight) * y_weight)
            scaled_img[i-1][j-1] = pixel_value

    return scaled_img.astype(np.uint8)


# Using Bilinear Interpolation (scale=1)
# scale_factor = 1
# scaled_img1 = bilinear_interpolation(img, scale_factor)
# cv2.imshow("Scaled Image using Bilinear Interpolation (scale=1)", scaled_img1)

# # Using built-in function
# scaled_dim1 = (math.ceil(scale_factor * img_width), math.ceil(scale_factor * img_height))
# scaled_img_builtin1 = cv2.resize(img, scaled_dim1, interpolation=cv2.INTER_CUBIC)
# cv2.imshow("Scaled Image using Built-in Functions (scale=1)", scaled_img_builtin1)


# Using Bilinear Interpolation (scale=2)
scale_factor = 2
scaled_img2 = bilinear_interpolation(img, 2)
cv2.imshow("Scaled Image using Bilinear Interpolation (scale=2)", scaled_img2)

# Using built-in function
scaled_dim2 = (math.ceil(scale_factor * img_width), math.ceil(scale_factor * img_height))
scaled_img_builtin2 = cv2.resize(img, scaled_dim2, interpolation=cv2.INTER_CUBIC)
cv2.imshow("Scaled Image using Built-in Functions (scale=2)", scaled_img_builtin2)


# Using Bilinear Interpolation (scale=0.5)
scale_factor = 0.5
scaled_img3 = bilinear_interpolation(img, scale_factor)
cv2.imshow("Scaled Image using Bilinear Interpolation (scale=0.5)", scaled_img3)

# Using built-in function
scaled_dim3 = (math.ceil(scale_factor * img_width), math.ceil(scale_factor * img_height))
scaled_img_builtin3 = cv2.resize(img, scaled_dim3, interpolation=cv2.INTER_CUBIC)
cv2.imshow("Scaled Image using Built-in Functions (scale=0.5)", scaled_img_builtin3)

cv2.waitKey(0)
cv2.destroyAllWindows()