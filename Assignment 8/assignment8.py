"""
    CS20B1097 HIMANSHU
    
    Create a concentric square image, with inner square pixel intensity values as 125 and outer
    square as 0 everywhere. Find the projection along the row, column, diagonal with 45 degree
    and 135 degree. Reconstruct the image by applying back projection algorithm using the
    following ways:
    1. the row projection only
    2. the row and column projections
    3. the row and column and diagonal 45-degree projections
    4. the row and column and diagonal 45-degree and 135-degree projections
"""

import numpy as np
import cv2

def normalize_image(img):
    img = img - np.min(img)
    img = img / np.max(img)
    img = img * 255
    return img.astype(np.uint8)

n = 256
img = np.zeros((n, n))

inner_sq_size=int(n/2)
start=int(n/2-int(inner_sq_size/2))
end=int(n/2+int(inner_sq_size/2))

img[start:end,start:end]=125

#row projection
row_proj=np.sum(img,axis=1)

#column projection
col_proj=np.sum(img,axis=0)


# diagoonal projection at 45
p_45=np.zeros(2*n-1)
d_45=[]
for i in range(n):
    diagonal = []
    for j in range(i+1):
        diagonal.append(img[j][i-j])
    d_45.append(diagonal)

for i in range(1,n):
    diagonal = []
    for j in range(n-i):
        diagonal.append(img[i+j][n-j-1])
    d_45.append(diagonal)

for i in range(2*n-1):
    p_45[i]=sum(d_45[i])

# diagonal projection at 135
p_135=np.zeros(2*n-1)
d_135=[]
for i in range(n):
    diagonal = []
    for j in range(i+1):
       diagonal.append(img[n-j-1][n-i+j-1])
    d_135.append(diagonal)
for i in range(1,n):
    diagonal = []
    for j in range(n-i):
       diagonal.append(img[j][i+j])
    d_135.append(diagonal)
for i in range(2*n-1):
    p_135[i]=sum(d_135[i])

#reconstruction using row projection
img_row=np.zeros((n,n))
avg_intensity=row_proj/n
for i in range(n):
    img_row[i,:]=avg_intensity[i]

img_row=normalize_image(img_row)

#reconstruction using row and column projection
img_row_col=np.zeros((n,n))
for i in range(n):
    img_row_col[i,:]=row_proj[i]

for i in range(n):
    img_row_col[:,i]+=col_proj[i]

img_row_col=normalize_image(img_row_col)

#reconstruction using row and column and diagonal projection at 45 degree
r_c_45 = np.zeros((n,n))
for i in range(n):
    for j in range(n):
        d = i+j
        r_c_45[i,j] = row_proj[i] + col_proj[j] + p_45[d]
        
r_c_45 = normalize_image(r_c_45)

# reconstruction using row, column, diagonal projection at 45 degree, and diagonal 135 degree
r_c_45_135 = np.zeros((n,n))
for i in range(n):
    for j in range(n):
        d1 = i+j
        d2 = n-1-(i-j)
        r_c_45_135[i,j] = row_proj[i] + col_proj[j] + p_45[d1] + p_135[d2]

r_c_45_135 = normalize_image(r_c_45_135)

cv2.imshow("Row Projection Reconstruction",img_row.astype(np.uint8))
cv2.imshow("Row and Column Projection Reconstruction",img_row_col.astype(np.uint8))
cv2.imshow("45-Degree, Row and Column Projection Reconstruction",r_c_45.astype(np.uint8))
cv2.imshow("135-Degree and 45-Degree, Row and Column Projection Reconstruction",r_c_45_135.astype(np.uint8))
cv2.imshow("concentric sq",img)
cv2.waitKey(0)
cv2.destroyAllWindows()