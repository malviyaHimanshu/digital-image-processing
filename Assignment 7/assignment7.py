"""
    CS20B1097 HIMANSHU

    1. Swap phase of the dog image and magnitude of the Lena image and display the output.
    2. Swap phase of the Lena image and magnitude of the dog image and display the output
    Solve 1 & 2 using built-in function (4 marks) and user defined function (6 marks).
"""

import cv2 
import numpy as np

# Write a function to implement FFT for 1D signal.
def fft_1d(signal):
    n = len(signal)
    F = np.zeros(n, dtype=complex)
    for u in range(n):
        for x in range(n):
            F[u] = F[u] + (signal[x]*(round(np.cos(2*np.pi*u*x/n), 6) - 1j*round(np.sin(2*np.pi*u*x/n), 6)))
    return F

# Write a function to implement FFT for 2D signal.
def fft_2d(signal):
    m = len(signal)
    n = len(signal[0])
    F = np.zeros((m, n), dtype=complex)

    temp = []
    for u in range(m):
        row_signal = signal[u, :]
        temp.append(fft_1d(row_signal))

    final = []
    for v in range(n):
        col_signal = [i[v] for i in temp]
        final.append(fft_1d(col_signal))
    F = np.transpose(final)
    return F

# For calculating Phase
def find_angle(complex_arr):
    phase = []
    for i in range(len(complex_arr)):
        temp = []
        for z in complex_arr[i]:
            if z.imag == 0:
                temp.append(0)
            elif z.real == 0:
                # 90 degree = 1.5707963267948966 radian
                temp.append((abs(z.imag)/z.imag) * np.pi/2)
            else:
                temp.append(np.arctan(z.imag/z.real))
        phase.append(temp)
    return np.array(phase, dtype=np.float32)

# Find phase and magnitude of the dog and Lena images using the DFT function.
dog = cv2.imread('dog.jpg', 0)
lena = cv2.imread('lena.png', 0)

# Fourier Transform 
dog_fft = fft_2d(dog)
lena_fft = fft_2d(lena)

# Swap phase of the dog image and magnitude of the Lena image and display the output.
dog_phase = np.angle(dog_fft)
lena_phase = np.angle(lena_fft)

dog_mag = np.abs(dog_fft)
lena_mag = np.abs(lena_fft)

dog_new = lena_mag * np.exp(1j * dog_phase)
lena_new = dog_mag * np.exp(1j * lena_phase)

dog_new = np.abs(np.fft.ifft2(dog_new))
lena_new = np.abs(np.fft.ifft2(lena_new))

cv2.imshow('dog_new', dog_new.astype(np.uint8))
cv2.imshow('lena_new', lena_new.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()