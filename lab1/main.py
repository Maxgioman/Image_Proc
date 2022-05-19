import matplotlib.pyplot as plt
import numpy as np
import math
from copy import deepcopy
import cv2


def plot_img(input_image, output_image):
    """
    Converts an image from BGR to RGB and plots
    """
    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
    ax[0].set_title('Basic Image')
    ax[0].axis('off')
    ax[1].imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    ax[1].set_title('Gaussian Blurred')
    ax[1].axis('off')
    plt.show()


def kernel(size, k, sigma):
    kernel_matrix = np.zeros((size, size), np.float32)
    for i in range(size):
        for j in range(size):
            norm = math.pow(i - k, 2) + math.pow(j - k, 2)
            kernel_matrix[i, j] = math.exp(-norm / (2 * math.pow(sigma, 2))) / 2 * math.pi * pow(sigma, 2)
    return kernel_matrix / np.sum(kernel_matrix)


def Filter(img, kernel):
    ih, iw = img.shape
    kh, kw = kernel.shape
    for i in range(int(kh / 2), ih - int(kh / 2)):
        for j in range(int(kh / 2), iw - int(kh / 2)):
            suma = 0
            for m in range(0, kh):
                for l in range(0, kh):
                    suma += img[i - int(kh / 2) + m, j - int(kh / 2) + l] * kernel[m, l]
            img[i, j] = suma
    return img


def result(img, kernel):
    gauss_B = Filter(img[:, :, 0], kernel)
    gauss_G = Filter(img[:, :, 1], kernel)
    gauss_R = Filter(img[:, :, 2], kernel)
    return np.dstack([gauss_B, gauss_G, gauss_R])


img1 = cv2.imread('../lab2/images/high_contrast.jpg')

k = 5
size = 2 * k + 1
sigma = 2

gauss_kernel = kernel(size, k, sigma)

img1_blur = result(img1, gauss_kernel)

plot_img(img1, img1_blur)

cv2.imwrite('images/high_contrast_k'+str(k)+'sigma'+str(sigma)+'.jpg', img1_blur)
