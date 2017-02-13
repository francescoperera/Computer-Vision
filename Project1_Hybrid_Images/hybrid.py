import sys
sys.path.append('/Users/kb/bin/opencv-3.1.0/build/lib/')

import cv2
import numpy as np

def cross_correlation_2d(img, kernel):
    '''Given a kernel of arbitrary m x n dimensions, with both m and n being
    odd, compute the cross correlation of the given image with the given
    kernel, such that the output is of the same dimensions as the image and that
    you assume the pixels out of the bounds of the image to be zero. Note that
    you need to apply the kernel to each channel separately, if the given image
    is an RGB image.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    output = np.zeros(img.shape)
    # if len(img.shape) == 3:
    #     w,h,k = img.shape
    # else:
    #     w,h = img.shape
    w = img.shape[0]
    h = img.shape[1]

    for x in range(w):
        for y in range(h):
            window = get_window(img,x,y,kernel)
            new_pixel = calculate_new_pixel(window,kernel)
            output[x][y] = new_pixel
    return output


def get_window(im,i,j,kern):
    '''
    Inputs:
        arr: image array
        i : current row index in arr
        j: current column index in arr
        kern: kernel array

    Output:
        window centered at i,j with the same dimensions as k.
    '''
    kx,ky = kern.shape
    x1 = i - kx/2
    x2 = i + kx/2
    y1 = j - ky/2
    y2 = j + ky/2

    window = []
    for a in range(x1,x2+1):
        w = []
        for b in range(y1,y2+1):
            if a < 0 or b < 0 or a > im.shape[0]-1 or b > im.shape[1]-1:
                if len(im.shape) == 3:
                    w.append([0,0,0])
                else:
                    w.append(0)
            else:
                w.append(im[a][b])
        window.append(w)
    return np.array(window)

def calculate_new_pixel(w,k):
    new_pixel = None
    if len(w.shape) == 3:
        pix = [0,0,0]
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                for color in range(w.shape[2]):
                    pix[color] += w[i][j][color] * k[i][j]
        new_pixel = pix
    else:
        new_pixel = np.sum(w * k)
    return new_pixel

def convolve_2d(img, kernel):
    '''Use cross_correlation_2d() to carry out a 2D convolution.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    return cross_correlation_2d(img,flip_kernel(kernel))

def flip_kernel(k):
    return np.flipud(np.fliplr(k))

def gaussian_blur_kernel_2d(sigma, width, height):
    '''Return a Gaussian blur kernel of the given dimensions and with the given
    sigma. Note that width and height are different.

    Input:
        sigma:  The parameter that controls the radius of the Gaussian blur.
                Note that, in our case, it is a circular Gaussian (symmetric
                across height and width).
        width:  The width of the kernel.
        height: The height of the kernel.

    Output:
        Return a kernel of dimensions width x height such that convolving it
        with an image results in a Gaussian-blurred image.
    '''
    w = (width - 1)/2.0
    h = (height - 1)/2.0
    x,y = np.ogrid[-w:w+1,-h:h+1]
    g = gaussian(x,y,sigma)
    return g/g.sum()


def gaussian(x,y,sigma):
    return (np.exp(-(x ** 2 + y** 2) / (2 * sigma ** 2)))/ (2 * np.pi * sigma ** 2)

def low_pass(img, sigma, size):
    '''Filter the image as if its filtered with a low pass filter of the given
    sigma and a square kernel of the given size. A low pass filter supresses
    the higher frequency components (finer details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''



def high_pass(img, sigma, size):
    '''Filter the image as if its filtered with a high pass filter of the given
    sigma and a square kernel of the given size. A high pass filter suppresses
    the lower frequency components (coarse details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''


def create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2,
        high_low2, mixin_ratio):
    '''This function adds two images to create a hybrid image, based on
    parameters specified by the user.'''
    high_low1 = high_low1.lower()
    high_low2 = high_low2.lower()

    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

    if high_low1 == 'low':
        img1 = low_pass(img1, sigma1, size1)
    else:
        img1 = high_pass(img1, sigma1, size1)

    if high_low2 == 'low':
        img2 = low_pass(img2, sigma2, size2)
    else:
        img2 = high_pass(img2, sigma2, size2)

    img1 *= 2 * (1 - mixin_ratio)
    img2 *= 2 * mixin_ratio
    hybrid_img = (img1 + img2)
    return (hybrid_img * 255).clip(0, 255).astype(np.uint8)
