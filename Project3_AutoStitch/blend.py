import math
import sys

import cv2
import numpy as np


class ImageInfo:
    def __init__(self, name, img, position):
        self.name = name
        self.img = img
        self.position = position


def imageBoundingBox(img, M):
    """
       This is a useful helper function that you might choose to implement
       that takes an image, and a transform, and computes the bounding box
       of the transformed image.

       INPUT:
         img: image to get the bounding box of
         M: the transformation to apply to the img
       OUTPUT:
         minX: int for the minimum X value of a corner
         minY: int for the minimum Y value of a corner
         minX: int for the maximum X value of a corner
         minY: int for the maximum Y value of a corner
    """
    #TODO 8
    c1 = np.array([0.0, 0.0, 1], dtype=float) # corner 1
    c2 = np.array([img.shape[1], 0, 1], dtype=float)  # corner 2
    c3 = np.array([0.0, img.shape[0], 1], dtype=float)  # corner 3
    c4 = np.array([img.shape[1], img.shape[0], 1], dtype=float)  # corner 4

    c1_tr = np.dot(M, c1.T)  # np.array of transformed c1 coordinates
    c1_tr_hm = np.array([c1_tr[0] / c1_tr[2], c1_tr[1] / c1_tr[2]])

    c2_tr = np.dot(M,c2.T)
    c2_tr_hm = np.array([c2_tr[0] / c2_tr[2], c2_tr[1] / c2_tr[2]])

    c3_tr = np.dot(M,c3.T)
    c3_tr_hm = np.array([c3_tr[0] / c3_tr[2],c3_tr[1] / c3_tr[2]])

    c4_tr = np.dot(M,c4.T)
    c4_tr_hm = np.array([c4_tr[0] / c4_tr[2], c4_tr[1] / c4_tr[2]])

    c_tr_list = [c1_tr_hm, c2_tr_hm, c3_tr_hm, c4_tr_hm] #c_tr_list = list transformed coords.

    x_c_tr = sorted([c[0] for c in c_tr_list]) #x_c_tr = list all x coords in c_tr_list
    y_c_tr = sorted([c[1] for c in c_tr_list]) #y_c_tr = list all y coords in c_tr_list

    minX = x_c_tr[1]
    minY = y_c_tr[1]
    maxX = x_c_tr[2]
    maxY = y_c_tr[2]

    return int(minX), int(minY), int(maxX), int(maxY)


def accumulateBlend(img, acc, M, blendWidth):
    """
       INPUT:
         img: image to add to the accumulator
         acc: portion of the accumulated image where img should be added
         M: the transformation mapping the input image to the accumulator
         blendWidth: width of blending function. horizontal hat function
       OUTPUT:
         modify acc with weighted copy of img added where the first
         three channels of acc record the weighted sum of the pixel colors
         and the fourth channel of acc records a sum of the weights
    """
    # BEGIN TODO 10
    minX, minY, maxX, maxY = imageBoundingBox(img,M)


    img_width = img.shape[1]
    img_height = img.shape[0]
    M_inv = np.linalg.inv(M)

    # Pad image so that we don't run into out of bounds errors
    img_pad = np.zeros((img_height + 2, img_width + 2, 3))
    img_pad[1: -1, 1:-1] = img
    img_pad[1: -1, 0] = img[:, 1]
    img_pad[1: -1, -1] = img[:, -2]
    img_pad[0, 1: -1] = img[1, :]
    img_pad[-1, 1: -1] = img[-2, :]
    img_pad[0, 0] = img[0, 0]
    img_pad[0, -1] = img[0, -1]
    img_pad[-1, 0] = img[-1, 0]
    img_pad[-1, -1] = img[-1, -1]

    blend_range = [i / float(blendWidth) for i in range(blendWidth)]

    # range(minX, maxX) iterates through columns
    for x in range(minY,maxY): # iterate through the rows in bounding box
        for y in range(minX,maxX): #iterate through the cols in bounding box

            p_dest = np.array([y,x,1], dtype=float) # 3d array of image
            p_source = np.dot(M_inv,p_dest)
            img_x = p_source[0] / p_source[2]
            img_y = p_source[1] / p_source[2]
            rgb_val = np.zeros(3)

            #if out of bounds, ignore
            if (img_x < 0 or img_x >= img_height or img_y < 0 \
                or img_y >=  img_width):
                continue
            # A lot of the points are actually in the image, so let's
            # set a threshold and only use interpolation for those points
            # that pass it.
            if abs(np.round(img_x) - img_x) <= 0.05 and abs(np.round(img_y) - img_y) <= 0.05:
                img_x = int(np.round(img_x))
                img_y = int(np.round(img_y))

                if not np.array_equal(img[img_y, img_x], np.array([0,0,0])):
                    rgb_val = img[img_y, img_x]
            else:
                # Bilinear interpolation
                xf = int(math.floor(img_x)) # xfloor
                xc = xf + 1  # xceiling
                yf = int(math.floor(img_y))  # yfloor
                yc = yf + 1  # yceiling

                Q_11 = img_pad[yf + 1, xf + 1]
                Q_12 = img_pad[yc + 1, xf + 1]
                Q_21 = img_pad[yf + 1, xc + 1]
                Q_22 = img_pad[yc + 1, xc + 1]

                # ignore black points:
                if np.array_equal(Q_11,np.array([0,0,0])) or np.array_equal(Q_12,np.array([0,0,0]))\
                   or np.array_equal(Q_21,np.array([0,0,0])) or np.array_equal(Q_22,np.array([0,0,0])):
                    continue

                val = (1.0 / ((xc - xf) * (yc - yf)))
                a = np.array([xc - img_x, img_x - xf])
                c = np.array([[yc - img_y], [img_y - yf]])
                for q in range(3):
                    q11, q12, q21, q22 = Q_11[q], Q_12[q], Q_21[q], Q_22[q]
                    b = np.array([[q11, q12], [q21, q22]])
                    rgb_val[q] = val * np.dot(a, np.dot(b, c))

            # Blending
            acc_coord_rgb = acc[x, y, 0:3]
            if np.array_equal(acc_coord_rgb, np.array([0,0,0])):
                acc[x, y, 0:3] = rgb_val
                acc[x, y, 3] = 1
            else:
                if maxX - y <= blendWidth:
                    alpha = 1 - float(blend_range[blendWidth - (maxX - y)])
                    acc[x, y, 0:3] = (1 - alpha) * acc[x, y, 3] * acc_coord_rgb + alpha * rgb_val
                    acc[x, y, 3] = alpha + (1 - alpha) * acc[x, y, 3]
                elif y - minX < blendWidth:
                    alpha = float(blend_range[y - minX])
                    acc[x, y, 0:3] = (1 - alpha) * acc[x, y, 3] * acc_coord_rgb + alpha * rgb_val
                    acc[x, y, 3] = alpha + (1 - alpha) * acc[x, y, 3]
                else:
                    acc[x, y, 0:3] = rgb_val
                    acc[x, y, 3] = 1
    # END TODO


def normalizeBlend(acc):
    """
       INPUT:
         acc: input image whose alpha channel (4th channel) contains
         normalizing weight values
       OUTPUT:
         img: image with r,g,b values of acc normalized
    """
    img = np.zeros((acc.shape[0], acc.shape[1], 3), dtype=np.uint8)
    for row in range(acc.shape[0]):
        for col in range(acc.shape[1]):
            if acc[row, col, 3] > 0:
                img[row, col, 0:3] = [int(i) for i in (acc[row, col, 0:3] / acc[row, col, 3]) ]
            else:
                img[row, col, 0:3] = [int(i) for i in (acc[row, col, 0:3])]
    return img


def getAccSize(ipv):
    # Compute bounding box for the mosaic
    minX = sys.maxint
    minY = sys.maxint
    maxX = 0
    maxY = 0
    channels = -1
    width = -1  # Assumes all images are the same width
    M = np.identity(3)
    for i in ipv:
        M = i.position
        img = i.img
        _, w, c = img.shape
        if channels == -1:
            channels = c
            width = w

        # BEGIN TODO 9
        local_minX, local_minY, local_maxX, local_maxY = imageBoundingBox(img,M)

        minX = min(local_minX, minX)
        minY = min(local_minY, minY)
        maxX = max(local_maxX, maxX)
        maxY = max(local_maxY, maxY)
        # END TODO

    # Create an accumulator image
    accWidth = int(math.ceil(maxX) - math.floor(minX))
    accHeight = int(math.ceil(maxY) - math.floor(minY))
    print 'accWidth, accHeight:', (accWidth, accHeight)
    translation = np.array([[1, 0, -minX], [0, 1, -minY], [0, 0, 1]])

    return accWidth, accHeight, channels, width, translation


def pasteImages(ipv, translation, blendWidth, accWidth, accHeight, channels):

    acc = np.zeros((accHeight, accWidth, channels + 1))
    # Add in all the images
    M = np.identity(3)
    for count, i in enumerate(ipv):
        M = i.position
        img = i.img
        M_trans = translation.dot(M)
        accumulateBlend(img, acc, M_trans, blendWidth)

    return acc


def getDriftParams(ipv, translation, width):
    # Add in all the images
    M = np.identity(3)
    for count, i in enumerate(ipv):
        if count != 0 and count != (len(ipv) - 1):
            continue

        M = i.position

        M_trans = translation.dot(M)

        p = np.array([0.5 * width, 0, 1])
        p = M_trans.dot(p)

        # First image
        if count == 0:
            x_init, y_init = p[:2] / p[2]
        # Last image
        if count == (len(ipv) - 1):
            x_final, y_final = p[:2] / p[2]

    return x_init, y_init, x_final, y_final


def computeDrift(x_init, y_init, x_final, y_final, width):
    A = np.identity(3)
    drift = (float)(y_final - y_init)
    # We implicitly multiply by -1 if the order of the images is swapped...
    length = (float)(x_final - x_init)
    A[0, 2] = -0.5 * width
    # Negative because positive y points downwards
    A[1, 0] = -drift / length

    return A


def blendImages(ipv, blendWidth, is360=False, A_out=None):
    """
       INPUT:
         ipv: list of input images and their relative positions in the mosaic
         blendWidth: width of the blending function
       OUTPUT:
         croppedImage: final mosaic created by blending all images and
         correcting for any vertical drift
    """
    accWidth, accHeight, channels, width, translation = getAccSize(ipv)
    acc = pasteImages(
        ipv, translation, blendWidth, accWidth, accHeight, channels
    )
    compImage = normalizeBlend(acc)

    # Determine the final image width
    outputWidth = (accWidth - width) if is360 else accWidth
    x_init, y_init, x_final, y_final = getDriftParams(ipv, translation, width)
    # Compute the affine transform
    A = np.identity(3)
    # BEGIN TODO 12
    # fill in appropriate entries in A to trim the left edge and
    # to take out the vertical drift if this is a 360 panorama
    # (i.e. is360 is true)
    if (is360):
        A = computeDrift(x_init, y_init, x_final, y_final, outputWidth)
    # Shift it left by the correct amount
    # Then handle the vertical drift
    # Note: warpPerspective does forward mapping which means A is an affine
    # transform that maps accumulator coordinates to final panorama coordinates
    # END TODO

    if A_out is not None:
        A_out[:] = A

    # Warp and crop the composite
    croppedImage = cv2.warpPerspective(
        compImage, A, (outputWidth, accHeight), flags=cv2.INTER_LINEAR
    )

    return croppedImage
