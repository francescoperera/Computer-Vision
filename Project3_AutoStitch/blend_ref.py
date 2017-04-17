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
         maxX: int for the maximum X value of a corner
         maxY: int for the maximum Y value of a corner
    """
    #TODO 8
    #TODO-BLOCK-BEGIN
    # The idea here is to compute the homography of all the corners
    # and then choose the second smallest and second largest of
    # all the x's and all the y's. This should ensure that the
    # bounding box only contains image and no "dead" space
    c1 = M.dot(np.array([0.0,0.0, 1], dtype=float).T)
    c2 = M.dot(np.array([0.0, len(img), 1], dtype=float).T)
    c3 = M.dot(np.array([len(img[0]), 0.0, 1], dtype=float).T)
    c4 = M.dot(np.array([len(img[0]), len(img), 1], dtype=float).T)
    all_corners = [c1, c2, c3, c4]
    all_corners = [c[:2]/c[2] for c in all_corners]
    all_x = sorted([c[0] for c in all_corners])
    all_y = sorted([c[1] for c in all_corners])
    minX = all_x[1]
    minY = all_y[1]
    maxX = all_x[2]
    maxY = all_y[2]
    #TODO-BLOCK-END
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
    # Fill in this routine
    #TODO-BLOCK-BEGIN
    minX, minY, maxX, maxY = imageBoundingBox(img, M)

    blend_func = [i/float(blendWidth) for i in range(blendWidth)]
    M_inv = np.linalg.inv(M)

    # Pad img matrix to do linear interpolation properly
    img_padded = np.zeros((img.shape[0]+2, img.shape[1]+2, 3))
    img_padded[1:-1, 1:-1] = img

    img_padded[1:-1, 0] = img[:, 1]
    img_padded[1:-1, -1] = img[:, -2]
    img_padded[0, 1:-1] = img[1, :]
    img_padded[-1, 1:-1] = img[-2, :]

    img_padded[0,0] = img[0,0]
    img_padded[0,-1] = img[0, -1]
    img_padded[-1,0] = img[-1,0]
    img_padded[-1,-1] = img[-1,-1]

    for r in range(minY, maxY):
        for c in range(minX, maxX):
            # Reverse warp back to img point
            pt = M_inv.dot(np.array([c,r,1.0]).T)
            # Convert from homogeneous to cartesian coordinates
            pt = pt[0:2]/float(pt[2])
            ptx = pt[0]
            pty = pt[1]
            pt_rgb = np.zeros(3)
            color_set = False
            # Only use interpolation if necessary
            if abs(np.round(ptx) - ptx) < 0.1 and abs(np.round(pty) - pty) < 0.1:
                ptx = int(np.rint(ptx))
                pty = int(np.rint(pty))
                if all(img[pty, ptx] != np.array([0,0,0])):
                    pt_rgb = img[pty, ptx]
                    color_set = True
            else:
                # Bilinear Interpolation
                int_x_min = int(np.floor(ptx))
                int_y_min = int(np.floor(pty))
                int_x_max = int(np.floor(ptx+1))
                int_y_max = int(np.floor(pty+1))

                pt11_rgb = img_padded[int_y_max+1, int_x_max+1]
                pt12_rgb = img_padded[int_y_min+1, int_x_max+1]
                pt21_rgb = img_padded[int_y_max+1, int_x_min+1]
                pt22_rgb = img_padded[int_y_min+1, int_x_min+1]

                # For any points during interpolation that are black,
                # don't use them in the final image (they don't have a good interpolation approximation)
                if all(pt11_rgb == np.array([0,0,0])) or all(pt12_rgb == np.array([0,0,0]))\
                        or all(pt21_rgb == np.array([0,0,0])) or all(pt22_rgb == np.array([0,0,0])):
                    continue

                x_mat = np.array([int_x_max - ptx, ptx - int_x_min])
                y_mat = np.array([int_y_max - pty, pty - int_y_min]).T
                for i in range(3):
                    f_mat = np.array([[pt11_rgb[i], pt12_rgb[i]], [pt21_rgb[i], pt22_rgb[i]]])
                    pt_rgb[i] = x_mat.dot(f_mat.dot(y_mat))
            # Blend
            acc_rgb = acc[r, c, 0:3]
            if all(acc_rgb == np.array([0,0,0])):
                acc[r, c, 0:3] = pt_rgb
                acc[r, c, 3] = 1
            elif not all(pt_rgb == np.array([0,0,0])):
                if (maxX - c) <= blendWidth:
                    alpha = 1 - float(blend_func[blendWidth - (maxX - c)])
                    acc[r,c,0:3] = (1-alpha)*acc[r,c,3]*acc_rgb + alpha*pt_rgb
                    acc[r,c,3] = alpha + (1-alpha)*acc[r,c,3]
                elif (c - minX) < blendWidth:
                    alpha = float(blend_func[c-minX])
                    acc[r, c, 0:3] = (1 - alpha)*acc[r, c, 3]*acc[r, c, 0:3] + alpha*pt_rgb
                    acc[r, c, 3] = alpha + (1 - alpha) * acc[r, c, 3]
                else:
                    acc[r,c,0:3] = pt_rgb
                    acc[r,c,3] = 1.0
    #TODO-BLOCK-END
    # END TODO


def normalizeBlend(acc):
    """
       INPUT:
         acc: input image whose alpha channel (4th channel) contains
         normalizing weight values
       OUTPUT:
         img: image with r,g,b values of acc normalized
    """
    # BEGIN TODO 11
    # fill in this routine..
    img = np.zeros((acc.shape[0], acc.shape[1], 3), dtype=np.uint8)
    #TODO-BLOCK-BEGIN
    for r in range(len(acc)):
        for c in range(len(acc[0])):
            if acc[r,c,3] > 0:
                img[r,c, 0:3] = [int(color) for color in (acc[r,c,0:3]/acc[r,c,3])]
            else:
                img[r,c, 0:3] = [int(color) for color in (acc[r,c,0:3])]
            #img[r,c,3] = 1
    #TODO-BLOCK-END
    # END TODO
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
        # add some code here to update minX, ..., maxY
        #TODO-BLOCK-BEGIN
        minXnew, minYnew, maxXnew, maxYnew = imageBoundingBox(img, M)
        minX = np.minimum(minX, minXnew)
        minY = np.minimum(minY, minYnew)
        maxX = np.maximum(maxX, maxXnew)
        maxY = np.maximum(maxY, maxYnew)
        #TODO-BLOCK-END
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
    # Shift it left by the correct amount
    # Then handle the vertical drift
    # Note: warpPerspective does forward mapping which means A is an affine
    # transform that maps accumulator coordinates to final panorama coordinates
    #TODO-BLOCK-BEGIN

    if is360 is True:
        A = computeDrift(x_init, y_init, x_final, y_final, width)

    #TODO-BLOCK-END
    # END TODO

    if A_out is not None:
        A_out[:] = A

    # Warp and crop the composite
    croppedImage = cv2.warpPerspective(
        compImage, A, (outputWidth, accHeight), flags=cv2.INTER_LINEAR
    )

    return croppedImage
