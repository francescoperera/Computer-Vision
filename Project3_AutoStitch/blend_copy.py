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
    c1 = np.array([0,0,1]) #corner 1
    c2 = np.array([img.shape[0]-1,0,1]) #corner 2
    c3 = np.array([0,img.shape[1]-1,1]) #corner 3
    c4 = np.array([img.shape[0]-1 ,img.shape[1]-1,1]) #corner 4

    c1_tr = np.dot(M,c1)#np.array of transformed c1 coordinates
    c1_tr_hm = np.array([c1_tr[0] / float(c1_tr[2]), c1_tr[1] / float(c1_tr[2])])
    c2_tr = np.dot(M,c2)
    c2_tr_hm = np.array([c2_tr[0] / float(c2_tr[2]), c2_tr[1] / float(c2_tr[2])])
    c3_tr = np.dot(M,c3)
    c3_tr_hm = np.array([c3_tr[0] / float(c3_tr[2]),c3_tr[1] / float(c3_tr[2])])
    c4_tr = np.dot(M,c4)
    c4_tr_hm = np.array([c4_tr[0] / float(c4_tr[2]),c4_tr[1] / float(c4_tr[2])])
    c_tr_list = [c1_tr_hm,c2_tr_hm,c3_tr_hm,c4_tr_hm] #c_tr_list = list transformed coords.
    x_c_tr = [c[0] for c in c_tr_list] #x_c_tr = list all x coords in c_tr_list
    y_c_tr = [c[1] for c in c_tr_list] #y_c_tr = list all y coords in c_tr_list

    # print
    # print "LIST"
    # print x_c_tr
    # print y_c_tr
    # print "LIST"


    minX = min(x_c_tr)
    minY = min(y_c_tr)
    maxX = max(x_c_tr)
    maxY = max(y_c_tr)

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
    print acc.shape
    print img.shape
    print acc[0][0][0]
    print img[0][0][0]
    # BEGIN TODO 10
    minX,minY,maxX,maxY = imageBoundingBox(img,M)
    img_width = img.shape[1]
    img_height = img.shape[0]
    M_inv = np.linalg.inv(M)
    for x in range(minX,maxX):
        for y in range(minY,maxY):

            # check that the coords x,y are within the src ( acc)
            if (x < 0 or x >= acc.shape[0] or y < 0 or y >= acc.shape[1]):
                continue



            p_dest = np.array([x,y,1]) # 3d array
            p_source = np.dot(M_inv,p_dest)

            #getting x,y coords. in img_loc as homogenous coords.
            img_x = p_source[0] / float(p_source[2]) # float
            img_y = p_source[1] / float(p_source[2]) # float
            #img _x = x coord of acc
            #img_y = y coord of acc

            #make sure tham img_x,img_y are within the bounds of img
            if (img_x < 0 or img_x >= img_height\
                or img_y < 0 or img_y >= img_width):
                continue

            #calculate coords of 4 pixel neighbors
            xf = int(math.floor(img_x)) #xf = x floor
            xc = xf + 1 #xc = x ceiling
            yf = int(math.floor(img_y)) #yf = y floor
            yc = yf + 1 #yc = y ceiling

            #check that neighbors are within the img bounds.
            if xc < 0:
                xc = 0
            elif xc >= img_height:
                xc = img_height - 1

            if xf < 0:
                xf = 0
            elif xf >= img_height:
                xf = img_height - 1

            if yc < 0:
                yc = 0
            elif yc >= img_width:
                yc = img_width - 1

            if yf < 0:
                yf = 0
            elif yf >= img_width:
                yf = img_width - 1

            #check that neighbors are not black pixels
            if np.array_equal(img[xc][yc],np.array([0,0,0])):
                continue
            if np.array_equal(img[xc][yf],np.array([0,0,0])):
                continue
            if np.array_equal(img[xf][yc],np.array([0,0,0])):
                continue
            if np.array_equal(img[xf][yf],np.array([0,0,0])):
                continue

            Q_11 = img[xf, yf]
            Q_12 = img[xf, yc]
            Q_21 = img[xc, yf]
            Q_22 = img[xc, yc]

            # fQ11 = img[y0,x0]
            # fQ12 = img[y1,x0]
            # fQ21 = img[y0,x1]
            # fQ22 = img[y1,x1]

            #Bilinear interpolation
            #Do checks to avoid dividing by zero
            if(xf == xc):
                fxy1 = Q_11
                fxy2 = Q_22
            else:
                fxy1 = np.dot((xf-img_x)/(xf-xc),Q_11) + np.dot((img_x-xc)/(xf-xc),Q_21)
                fxy2 = np.dot((xf-img_x)/(xf-xc),Q_12) + np.dot((img_x-xc)/(xf-xc),Q_22)

            if(yf == yc):
                fxy = fxy1
            else:
                fxy = np.dot((yf-img_y)/(yf-yc),fxy1) + np.dot((img_y-yc)/(yf-yc),fxy2)

            weight = 1.0

            if (img_x < blendWidth):
                weight = img_x / blendWidth
            elif(img_x > img_width - blendWidth):
                weight = (img_width - img_x) / blendWidth

            if (img_y < blendWidth):
                weight *= (img_y / blendWidth)
            elif(img_y > img_height - blendWidth):
                weight *= (img_height - img_y) / blendWidth

            #print img_x,img_y
            #print img[img_x][img_y][0]
            #print acc[row][col][0]
            acc[x][y][0] += float(weight * fxy[0])
            acc[x][y][1] += float(weight * fxy[1])
            acc[x][y][2] += float(weight * fxy[2])
            acc[x][y][3] += float(weight)
            #print acc[x][y]
    # END TODO


def normalizeBlend(acc):
    """
       INPUT:
         acc: input image whose alpha channel (4th channel) contains
         normalizing weight values
       OUTPUT:
         img: image with r,g,b values of acc normalized
    """
    for row in range(acc.shape[0]):
        for col in range(acc.shape[1]):
            arr = acc[row][col]
            if arr[3] == 0:
                arr[3] = 255
                newArr = arr.astype(int)
            else:
                newArr = np.array([float(arr[0])/arr[3],
                                float(arr[1])/arr[3],
                                float(arr[2])/arr[3],
                                255],
                                dtype = int
                                )
            #newArr.astype(np.int8)
            #print newArr.dtype
            print newArr
            acc[row][col] = newArr
    img = acc
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
        local_minX,local_minY,local_maxX,local_maxY = imageBoundingBox(img,M)

        minX = min(local_minX,minX)
        minY = min(local_minY,minY)
        maxX = max(local_maxX,maxX)
        maxY = max(local_maxY,maxY)
        # END TODO

    # Create an accumulator image
    accWidth = int(math.ceil(maxX) - math.floor(minX))
    accHeight = int(math.ceil(maxY) - math.floor(minY))
    print 'accWidth, accHeight:', (accWidth, accHeight)
    translation = np.array([[1, 0, -minX], [0, 1, -minY], [0, 0, 1]])

    return accWidth, accHeight, channels, width, translation


def pasteImages(ipv, translation, blendWidth, accWidth, accHeight, channels):

    acc = np.zeros((accHeight, accWidth, channels + 1))
    print "paste"
    print acc.dtype
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
    print "acc"
    print acc[:10,:40,:]
    acc = acc.astype(int)
    print acc.dtype
    compImage = normalizeBlend(acc)
    print "compImage"
    print(compImage.dtype)

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
    print "croppedImage"
    croppedImage.dtype

    return croppedImage
