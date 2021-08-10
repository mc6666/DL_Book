import math

import cv2
import numpy as np
from scipy import ndimage


def preprocess(img):
    """
    Preprocess incoming images in the same way that
    images in the MNIST dataset were processed.
    This code was adapted from
    http://opensourc.es/blog/tensorflow-mnist

    : param img : input black-and-white image
    : returns   : processed image
    """

    # Crop out completely white edges
    while int(np.mean(img[0])) == 255:
        img = img[1:]

    while np.mean(img[:, 0]) == 255:
        img = np.delete(img, 0, 1)

    while np.mean(img[-1]) == 255:
        img = img[:-1]

    while np.mean(img[:, -1]) == 255:
        img = np.delete(img, -1, 1)

    # Resize to the proper shape
    rows, cols = img.shape

    if rows > cols:
        factor = 20.0/rows
        rows = 20
        cols = int(round(cols*factor))
        img = cv2.resize(img, (cols, rows))
    else:
        factor = 20.0/cols
        cols = 20
        rows = int(round(rows*factor))
        img = cv2.resize(img, (cols, rows))

    # Add predefined padding
    colsPadding = (int(math.ceil((28-cols)/2.0)),
                   int(math.floor((28-cols)/2.0)))
    rowsPadding = (int(math.ceil((28-rows)/2.0)),
                   int(math.floor((28-rows)/2.0)))
    img = np.lib.pad(img, (rowsPadding, colsPadding),
                     'constant', constant_values=255)

    # Shift image so that the digit's center of mass
    # is nicely centered
    shiftx, shifty = getBestShift(img)
    shifted = shift(img, shiftx, shifty)
    img = shifted

    return img


def getBestShift(img):
    """
    Calculate how to shift an image of a digit so that its
    center of mass is nicely centered.

    : param img : black and white image of a digit
    : returns   : optimal shifts (x, y)
    """

    # Calculate center of mass
    cy, cx = ndimage.measurements.center_of_mass(img)

    # Calculate difference between center of mass and actual
    # image center to get shifts
    rows, cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)

    return shiftx, shifty


def shift(img, sx, sy):
    """
    Shift an image by some offsets

    : param img : black and white image
    : param sx  : shift in x-direction
    : param sy  : shift in y-direction
    : returns   : shifted image
    """

    # Generate warping matrix representing translation only (rotation)
    # part is unit matrix
    M = np.float32([[1, 0, sx], [0, 1, sy]])

    # Apply warping matrix
    rows, cols = img.shape
    shifted = cv2.warpAffine(img, M, (cols, rows),
                             borderMode=cv2.BORDER_CONSTANT, borderValue=255)

    return shifted
