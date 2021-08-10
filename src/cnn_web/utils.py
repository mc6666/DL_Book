import base64

import cv2
import numpy as np


def data_uri_to_cv2_img(uri):
    """
    Convert a data URL to an OpenCV image
    Credit: https://stackoverflow.com/a/54205640/2415512

    : param uri : data URI representing a BW image
    : returns   : OpenCV image
    """

    encoded_data = uri.split(',')[1]
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    return img


def value_invert(array):
    """
    Takes an array, assumed to contain values between 0 and 1, and inverts
    those values with the transformation x -> 1 - x.
    """

    # Flatten the array for looping
    flatarray = array.flatten()

    # Apply transformation to flattened array
    for i in range(flatarray.size):
        flatarray[i] = 1 - flatarray[i]

    # Return the transformed array, with the original shape
    return flatarray.reshape(array.shape)
