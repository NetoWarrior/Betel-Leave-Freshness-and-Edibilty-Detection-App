import os
import argparse
import numpy as np
import cv2

from utils import *
from background_marker import *


def generate_background_marker(file):
    """
    Generate background marker for an image

    Args:
        file (string): full path of an image file

    Returns:
        tuple[0] (ndarray of an image): original image
        tuple[1] (ndarray size of an image): background marker
    """

    # check file name validity
    if not os.path.isfile(file):
        raise ValueError('{}: is not a file'.format(file))

    original_image = read_image(file)

    marker = np.full((original_image.shape[0], original_image.shape[1]), True)

    # update marker based on vegetation color index technique
    color_index_marker(index_diff(original_image), marker)

    # update marker to remove blues
    # remove_blues(original_image, marker)

    return original_image, marker


def segment_leaf(image_file, filling_mode, smooth_boundary, marker_intensity):
    """
    Segments leaf from an image file

    Args:
        image_file (string): full path of an image file
        filling_mode (string {no, flood, threshold, morph}): 
            how holes should be filled in segmented leaf
        smooth_boundary (boolean): should leaf boundary smoothed or not
        marker_intensity (int in rgb_range): should output background marker based
                                             on this intensity value as foreground value

    Returns:
        tuple[0] (ndarray): original image to be segmented
        tuple[1] (ndarray): A mask to indicate where leaf is in the image
                            or the segmented image based on marker_intensity value
    """
    # get background marker and original image
    original, marker = generate_background_marker(image_file)

    # set up binary image for futher processing
    bin_image = np.zeros((original.shape[0], original.shape[1]))
    bin_image[marker] = 255
    bin_image = bin_image.astype(np.uint8)

    # further processing of image, filling holes, smoothing edges
    largest_mask = \
        select_largest_obj(bin_image, fill_mode=filling_mode,
                           smooth_boundary=smooth_boundary)

    if marker_intensity > 0:
        largest_mask[largest_mask != 0] = marker_intensity
        image = largest_mask
    else:
        # apply marker to original image
        image = original.copy()
        image[largest_mask == 0] = np.array([0, 0, 0])

    return image


def rgb_range(arg):
    """
    Check if arg is in range for rgb value(between 0 and 255)

    Args:
        arg (int convertible): value to be checked for validity of range

    Returns:
        arg in int form if valid

    Raises:
        argparse.ArgumentTypeError: if value can not be integer or not in valid range
    """

    try:
        value = int(arg)
    except ValueError as err:
       raise argparse.ArgumentTypeError(str(err))

    if value < 0 or value > 255:
        message = "Expected 0 <= value <= 255, got value = {}".format(value)
        raise argparse.ArgumentTypeError(message)

    return value


output_image = segment_leaf('C:\\Users\\DELL\\Downloads\\IMG_20210331_231552.jpg','flood', True, 0)
cv2.imwrite('C:\\Users\\DELL\\Downloads\\IMG_20210331_231552.jpg',output_image,)
