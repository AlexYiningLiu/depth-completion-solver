import enum
import os

from cv2 import IMREAD_GRAYSCALE
import cv2
import sys
import numpy as np
from matplotlib import pyplot as plt
import h5py
import imageio

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import utils
import math


IMAGE_DIRECTORY = '/home/alex/Projects/TRAIL/datasets/ExampleData'
CUSTOM_DEPTH_IMAGE = 'zigzag_depth_input.png'
INPUT_DEPTH_IMAGE = 'input-depth.png'
CUSTOM_NORMAL = 'computed_normals.h5'
INPUT_NORMAL = 'normals.h5'
DEPTH_UNIT_ZIGZAG = 32
DEPTH_SCALE_INPUT = 4000
VISUALIZATION_SCALE = 125


def select_roi(path):
    depth_image = cv2.imread(path)
    roi = cv2.selectROI("Depth Image ROI", depth_image, showCrosshair=False)
    xl, yl, w, h = roi
    xh = xl + w
    yh = yl + h
    print("Selected ROI:", xl, yl, xh, yh)
    convert_roi_to_bottom_left(depth_image.shape[0], xl, yl, xh, yh)


def convert_roi_to_bottom_left(height, xl, yl, xh, yh):
    new_yl = height - yh - 1
    new_yh = height - yl - 1
    print("ROI in depth2depth:", xl, new_yl, xh, new_yh)


def get_image_statistics(name, path, scale):
    print("Stats for {}".format(name) + "-"*50)

    depth_image = cv2.imread(path, cv2.IMREAD_ANYDEPTH) / scale
    print("Shape: {}, Type: {}".format(depth_image.shape, depth_image.dtype))

    print("image min = {}, max = {}".format(np.min(depth_image[np.nonzero(depth_image)]), np.max(depth_image)))

    print("Contains NAN:", np.isnan(depth_image).any())

    # print("L1 Norm:", np.linalg.norm(depth_image, ord=1))
    # print("L1 Norm:", np.sum(depth_image))
    # print("L2 Norm:", np.linalg.norm(depth_image, ord=2))
    # xl, yl, xh, yh = 566, 504, 766, 698
    # cropped_depth_img = depth_image[yl:yh+1, xl:xh+1]
    # print(cropped_depth_img.reshape(-1)[:200])

    # sublist = [0.39, 0.39025, 0.39075, 0.39075, 0.39075, 0.39075, 0.39075, 0.391, 0.391, 0.39125]
    # x = depth_image.flatten()
    # for i in range(0, 200):
    #     print("{}: {}".format(i, x[i]))
    # print(depth_image[1024-230-1, :201])
    # sublist_indices = [i for i in range(len(x)) if list(x[i:i+len(sublist)]) == sublist]
    # print(x[867943:867943+len(sublist)])
    # cv2.imshow("cropped_depth_img", cropped_depth_img)
    # cv2.waitKey(0)

    print()


def compare_normals(name, path):
    print("Stats for {}".format(name) + "-"*50)

    with h5py.File(path, 'r') as hf:
        normals = np.array(hf.get('/result'))

    _, height, width = normals.shape

    # Check basic stats
    print("Shape: {}, Type: {}".format(normals.shape, normals.dtype))
    # print("Normals[0] min = {}, max = {}".format(np.min(normals[0,...]), np.max(normals[0,...])))
    print("Contains NAN:", np.isnan(normals).any())

    total_pixels = 0
    y = 130
    for x in range(100):
        print(x, normals[0, y, x], normals[1, y, x], normals[2, y, x])

    # ny_channel = normals[1, ...]
    # nz_channel = normals[2, ...]
    # print(ny_channel.shape)
    # print("Negative or zero ny elements: {} out of {}".format(np.count_nonzero(ny_channel <= 0), ny_channel.size))  
    # print("Negative or zero nz elements: {} out of {}".format(np.count_nonzero(nz_channel <= 0), nz_channel.size))

    # Check NAN percentage and whether vectors are unit
    # normalized = np.apply_along_axis(np.linalg.norm, 0, normals)
    # nan_array = np.isnan(normalized)
    # print("{} NAN elements in {} total elements, {} percent".format(np.count_nonzero(nan_array), normalized.size, np.count_nonzero(nan_array)/normalized.size*100))
    # non_nan_array = ~nan_array
    # normalized = normalized[non_nan_array]
    # print("All non NAN vectors are unit:", (np.isclose(normalized, 1.0)).all())

    # Check norms
    # print("L1 Norm:", np.linalg.norm(normals[0,...], ord=1))
    # print("L1 Norm:", np.sum(normals[2,...]))
    # print("L2 Norm:", np.linalg.norm(normals[0,...], ord=2))

    print()


def convert_ROBI_images(path, name):
    depth_image = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
    depth_image = np.true_divide(depth_image, 1000.0)  # convert to meters
    depth_image = np.multiply(depth_image, VISUALIZATION_SCALE)  # apply visualization scale to preserve precision
    depth_image = depth_image.astype(np.uint16)
    imageio.imwrite(os.path.join(os.getcwd(), IMAGE_DIRECTORY, name), depth_image)


if __name__ == "__main__":
    convert_ROBI_images(os.path.join(os.getcwd(), IMAGE_DIRECTORY, "view_71_raw.png"), "converted_input_view_71.png")
    convert_ROBI_images(os.path.join(os.getcwd(), IMAGE_DIRECTORY, "view_71_GT.png"), "converted_GT_view_71.png")

    # get_image_statistics("input-depth-image", os.path.join(os.getcwd(), IMAGE_DIRECTORY, INPUT_DEPTH_IMAGE), DEPTH_SCALE_INPUT)
    # get_image_statistics("zigzag-depth-input", os.path.join(os.getcwd(), IMAGE_DIRECTORY, "converted_depth_input.png"), DEPTH_UNIT_ZIGZAG * VISUALIZATION_SCALE)
    
    # get_image_statistics("input-depth-output", os.path.join(os.getcwd(), IMAGE_DIRECTORY, "output-depth.png"), DEPTH_SCALE_INPUT)
    # get_image_statistics("zigzag-depth-output", os.path.join(os.getcwd(), IMAGE_DIRECTORY, "zigzag_depth_output.png"), DEPTH_UNIT_ZIGZAG * VISUALIZATION_SCALE)

    # compare_normals("input-normals", os.path.join(os.getcwd(), IMAGE_DIRECTORY, INPUT_NORMAL))
    # compare_normals("zigzag-normals", os.path.join(os.getcwd(), IMAGE_DIRECTORY, CUSTOM_NORMAL))

    # select_roi(os.path.join(os.getcwd(), IMAGE_DIRECTORY, "converted_depth_input-rgb.jpg"))
    # select_roi(os.path.join(os.getcwd(), IMAGE_DIRECTORY, "zigzag_depth_output.png"))
