import cv2
import numpy as np
from matplotlib import pyplot as plt
import h5py
import imageio

import os
import sys
import argparse

def select_roi(path, out_dir):
    depth_image = cv2.imread(path)
    roi = cv2.selectROI("Depth Image ROI", depth_image, showCrosshair=False)
    xl, yl, w, h = roi
    xh = xl + w
    yh = yl + h

    convert_roi_to_bottom_left(depth_image.shape[0], xl, yl, xh, yh)
    cropped_image = depth_image[yl:yh, xl:xh]
    cropped_filename = os.path.basename(path).replace('.png', '_cropped.png')
    cv2.imwrite(out_dir + cropped_filename, cropped_image)
    print("Cropped image file successfully written at:", out_dir + cropped_filename)


def convert_roi_to_bottom_left(height, xl, yl, xh, yh):
    new_yl = height - yh - 1
    new_yh = height - yl - 1
    print()
    print("ROI for depth2depth:", xl, new_yl, xh, new_yh)
    print()


def get_image_statistics(name, path, scale):
    print("Stats for {}".format(name) + "-"*50)
    depth_image = cv2.imread(path, cv2.IMREAD_ANYDEPTH) / scale
    print("Shape: {}, Type: {}".format(depth_image.shape, depth_image.dtype))
    print("image min = {}, max = {}".format(np.min(depth_image[np.nonzero(depth_image)]), np.max(depth_image)))
    print("Contains NAN:", np.isnan(depth_image).any())
    print()


def compare_normals(name, path):
    print("Stats for {}".format(name) + "-"*50)

    with h5py.File(path, 'r') as hf:
        normals = np.array(hf.get('/result'))

    # Check basic stats
    print("Shape: {}, Type: {}".format(normals.shape, normals.dtype))
    print("Contains NAN:", np.isnan(normals).any())

    # Check NAN percentage and whether vectors are unit
    # normalized = np.apply_along_axis(np.linalg.norm, 0, normals)
    # nan_array = np.isnan(normalized)
    # print("{} NAN elements in {} total elements, {} percent".format(np.count_nonzero(nan_array), normalized.size, np.count_nonzero(nan_array)/normalized.size*100))
    # non_nan_array = ~nan_array
    # normalized = normalized[non_nan_array]
    # print("All non NAN vectors are unit:", (np.isclose(normalized, 1.0)).all())

    print()


def convert_ROBI_images(in_path, out_path, vis_scale):
    depth_image = cv2.imread(in_path, cv2.IMREAD_ANYDEPTH)
    depth_image = np.true_divide(depth_image, 1000.)  # convert to meters
    depth_image = np.multiply(depth_image, vis_scale)  # apply visualization scale to preserve precision
    depth_image = depth_image.astype(np.uint16)
    imageio.imwrite(out_path, depth_image)
    print("Wrote converted image to:", out_path)


if __name__ == "__main__":
    IMAGE_DIRECTORY = '/home/alex/Projects/TRAIL/depth-completion-solver/input_data/'
    OUTPUT_DIRECTORY = '/home/alex/Projects/TRAIL/depth-completion-solver/temporary_data/'
    VISUALIZATION_SCALE = 125.  # chosen because 125 * 32 = 4000 which was the scale used by original depth2depth sample inputs

    parser = argparse.ArgumentParser(description='Data processing tools')
    parser.add_argument('--convert', action='store_true', help='Convert ROBI depth images to form usable by depth2depth')
    parser.add_argument('--roi', action='store_true', help='Select ROI to use for depth2depth computation')
    parser.add_argument('--png_raw', required=False, type=str, default='view_71_raw.png', help='Name of raw depth .png file')
    parser.add_argument('--png_GT', required=False, type=str, default='view_71_GT.png', help='Name of GT depth .png file')
    args = parser.parse_args()

    if args.convert:
        convert_ROBI_images(
            in_path=IMAGE_DIRECTORY + args.png_raw,
            out_path=OUTPUT_DIRECTORY + 'converted_' + args.png_raw, 
            vis_scale=VISUALIZATION_SCALE
        )
        convert_ROBI_images(
            in_path=IMAGE_DIRECTORY + args.png_GT,
            out_path=OUTPUT_DIRECTORY + 'converted_' + args.png_GT, 
            vis_scale=VISUALIZATION_SCALE
        )

    if args.roi:
        select_roi(IMAGE_DIRECTORY + args.png_raw, OUTPUT_DIRECTORY)
