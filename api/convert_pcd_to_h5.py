from email.mime import base
import h5py
import numpy as np
from pandas import concat
from pclpy import pcl
import pclpy
import cv2
from scipy import spatial

import sys
import os
import argparse

from PCD_io_ext import Load


def load_normals_pcd(output_dir, original_path, concat_path):
    print("Loading PCD files at {} and {}".format(original_path, concat_path))
    pcd_original = Load(original_path)
    height, width, _ = pcd_original.shape

    pcd = pcl.PointCloud.PointNormal()
    reader = pcl.io.PCDReader()
    reader.read(concat_path, pcd)

    normals = np.reshape(pcd.normals, pcd_original.shape)

    # Replace nan pixels with (0, 0, 0)
    normals_mask = np.ones((height, width))

    for x in range(width):
        for y in range(height):           
            if np.isnan(normals[y,x,:]).any():
                normals[y,x,:] = np.zeros(normals[y,x,:].shape)
                normals_mask[y, x] = 0

    base_filename = os.path.basename(original_path)
    cv2.imwrite(output_dir + base_filename.replace('.pcd', '_normals_mask.png'), normals_mask * 255)
    print("Normals mask file written successfully")
    normals_depth2depth = process_normals_for_depth2depth(np.copy(normals), width, height)

    with h5py.File(output_dir + base_filename.replace('.pcd', '_normals.h5'), "w") as f:
        dset = f.create_dataset('/result', data=normals_depth2depth)
    print("H5 file written successfully")


def show_normal_stats(normals):
    print("Shape: {}, Type: {}".format(normals.shape, normals.dtype))
    print("Normals[0] min = {}, max = {}".format(np.min(normals[0,...]), np.max(normals[0,...])))
    print("Contains NAN:", np.isnan(normals).any())  
    print()


def process_normals_for_depth2depth(normals, width=1280, height=1024):
    normals_depth2depth_format = normals.transpose(2, 0, 1)    # convert to 3 x H x W
    if (normals_depth2depth_format.shape != (3, height, width)):
        raise ValueError('Shape of normals should be (3, H, W). Got shape: {}'.format(normals_depth2depth_format.shape))

    # # Convert normals to shape (N, 3)
    # normals_list = np.reshape(normals_depth2depth_format, (3, -1)).transpose((1, 0))

    # # Apply Rotation
    # r = spatial.transform.Rotation.from_euler('x', 90, degrees=True)
    # normals_list = r.apply(normals_list)

    # # Convert normals back to shape (3, H, W)
    # normals_depth2depth_format = np.reshape(normals_list.transpose(1, 0), normals.shape).transpose(2, 0, 1)

    # normals_depth2depth_format = normals_depth2depth_format.transpose(1, 2, 0)
    # normals_depth2depth_format = cv2.resize(normals_depth2depth_format, (width, height), interpolation=cv2.INTER_NEAREST)
    # normals_depth2depth_format = normals_depth2depth_format.transpose(2, 0, 1)

    return normals_depth2depth_format


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert Normals PCD to H5')
    parser.add_argument('--pcd_file_gt', required=False, type=str, default='view_71_GT.pcd', help='Name of .pcd file with extension')
    args = parser.parse_args()

    input_dir = '/home/alex/Projects/TRAIL/depth-completion-solver/input_data/'
    output_dir = '/home/alex/Projects/TRAIL/depth-completion-solver/temporary_data/'
    original_path = input_dir + args.pcd_file_gt
    concat_path = output_dir + args.pcd_file_gt.replace('.pcd', '_concat.pcd')

    load_normals_pcd(output_dir, original_path, concat_path)
