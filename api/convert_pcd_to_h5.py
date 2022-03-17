import h5py
import numpy as np
import argparse
from PCD_io_ext import Load
from pclpy import pcl
import pclpy
import cv2
import os
from scipy import spatial
import sys


def load_normals_pcd():
    pcd_original = Load(original_path)
    height, width, _ = pcd_original.shape

    pcd = pcl.PointCloud.PointNormal()
    reader = pcl.io.PCDReader()
    reader.read(concat_path, pcd)

    normals = np.reshape(pcd.normals, pcd_original.shape)
    coordinates = np.reshape(pcd.xyz, pcd_original.shape)

    # Replace nan pixels with (0, 0, 0)
    negative_or_zero_z = 0
    positive_z = 0

    normals_mask = np.ones((height, width))

    for x in range(width):
        for y in range(height):           
            if np.isnan(normals[y,x,:]).any():
                normals[y,x,:] = np.zeros(normals[y,x,:].shape)
                normals_mask[y, x] = 0

    cv2.imwrite(output_dir + "normals_mask.png", normals_mask * 255)

    # ny_channel = normals[...,1]
    # nz_channel = normals[...,2]
    # print(ny_channel.shape)
    # print("Negative or zero ny elements: {} out of {}".format(np.count_nonzero(ny_channel <= 0), ny_channel.size))  
    # print("Negative or zero nz elements: {} out of {}".format(np.count_nonzero(nz_channel <= 0), nz_channel.size))  

    normals_depth2depth = process_normals_for_depth2depth(np.copy(normals))

    # ny_channel = normals_depth2depth[1,...]
    # nz_channel = normals_depth2depth[2,...]
    # print(ny_channel.shape)
    # print("Negative or zero ny elements: {} out of {}".format(np.count_nonzero(ny_channel <= 0), ny_channel.size))  
    # print("Negative or zero nz elements: {} out of {}".format(np.count_nonzero(nz_channel <= 0), nz_channel.size))  

    with h5py.File('/home/alex/Projects/TRAIL/cleargrasp/api/depth2depth/gaps/sample_files/computed_normals.h5', "w") as f:
        dset = f.create_dataset('/result', data=normals_depth2depth)



def show_normal_stats(normals):
    print("Shape: {}, Type: {}".format(normals.shape, normals.dtype))
    print("Normals[0] min = {}, max = {}".format(np.min(normals[0,...]), np.max(normals[0,...])))
    print("Contains NAN:", np.isnan(normals).any())

    # Check NAN percentage and whether vectors are unit
    # normalized = np.apply_along_axis(np.linalg.norm, 2, normals)
    # nan_array = np.isnan(normalized)
    # print("{} NAN elements in {} total elements".format(np.count_nonzero(nan_array), normalized.size))
    # non_nan_array = ~nan_array
    # normalized = normalized[non_nan_array]
    # print("All non NAN vectors are unit:", (np.isclose(normalized, 1.0)).all())

    ny_channel = normals[1,...]
    nz_channel = normals[2,...]
    print(ny_channel.shape)
    print("Negative or zero ny elements: {} out of {}".format(np.count_nonzero(ny_channel <= 0), ny_channel.size))  
    print("Negative or zero nz elements: {} out of {}".format(np.count_nonzero(nz_channel <= 0), nz_channel.size))    
    
    print()


def process_normals_for_depth2depth(normals, width=1280, height=1024):
    normals_depth2depth_format = normals.transpose(2, 0, 1)    # convert to 3 x H x W
    if (normals_depth2depth_format.shape != (3, 1024, 1280)):
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
    # parser = argparse.ArgumentParser(description='Normals PCD to H5')
    # parser.add_argument('--pcd_path', required=False, type=str, default="", help='Path to sample_files dir')
    # args = parser.parse_args()

    # load_normals_pcd(args.pcd_path)
    input_dir = '/home/alex/Projects/TRAIL/datasets/SCS_Sample/'
    output_dir = '/home/alex/Projects/TRAIL/cleargrasp/api/depth2depth/gaps/sample_files/'
    original_path = input_dir + "zigzag_scene5_view0.pcd"
    concat_path = input_dir + "zigzag_scene5_view0_concat.pcd"

    load_normals_pcd()
