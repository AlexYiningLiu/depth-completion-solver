import numpy as np
import open3d as o3d

import os
import sys
import argparse

# ROBI parameters
w = 1280
h = 1024
fx = 1083.097046
fy = 1083.097046
cx = 379.326874
cy = 509.437195

# depth2depth sample input parameters
# w = 256
# h = 144
# fx = 185
# fy = 185
# cx = 128
# cy = 72

def create_point_cloud(depth_img_path, scale=1000.0):
    depth_img = o3d.io.read_image(depth_img_path)

    cam = o3d.camera.PinholeCameraParameters()
    intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
    cam.intrinsic = intrinsic
    cam.extrinsic = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])

    pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_img, intrinsic, cam.extrinsic, scale)
    # pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])

    origin = o3d.geometry.PointCloud()
    origin_point = np.zeros((1, 3))
    origin.points = o3d.utility.Vector3dVector(origin_point)

    o3d.visualization.draw_geometries([pcd, origin])
    print(pcd)

    converted_pcd_path = depth_img_path.replace('.png', '.pcd')
    r = o3d.io.write_point_cloud(converted_pcd_path, pcd)
    if r:
        print("Converted PCD successfully written to:", converted_pcd_path)
    else:
        print("Could not write PCD")


def compute_normals(pcd_path):
    pcd = o3d.io.read_point_cloud(pcd_path)
    pcd.estimate_normals()
    o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert Normals PCD to H5')
    parser.add_argument('--png_computed', required=False, type=str, default='view_71_output.png', help='Name of .png file outputted by depth2depth')
    args = parser.parse_args()

    input_dir = '/home/alex/Projects/TRAIL/depth-completion-solver/output_data/'
    depth_img_path = input_dir + args.png_computed
    
    create_point_cloud(depth_img_path, scale=4.0)  # Take png_depth_scale used in depth2depth divide by 1000 to get scale here
