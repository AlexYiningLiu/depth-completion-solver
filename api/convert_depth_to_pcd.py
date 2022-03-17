from cv2 import IMREAD_ANYDEPTH, IMREAD_UNCHANGED
import numpy as np
import os
import sys
import open3d as o3d
import cv2

# depth_unit: 0.03125
# width: 1280
# height: 1024
# fx: 1083.097046
# fy: 1083.097046
# cx: 379.326874
# cy: 509.437195
# baseline: 101.1839
# disparity_shift: 0

w = 1280
h = 1024
fx = 1083.097046
fy = 1083.097046
cx = 379.326874
cy = 509.437195

# w = 256
# h = 144
# fx = 185
# fy = 185
# cx = 128
# cy = 72


def scale_depth_image(depth_img_path, out_path, scale=32.0):
    depth_img = cv2.imread(depth_img_path, IMREAD_ANYDEPTH)
    # depth_img = np.true_divide(depth_img, scale)
    # depth_img = np.multiply(depth_img, 32)
    depth_img = depth_img.astype(np.uint16)
    print(np.min(depth_img), np.max(depth_img))
    cv2.imwrite(out_path, depth_img)


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
    r = o3d.io.write_point_cloud("/home/alex/Projects/TRAIL/datasets/SCS_Sample/depth_img_converted.pcd", pcd)
    if r:
        print('written')
    else:
        print('could not write')


def compute_normals(pcd_path):
    pcd = o3d.io.read_point_cloud(pcd_path)
    pcd.estimate_normals()
    o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    input_depth = '/home/alex/Projects/TRAIL/cleargrasp/api/depth2depth/gaps/sample_files/input-depth.png'
    zigzag_depth = '/home/alex/Projects/TRAIL/cleargrasp/api/depth2depth/gaps/sample_files/zigzag_depth_output.png'
    # scaled_out_path = '/home/alex/Projects/TRAIL/cleargrasp/api/depth2depth/gaps/sample_files/correct_scaled.png'
    # scale_depth_image(zigzag_depth, scaled_out_path, 32.0)
    # create_point_cloud(zigzag_depth, 4.0)

    input_pcd_path = "/home/alex/Projects/TRAIL/datasets/SCS_Sample/zigzag_scene5_view0.pcd"
    compute_normals(input_pcd_path)
