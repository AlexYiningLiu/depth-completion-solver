import cv2
import numpy as np
import h5py
import imageio

import sys
import argparse
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import utils


def scaled_depth_to_rgb_depth(path_scaled_depth_img, output_filename, scale):
    if not os.path.isfile(path_scaled_depth_img):
        print('\nError: Source file does not exist: {}\n'.format(path_scaled_depth_img))
        exit()

    scaled_depth = cv2.imread(path_scaled_depth_img, cv2.IMREAD_UNCHANGED)
    metric_depth = utils.unscale_depth(scaled_depth, scale)
    print("{} Max depth: {}, min depth: {}".format(os.path.basename(path_scaled_depth_img), np.max(metric_depth), np.min(metric_depth)))

    rgb_depth = utils.depth2rgb(metric_depth, dynamic_scaling=True, color_mode=cv2.COLORMAP_JET)

    cv2.imwrite(output_filename, rgb_depth)
    return


def normals_to_rgb_normals(path_input_normals, output_filename):
    if not os.path.isfile(path_input_normals):
        print('\nError: Source file does not exist: {}\n'.format(path_input_normals))
        exit()

    with h5py.File(path_input_normals, 'r') as hf:
        normals = hf.get('/result')
        normals = np.array(normals)
        hf.close()

    # From (3, height, width) to (height, width, 3)
    normals = normals.transpose(1, 2, 0)

    rgb_normals = utils.normal_to_rgb(normals, output_dtype='uint8')
    cv2.imwrite(output_filename, rgb_normals)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert the black, 16-bit .png depth images to RGB depth images.')
    parser.add_argument('--png_input', type=str, required=False, default='converted_view_71_raw.png', help='Name of input to depth2depth')
    parser.add_argument('--png_expected', type=str, required=False, default='converted_view_71_GT.png', help='Name of converted GT .png')
    parser.add_argument('--png_output', type=str, required=False, default='view_71_output.png', help='Name of depth2depth output .png')
    parser.add_argument('--normals_input', type=str, required=False, default='view_71_GT_normals.h5', help='Name of normals input to depth2depth')
    parser.add_argument('--scale', type=int, default=4000, help='Depth unit scale')
    args = parser.parse_args()

    dir_path_output = '/home/alex/Projects/TRAIL/depth-completion-solver/output_data/'
    dir_path_temp = '/home/alex/Projects/TRAIL/depth-completion-solver/temporary_data/'

    # Input depth
    scaled_depth_to_rgb_depth(dir_path_temp + args.png_input, dir_path_output + args.png_input.replace('.png', '_rgb.jpg'), args.scale)

    # Output depth
    scaled_depth_to_rgb_depth(dir_path_output + args.png_output, dir_path_output + args.png_output.replace('.png', '_rgb.jpg'), args.scale)

    # Expected output depth
    scaled_depth_to_rgb_depth(dir_path_temp + args.png_expected, dir_path_output + args.png_expected.replace('.png', '_rgb.jpg'), args.scale)

    # Normals
    normals_to_rgb_normals(dir_path_temp + args.normals_input, dir_path_output + args.normals_input.replace('.h5', '_rgb.jpg'))
