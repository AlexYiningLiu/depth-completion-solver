#!/usr/bin/env bash

bin/x86_64/depth2depth \
 "sample_files/converted_depth_input.png" \
 "sample_files/zigzag_depth_output.png" \
 -xres 1280 -yres 1024 \
 -fx 1083 -fy 1083 \
 -cx 379 -cy 509 \
 -inertia_weight 1000 \
 -smoothness_weight 0.00001 \
 -tangent_weight 1 \
 -true_depth "sample_files/converted_expected_output.png" \
 -input_normals "sample_files/computed_normals.h5" \
 -debug \
 -verbose \
 -output_plot "sample_files/zigzag_error_plot" \
 -png_depth_scale 4000 \
 -x_left 407 \
 -y_bottom 372 \
 -x_right 533 \
 -y_top 500 \
 -min_depth_threshold 0.302 \

#  -minimum_depth 0.5 \
#  -maximum_depth 0.6 \
# need to be the product of the real png scale (32) and the additional visualization scale (125)

# bin/x86_64/depth2depth \
#  "sample_files/input-depth.png" \
#  "sample_files/output-depth.png" \
#  -xres 256 -yres 144 \
#  -fx 185 -fy 185 \
#  -cx 128 -cy 72 \
#  -inertia_weight 1000 \
#  -smoothness_weight 0.001 \
#  -tangent_weight 1 \
#  -input_normals "sample_files/normals.h5" \
#  -input_tangent_weight "sample_files/occlusion-weight.png" \
#  -debug \
#  -verbose \
#  -true_depth "sample_files/expected-output-depth.png" \
#  -output_plot "sample_files/input-error_plot" \


python convert_intermediate_data_to_rgb.py --sample_files_dir "sample_files/" --scale 4000
# need to be the product of the real png scale (32) and the additional visualization scale (125)

# depth_unit: 0.03125
# width: 1280
# height: 1024
# fx: 1083.097046
# fy: 1083.097046
# cx: 379.326874
# cy: 509.437195
# baseline: 101.1839
# disparity_shift: 0