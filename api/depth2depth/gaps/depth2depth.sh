#!/usr/bin/env bash

bin/x86_64/depth2depth \
 "converted_view_71_raw.png" \
 "view_71_output.png" \
 -xres 1280 -yres 1024 \
 -fx 1083 -fy 1083 \
 -cx 379 -cy 509 \
 -inertia_weight 1000 \
 -smoothness_weight 0.00001 \
 -tangent_weight 1 \
 -input_normals "view_71_GT_normals.h5" \
 -verbose \
 -png_depth_scale 4000 \
 -x_left 514 \
 -y_bottom 285 \
 -x_right 655 \
 -y_top 460 \
 -min_depth_threshold 0.302 \

# png_depth_scale needs to be the product of the real png scale (32) and the additional visualization scale (125)

cd /home/alex/Projects/TRAIL/depth-completion-solver/api
python convert_intermediate_data_to_rgb.py
