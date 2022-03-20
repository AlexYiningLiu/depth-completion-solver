#!/usr/bin/env bash

INPUT_PNG="converted_view_71_raw.png"
OUTPUT_PNG="view_71_output.png"
INPUT_NORMALS="view_71_GT_normals.h5"
PNG_EXPECTED="converted_view_71_GT.png"
SCALE=4000

bin/x86_64/depth2depth \
 ${INPUT_PNG} \
 ${OUTPUT_PNG} \
 -xres 1280 -yres 1024 \
 -fx 1083 -fy 1083 \
 -cx 379 -cy 509 \
 -inertia_weight 100 \
 -smoothness_weight 0.0001 \
 -tangent_weight 1 \
 -input_normals ${INPUT_NORMALS} \
 -verbose \
 -png_depth_scale ${SCALE} \
 -x_left 428 \
 -y_bottom 399 \
 -x_right 719 \
 -y_top 660 \
 -min_depth_threshold 0.32 \

# png_depth_scale needs to be the product of the real png scale (32) and the additional visualization scale (125)

args=(
 --png_input ${INPUT_PNG}
 --png_expected ${PNG_EXPECTED}
 --png_output ${OUTPUT_PNG}
 --normals_input ${INPUT_NORMALS}
 --scale ${SCALE}
)

cd /home/alex/Projects/TRAIL/depth-completion-solver/api
python convert_intermediate_data_to_rgb.py ${args[@]}
