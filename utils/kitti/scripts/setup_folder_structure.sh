#!/bin/bash

scripts=(
    # "depth_structure.py"
    "semantic_segmentation.py"
    "object_detection_3d.py"
)

path="./utils/kitti/folder_structure"

for script in "${scripts[@]}"; do
    echo "Running $path/$script"
    python3 "$path/$script"
done