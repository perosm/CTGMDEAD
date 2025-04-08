#!/bin/bash

semseg_url=https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/3bmmnfb4bp-1.zip # semantic segmentation
semseg_filename=$(basename $semseg_url)
semseg_foldername="${semseg_filename%.zip}"
target_tar="kitti_semseg_unizg.tar.gz"
final_folder="kitti_semseg_unizg"

echo "Downloading $semseg_filename"
wget "$semseg_url" -O "./data/kitti/$semseg_filename"
echo "Unzipping: $semseg_filename"
unzip -o "./data/kitti/$semseg_filename" -d "./data/kitti/$semseg_foldername"
tar -xzvf "./data/kitti/$semseg_foldername/KITTI-SEMSEG-UNIZG/$target_tar" -C "./data/kitti"
rm "./data/kitti/$semseg_filename"
rm -rf "./data/kitti/$semseg_foldername"

file_urls=(
    # https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_annotated.zip # depth
    https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip # object detection 3d
    https://s3.eu-central-1.amazonaws.com/avg-kitti/devkit_object.zip
)

for url in "${file_urls[@]}"; do
    filename=$(basename "$url")
    foldername="${filename%.zip}"

    echo "Downloading: $filename"
    wget "$url" -O "./data/kitti/$filename"
    
    if [ $? -eq 0 ]; then
        echo "Unzipping: $filename"
        unzip -o "./data/kitti/$filename" -d "./data/kitti/$foldername"
        rm "./data/kitti/$filename"
    else
        echo "Failed to download $filename"
        exit 1
    fi
done