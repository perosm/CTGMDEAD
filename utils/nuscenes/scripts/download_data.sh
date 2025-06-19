# !/bin/bash

root_folder="./data/nuscenes"
nuimages_folder=$root_folder/"nuimages"
nuscenes_folder=$root_folder/"nuscenes"

mkdir -p $root_folder
mkdir -p $nuimages_folder
mkdir -p $nuscenes_folder

# nuimages 
wget -P $nuimages_folder https://d36yt3mvayqw5m.cloudfront.net/public/nuimages-v1.0/nuimages-v1.0-all-metadata.tgz
tar -xvzf $nuimages_folder/nuimages-v1.0-all-metadata.tgz -C $nuimages_folder
rm $nuimages_folder/nuimages-v1.0-all-metadata.tgz


# wget -P $nuimages_folder https://d36yt3mvayqw5m.cloudfront.net/public/nuimages-v1.0/nuimages-v1.0-all-samples.tgz
# tar -xvzf $nuimages_folder/nuimages-v1.0-all-samples.tgz -C $nuimages_folder
# rm $nuimages_folder/nuimages-v1.0-all-samples.tgz


# nuscenes
# wget -P $nuscenes_folder https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval05_blobs.tgz
# wget -P $nuscenes_folder https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval07_blobs.tgz
# tar -xvzf $nuscenes_folder/v1.0-trainval05_blobs.tgz -C $nuscenes_folder
# rm $nuscenes_folder/v1.0-trainval05_blobs.tgz

# nuscenes mini
# wget -P $nuscenes_folder https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-mini.tgz
# tar -xvzf $nuscenes_folder/v1.0-mini.tgz -C $nuscenes_folder
# rm $nuscenes_folder/v1.0-mini.tgz


