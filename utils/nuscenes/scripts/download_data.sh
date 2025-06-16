# !/bin/bash

root_folder="./data/nuscenes"
mkdir -p ./data/nuscenes

# nuimages 
# wget -P $root_folder https://d36yt3mvayqw5m.cloudfront.net/public/nuimages-v1.0/nuimages-v1.0-all-metadata.tgz
# tar -xvzf $root_folder/nuimages-v1.0-all-metadata.tgz -C $root_folder
# rm $root_folder/nuimages-v1.0-all-metadata.tgz


# wget -P $root_folder https://d36yt3mvayqw5m.cloudfront.net/public/nuimages-v1.0/nuimages-v1.0-all-samples.tgz
# tar -xvzf $root_folder/nuimages-v1.0-all-samples.tgz -C $root_folder
# rm $root_folder/nuimages-v1.0-all-samples.tgz

# nuscenes
# wget -P $root_folder https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval05_blobs.tgz
# wget -P $root_folder https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval07_blobs.tgz
# tar -xvzf $root_folder/v1.0-trainval05_blobs.tgz -C $root_folder
# rm $root_folder/v1.0-trainval05_blobs.tgz

# nuscenes mini
# wget -P $root_folder https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-mini.tgz
# tar -xvzf $root_folder/v1.0-mini.tgz -C $root_folder
# rm $root_folder/v1.0-mini.tgz