cd /ws/external && ./build.sh
cd /ws/external/Examples
#!/usr/bin/env python3
# ./Monocular/mono_tum ../Vocabulary/ORBvoc.txt Monocular/TUM1.yaml /ws/data/rgbd_dataset_freiburg1_xyz /ws/data/rgbd_dataset_freiburg1_xyz/rgb
./Monocular/mono_tum ../Vocabulary/ORBvoc.txt Monocular/TUM1.yaml /ws/data/rgbd_dataset_freiburg1_xyz /ws/external/masks
# /ws/external/src/python/mask_rcnn_coco.h5
echo "DONE!!"
