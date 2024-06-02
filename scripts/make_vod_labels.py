import os
import sys
import numpy as np
from PIL import Image
# from progressbar import ProgressBar

from vod.configuration import KittiLocations
from vod.frame import FrameDataLoader

sys.path.append(os.path.abspath(os.path.join(__file__, '../..')))

from src.utils.configs import get_default_configuration
from src.data.utils import get_visible_mask, get_occlusion_mask, encode_binary_labels
from src.data.argoverse.utils import get_object_masks, get_map_mask


def process_frame(split, scene, camera, frame, map_data, config):
    # Compute object masks
    masks = get_object_masks(scene, camera, frame, config.map_extents,
                             config.map_resolution)

    # Compute drivable area mask
    masks[0] = get_map_mask(scene, camera, frame, map_data, config.map_extents,
                            config.map_resolution)

    # Ignore regions of the BEV which are outside the image
    calib = scene.get_calibration(camera)
    masks[-1] |= ~get_visible_mask(calib.K, calib.camera_config.img_width,
                                   config.map_extents, config.map_resolution)

    # Ignore regions of the BEV which are occluded (based on LiDAR data)
    lidar = scene.get_lidar(frame)
    cam_lidar = calib.project_ego_to_cam(lidar)
    masks[-1] |= get_occlusion_mask(cam_lidar, config.map_extents,
                                    config.map_resolution)

    # Encode masks as an integer bitmask
    labels = encode_binary_labels(masks)

    # Create a filename and directory
    timestamp = str(scene.image_timestamp_list_sync[camera][frame])
    output_path = os.path.join(config.argoverse.label_root, split,
                               scene.current_log, camera,
                               f'{camera}_{timestamp}.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save encoded label file to disk
    Image.fromarray(labels.astype(np.int32), mode='I').save(output_path)


if __name__ == '__main__':

    config = get_default_configuration()
    config.merge_from_file('../configs/datasets/vod.yml')

    # Create an vod instance
    root_dir = r"/home/hosico/Dataset/hdd1/Dataset/TUDelft_VOD_dataset/vod"
    kitti_locations = KittiLocations(root_dir=root_dir,
                                     output_dir="example_output",
                                     frame_set_path="",
                                     pred_dir="",
                                     )

    # get train and val
    train_list = []
    val_list = []
    train_txt = os.path.join(root_dir, 'lidar/ImageSets/train.txt')
    val_txt = os.path.join(root_dir, 'lidar/ImageSets/val.txt')
    with open(train_txt, 'r') as file:
        lines = file.readlines()
        for line in lines:
            train_list.append(line.strip())

    with open(val_txt, 'r') as file:
        lines = file.readlines()
        for line in lines:
            val_list.append(line.strip())

    for split in [train_list, val_list]:
        for frame_index in split:
            frame_data = FrameDataLoader(kitti_locations=kitti_locations, frame_number=frame_index)
            transforms = FrameTransformMatrix(frame_data)
            lidar2camera = transforms.t_lidar_camera


