import os
from datetime import datetime
from argparse import ArgumentParser

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.model_factory import build_model, build_criterion
from src.data.data_factory import build_dataloaders
from src.utils.configs import get_default_configuration, load_config
from src.utils.confusion import BinaryConfusionMatrix
from src.data.nuscenes.utils import NUSCENES_CLASS_NAMES
from src.data.argoverse.utils import ARGOVERSE_CLASS_NAMES
from src.utils.visualise import colorise
from src.data.utils import encode_binary_labels, decode_binary_labels

from PIL import Image
import matplotlib.pyplot as plt


def showColorImg(img_array, mask_name, saveImg):
    # img_array = img_array.astype(int)

    plt.imshow(img_array, cmap='coolwarm', interpolation='nearest')  # , vmin=0, vmax=1
    if mask_name:
        plt.colorbar()
        plt.title(mask_name, y=-0.1)
    if saveImg:
        plt.imsave('./outputs/output.jpg', img_array)
    plt.axis('off')
    plt.show()


def get_visible_mask(fu, cu, image_width, extents, resolution):
    # Get calibration parameters
    # fu, cu = instrinsics[0, 0], instrinsics[0, 2]

    # Construct a grid of image coordinates
    x1, z1, x2, z2 = extents
    x, z = np.arange(x1, x2, resolution), np.arange(z1, z2, resolution)
    ucoords = x / z[:, None] * fu + cu

    # Return all points which lie within the camera bounds
    ucoords = (ucoords >= 0) & (ucoords < image_width)
    # ucoords = np.flip(ucoords, axis=0)  # flip
    # ucoords = np.flip(ucoords, axis=1)  # flip
    return ucoords


# Load the configuration for this experiment
def get_configuration(args):
    # Load config defaults
    config = get_default_configuration()

    # Load dataset options
    config.merge_from_file(f'configs/datasets/{args.dataset}.yml')

    # Load model options
    config.merge_from_file(f'configs/models/{args.model}.yml')

    # Finalise config
    config.freeze()

    return config


def preprocessInput(image, config):
    image = cv2.resize(image, config.img_size)
    image_tensor = torch.from_numpy(image)

    return image_tensor.permute(2, 0, 1).unsqueeze(0).float()


def encode_binary_labels_new(masks):
    bits = np.power(2, np.arange(len(masks), dtype=np.int32))
    return (masks.astype(np.int32) * bits.reshape(-1, 1, 1)).sum(0)


def main():
    parser = ArgumentParser()
    parser.add_argument('--input',
                        default='./images/CAM_FRONT/n015-2018-07-24-11-22-45+0800__CAM_FRONT__1532402927612460.jpg',
                        help='input image for inference')
    parser.add_argument('--model', choices=['pyramid', 'vpn', 'ved'], default='pyramid',
                        help='model to train')
    parser.add_argument('--dataset', choices=['nuscenes', 'argoverse'], default='nuscenes',
                        help='model dataset')
    parser.add_argument('--model_path', default='./logs/run_24-05-30--09-07-21/best.pth',
                        help='model pth path')
    parser.add_argument('--show', default=True, help='whether to show result')
    parser.add_argument('--save', default=False, help='whether to save result')
    args = parser.parse_args()

    # Load configuration
    config = get_configuration(args)

    # Setup experiment
    model = build_model(config.model, config)

    # Initialise confusion matrix
    confusion = BinaryConfusionMatrix(config.num_class)

    # Load ckpt
    ckpt = torch.load(args.model_path)

    # Load model weights
    if isinstance(model, nn.DataParallel):
        model = model.module
    model.load_state_dict(ckpt['model'])

    # cuda
    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()

    # load image and calib
    image = cv2.imread(args.input)
    showColorImg(image, 'raw', args.save)
    calib = np.array([[626.4066, 0.0000, 413.2941],
                      [0.0000, 835.2087, 313.3231],
                      [0.0000, 0.0000, 1.0000]])
    calib = torch.from_numpy(calib).unsqueeze(0).float()
    image_tensor = preprocessInput(image, config)

    if torch.cuda.is_available():
        image_tensor = image_tensor.cuda()
        calib = calib.cuda()

    with torch.no_grad():
        logits = model(image_tensor, calib)

    # get score
    scores = logits.cpu().sigmoid().squeeze(0)  # [14, 600, 800]
    scores_encoded = encode_binary_labels_new(np.array(scores))
    showColorImg(scores_encoded, 'encoded', args.save)
    scores_array = colorise(scores, 'coolwarm', 0, 1)  # [14, 600, 800, 3]

    mask_list = ['drivable_area', 'ped_crossing', 'walkway', 'carpark',
                 'car', 'truck', 'bus', 'trailer', 'construction_vehicle',
                 'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier']

    # camera info
    fx, cx, width = 1266.417203046554, 816.2670197447984, 1600

    for i in range(len(scores_array)):
        mask = get_visible_mask(fx, cx, width, config.map_extents, config.map_resolution)
        masks_array = np.stack([mask[:, :] for _ in range(3)], axis=2)
        scores_array[i] *= masks_array  # fov mask filter

        ucoords = np.flip(scores_array[i], axis=0)  # flip
        ucoords = np.flip(ucoords, axis=1)  # flip
        showColorImg(ucoords, mask_list[i], args.save)

    print("\nDemo visualize complete!")


if __name__ == '__main__':
    main()
