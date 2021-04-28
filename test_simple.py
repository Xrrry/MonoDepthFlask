# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

import torch
from torchvision import transforms, datasets

import networks
from layers import disp_to_depth
from utils import download_model_if_doesnt_exist


def test_simple(ima_path, feed_width, feed_height, device, encoder, depth_decoder):
    """Function to predict for a single image or folder of images
    """

    # FINDING INPUT IMAGES
    # if os.path.isfile(ima_path):
    #     # Only testing on a single image
    #     paths = [ima_path]
    output_directory = os.path.dirname(ima_path)
    # elif os.path.isdir(ima_path):
    #     # Searching folder for images
    #     # paths = glob.glob(os.path.join(ima_path, '*.{}'.format(args.ext)))
    #     paths = glob.glob(os.path.join(ima_path, '*png'))
    #     output_directory = ima_path
    # else:
    #     raise Exception("Can not find ima_path: {}".format(ima_path))
    #
    # print("-> Predicting on {:d} test images".format(len(paths)))

    # PREDICTING ON EACH IMAGE IN TURN
    with torch.no_grad():
        # for idx, image_path in enumerate(paths):

        if ima_path.endswith("_disp.jpg"):
            # don't try to predict disparity for a disparity image!
            return

        # Load image and preprocess
        input_image = pil.open(ima_path).convert('RGB')
        original_width, original_height = input_image.size
        input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
        input_image = transforms.ToTensor()(input_image).unsqueeze(0)

        # PREDICTION
        input_image = input_image.to(device)
        features = encoder(input_image)
        outputs = depth_decoder(features)

        disp = outputs[("disp", 0)]
        disp_resized = torch.nn.functional.interpolate(
            disp, (original_height, original_width), mode="bilinear", align_corners=False)
        print(disp.shape)
        # print(disp.reshape(feed_height, feed_width).shape)
        print("disp_resized", disp_resized.shape)
        print(disp_resized.max(), disp_resized.min())
        disp_resized_rs = disp_resized.squeeze().flip(dims=[0])
        print("disp_resized_rs", disp_resized_rs.shape)
        print(disp_resized_rs[0].shape)
        # disp_resized_rs = torch.chunk(disp_resized_rs, 2, dim=0)[0]  # 截取下半部分
        out = torch.chunk(disp_resized_rs, 5, dim=1)
        print("out", out[0].shape)
        for index, o in enumerate(out):
            print(index + 1, o.max(), o.min())

        # Saving numpy file
        output_name = os.path.splitext(os.path.basename(ima_path))[0]
        # name_dest_npy = os.path.join(output_directory, "{}_disp.npy".format(output_name))
        # scaled_disp, _ = disp_to_depth(disp, 0.1, 100)
        # np.save(name_dest_npy, scaled_disp.cpu().numpy())

        # Saving colormapped depth image
        disp_resized_np = disp_resized.squeeze().cpu().numpy()
        # vmax = np.percentile(disp_resized_np, 95)
        # normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
        normalizer = mpl.colors.Normalize()
        mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
        colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
        im = pil.fromarray(colormapped_im)

        name_dest_im = os.path.join(output_directory, "{}_disp.jpeg".format(output_name))
        im.save(name_dest_im)

        # print("   Processed {:d} of {:d} images - saved prediction to {}".format(
        #     idx + 1, len(paths), name_dest_im))

    print('-> Done!')
    return sound_map(out)


def sound_map(out):
    maxs = out[0].max()
    maxi = 0
    for index, o in enumerate(out):
        if o.max() > maxs:
            maxs = o.max()
            maxi = index
    sound_level = 0
    if maxs >= 0.9:
        sound_level = 4
    elif maxs >= 0.8:
        sound_level = 3
    elif maxs >= 0.7:
        sound_level = 2
    elif maxs >= 0.6:
        sound_level = 1

    if maxi == 0:
        left_sound = 10
        right_sound = 0
    elif maxi == 1:
        left_sound = 10
        right_sound = 5
    elif maxi == 2:
        left_sound = 10
        right_sound = 10
    elif maxi == 3:
        left_sound = 5
        right_sound = 10
    else:
        left_sound = 0
        right_sound = 10
    # print(left_sound, right_sound, sound_level)
    return left_sound, right_sound, sound_level


if __name__ == '__main__':
    print('error_in_main')
