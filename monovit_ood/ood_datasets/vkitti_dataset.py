from __future__ import absolute_import, division, print_function

import os
import numpy as np
import sys
sys.path.append(os.getcwd()+"/../monodepth2_ood/")
from monodepth2.datasets.mono_dataset import MonoDataset
import PIL.Image as pil
import skimage.transform
import cv2


class virtualKITTIDataset(MonoDataset):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        self.full_res_shape = (1242, 375)
        self.side_map = {"0": 0, "1": 1, "l": 0, "r": 1}
        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        super(virtualKITTIDataset, self).__init__(*args, **kwargs)

    def check_depth(self):
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = int(line[1])
        side = line[2]

        depth_png_filename = os.path.join(
            self.data_path,
            scene_name, "frames",
            "depth/Camera_{}/depth_{:05d}.png".format(self.side_map[side], int(frame_index)))

        return os.path.isfile(depth_png_filename)

    def get_depth(self, folder, frame_index, side, do_flip):
        depth_png_filename = os.path.join(
            self.data_path,
            folder, "frames", "depth/Camera_{}/depth_{:05d}.png".format(self.side_map[side], int(frame_index)))

        depth_gt = cv2.imread(depth_png_filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        depth_gt = skimage.transform.resize(
            depth_gt, self.full_res_shape[::-1], order=0, preserve_range=True, mode='constant')

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        # convert cm to m
        depth_gt = depth_gt / 100.0
        return depth_gt

    def get_image_path(self, folder, frame_index, side):
        f_str = "rgb_{:05d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path, folder, "frames", "rgb", "Camera_{}/".format(self.side_map[side]), f_str)
        return image_path

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

