# ---------------------------------------------------------------
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for CMD. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------


import os
import os.path as osp
import random

import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.datasets.folder import make_dataset
from natsort import natsorted
import decord

from einops import rearrange
from typing import Tuple, List

def resize_crop(
        video: torch.Tensor, resolution: Tuple[int, int]
        ) -> torch.Tensor:
    """ Resizes video with smallest axis to `resolution * extra_scale`
        and then crops a `resolution` x `resolution` bock. If `crop_mode == "center"`
        do a center crop, if `crop_mode == "random"`, does a random crop
    Args
        video: a tensor of shape [c, t, h, w] in {0, ..., 255}
        resolution: an int
        crop_mode: 'center', 'random'
    Returns
        a processed video of shape [c, t, h, w]
    """
    _, _, h, w = video.shape

    if h > w:
        half = (h - w) // 2
        cropsize = (0, half, w, half + w)  # left, upper, right, lower
    elif w >= h:
        half = (w - h) // 2
        cropsize = (half, 0, half + h, h)

    video = video[:, :, cropsize[1]:cropsize[3],  cropsize[0]:cropsize[2]]
    video = F.interpolate(video, size=resolution, mode='bilinear', align_corners=False)

    return video.contiguous()


class InfiniteSampler(torch.utils.data.Sampler):
    def __init__(
            self,
            dataset: Dataset,
            rank: int = 0,
            num_replicas: int = 1,
            shuffle: bool = True,
            seed: int = 0,
            window_size: float = 0.5
            ):
        super().__init__(dataset)
        self.dataset = dataset
        self.rank = rank
        self.num_replicas = num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.window_size = window_size

    def __iter__(self):
        order = np.arange(len(self.dataset))
        rnd = None
        window = 0
        if self.shuffle:
            rnd = np.random.RandomState(self.seed)
            rnd.shuffle(order)
            window = int(np.rint(order.size * self.window_size))

        idx = 0
        while True:
            i = idx % order.size
            if idx % self.num_replicas == self.rank:
                yield order[i]
            if window >= 2:
                j = (i - rnd.randint(window)) % order.size
                order[i], order[j] = order[j], order[i]
            idx += 1


class VideoFolderDataset(Dataset):
    def __init__(
            self,
            root: str,
            train: bool = True,
            resolution: Tuple[int, int] = (256, 256),
            n_frames: int = 16,
            fold: int = 1,
            return_vid: bool = True,
            seed: int = 42,
            ret_class_idx: bool = False,
            **super_kwargs,
            ):

        video_root = osp.join(os.path.join(root))
        if not 1 <= fold <= 3:
            raise ValueError("fold should be between 1 and 3, got {}".format(fold))

        self.path = video_root
        name = video_root.split('/')[-1]
        self.name = name
        self.train = train
        self.fold = fold
        self.resolution = resolution
        self.nframes = n_frames
        self.annotation_path = os.path.join(video_root, 'ucfTrainTestlist')
        self.classes = list(
            natsorted(p for p in os.listdir(video_root) if osp.isdir(osp.join(video_root, p)))
            )
        self.classes.remove('ucfTrainTestlist')
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.samples = make_dataset(video_root, self.class_to_idx, ('avi',), is_valid_file=None)
        video_list = [x[0] for x in self.samples]

        self.video_list = video_list
        self.num_channels = 3
        self.return_vid = return_vid
        self.indices = self._select_fold(
            self.video_list, self.annotation_path, fold, train
            )
        self.size = len(self.indices)
        random.seed(seed)
        self.shuffle_indices = [i for i in range(self.size)]
        random.shuffle(self.shuffle_indices)
        self._need_init = True
        self.sample_start_idx = 0
        self.sample_frame_rate = 1 # unused 
        self.ret_class_idx = ret_class_idx


    def _select_fold(
            self,
            video_list: List[str],
            annotation_path: str,
            fold: int,
            train: bool) -> List[int]:
        name = "train" if train else "test"
        name = "{}list{:02d}.txt".format(name, fold)
        f = os.path.join(annotation_path, name)
        selected_files = []
        with open(f, "r") as fid:
            data = fid.readlines()
            data = [x.strip().split(" ") for x in data]
            data = [os.path.join(self.path, x[0]) for x in data]
            selected_files.extend(data)

        selected_files = set(selected_files)
        indices = [
            i for i in range(len(video_list)) if video_list[i] in selected_files
            ]
        return indices

    def __len__(self):
        return self.size

    def _preprocess(self, video: torch.Tensor) -> torch.Tensor:
        video = resize_crop(video, self.resolution)
        return video

    def __getitem__(self, idx: int) -> torch.Tensor:
        shuffled_idx = self.shuffle_indices[idx]
        idx = self.indices[shuffled_idx]
        vr = decord.VideoReader(self.video_list[idx])
        cls_name = self.video_list[idx].split('/')[-2]
        class_idx = self.class_to_idx[cls_name]
        rand_offset = np.random.randint(len(vr) - self.nframes + 1)
        sample_index = list(
            range(self.sample_start_idx, len(vr), 1)
            )[rand_offset:rand_offset+self.nframes]
        video = vr.get_batch(sample_index).asnumpy()
        video = rearrange(video, "f h w c -> c f h w")
        pixel_values = torch.from_numpy(video).float() / 127.5 - 1.0
        pixel_values = self._preprocess(pixel_values)

        return dict(x=pixel_values, y=class_idx)

def get_dataset(name: str, **kwargs) -> VideoFolderDataset:
    if name == 'UCF101':
        return VideoFolderDataset(**kwargs)
    else:
        raise NotImplementedError(name)
