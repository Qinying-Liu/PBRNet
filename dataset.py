import os
import cv2
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F


class THUMOS(Dataset):
    def __init__(self, sequences, dir_path, num_frame_clip, crop_size=96, max_instance=None, aug=True):
        self.sequences = sequences
        self.dir_path = dir_path
        self.aug = aug
        self.max_instance = max_instance
        self.crop_size = crop_size
        self.num_frame_clip = num_frame_clip

    def get_frame_data(self, vid_name, clip_start, clip_end):
        clip_data = []
        filepath = os.path.join(self.dir_path, vid_name)
        imgs = sorted(os.listdir(filepath))
        # s = max(clip_start, 0)
        # e = min(clip_end, len(imgs) - 1)
        assert clip_start >= 0
        assert clip_end <= len(imgs) - 1
        if self.aug:
            new_h = np.random.randint(self.crop_size, 128)
            new_w = np.random.randint(self.crop_size, 128)
            max_h = new_h - self.crop_size
            max_w = new_w - self.crop_size
            off_h = np.random.randint(max_h) if max_h > 0 else 0
            off_w = np.random.randint(max_w) if max_w > 0 else 0
        else:
            new_h = self.crop_size
            new_w = self.crop_size

        for i in range(clip_start, clip_end + 1):
            image_path = os.path.join(filepath, imgs[i])
            img_data = cv2.imread(image_path).astype(np.float32)
            img_data = cv2.resize(img_data, (new_w, new_h))
            clip_data.append(img_data)
        clip_data = np.stack(clip_data, 0)
        if self.aug:
            clip_data = clip_data[:, off_h:(off_h + self.crop_size), off_w:(off_w + self.crop_size)]

        clip_data = clip_data / 255.
        clip_data = np.transpose(clip_data, (3, 0, 1, 2))
        return clip_data

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, i):
        clip_info = self.sequences[i]
        vid_name = clip_info['name']
        clip_start, clip_end = clip_info['clip_interval']
        clip_data = self.get_frame_data(vid_name, clip_start, clip_end)
        if clip_data.shape[1] < self.num_frame_clip:
            clip_data = np.pad(clip_data, ((0, 0), (0, self.num_frame_clip - clip_data.shape[1]), (0, 0), (0, 0)),
                               mode='constant',
                               constant_values=0)
        instances = np.array(clip_info['instances'])
        actions = instances[:, :2].astype(np.float32)
        labels = instances[:, 2].astype(np.int64)
        if self.max_instance:
            if actions.shape[0] < self.max_instance:
                pad_len = self.max_instance - actions.shape[0]
                actions = np.pad(actions, ((0, pad_len), (0, 0)),
                                 mode='constant',
                                 constant_values=-1)
                labels = np.pad(labels, (0, pad_len),
                                mode='constant',
                                constant_values=-1)
            else:
                actions = actions[:self.max_instance]
                labels = labels[:self.max_instance]
        start_label = np.array(clip_info['start_label'], dtype=np.int32)
        end_label = np.array(clip_info['end_label'], dtype=np.int32)
        return clip_data, actions, labels, start_label, end_label
