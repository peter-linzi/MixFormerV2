import os
import os.path
import re
import numpy as np
import torch
import csv
import pandas
import random
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader
from lib.train.admin import env_settings


class WebUAV3M(BaseVideoDataset):
    """WebUAV-3M Dataset.

    Publication:
        ``WebUAV-3M: A Benchmark for Unveiling the Power of Million-Scale Deep UAV Tracking``,
        Chunhui Zhang, Guanjie Huang, Li Liu, Shan Huang, Yinan Yang, Xiang Wan,
        Shiming Ge, Dacheng Tao. arXiv 2022.

    Download dataset from https://github.com/flyers/drone-tracking
    """

    def __init__(self, root=None, image_loader=jpeg4py_loader, split=None, seq_ids=None, data_fraction=None):
        """
        args:
            root - path to the got-10k training data. Note: This should point to the 'train' folder inside GOT-10k
            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
            split - 'train' or 'val'. Note: The validation split here is a subset of the official got-10k train split,
                    not NOT the official got-10k validation split. To use the official validation split, provide that as
                    the root folder instead.
            seq_ids - List containing the ids of the videos to be used for training. Note: Only one of 'split' or 'seq_ids'
                        options can be used at the same time.
            data_fraction - Fraction of dataset to be used. The complete dataset is used by default
        """
        root = env_settings().webuav3m_dir if root is None else root
        super().__init__('WEBUAV3M', root, image_loader)

        # Split can be test, val, or ltrval (a validation split consisting of videos from the official train set)
        if split == 'test':
            self.root = os.path.join(root, 'Test')
        elif split == 'val':
            self.root = os.path.join(root, 'Val')
        else:
            self.root = os.path.join(root, 'Train')

        self.sequence_list = self._get_sequence_list()
        self.split = split

        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list)*data_fraction))

        self.sequence_attributes = self._load_attributes()
        self.seq_per_class = self._build_seq_per_class()

        self.class_list = list(self.seq_per_class.keys())
        self.class_list.sort()

    def get_name(self):
        return 'webuav3m'

    def has_class_info(self):
        return True

    def has_occlusion_info(self):
        return True

    def _load_attributes(self):
        sequence_attributes = {s: self._read_attributes(os.path.join(self.root, s)) for s in self.sequence_list}
        return sequence_attributes

    def _read_attributes(self, seq_path):
        try:
            with open(os.path.join(seq_path, 'attributes.txt')) as f:
                attributes = f.readlines()
            object_attributes = OrderedDict({'low_resolution': int(attributes[0]),
                                       'partial_occlusion': int(attributes[1]),
                                       'full_occlusion': int(attributes[2]),
                                       'out_of_view': int(attributes[3]),
                                       'fast_motion': int(attributes[4]),
                                       'camera_motion': int(attributes[5]),
                                       'viewpoint_changes': int(attributes[6]),
                                       'rotation': int(attributes[7]),
                                       'deformation': int(attributes[8]),
                                       'background_clutter': int(attributes[9]),
                                       'scale_variations': int(attributes[10]),
                                       'aspect_ratio_variations': int(attributes[11]),
                                       'illumination_variations': int(attributes[12]),
                                       'motion_blur': int(attributes[13]),
                                       'complexity': int(attributes[14]),
                                       'size': int(attributes[15]),
                                       'length': int(attributes[16])})
        except:
            object_attributes = OrderedDict({'low_resolution': None,
                                       'partial_occlusion': None,
                                       'full_occlusion': None,
                                       'out_of_view': None,
                                       'fast_motion': None,
                                       'camera_motion': None,
                                       'viewpoint_changes': None,
                                       'rotation': None,
                                       'deformation': None,
                                       'background_clutter': None,
                                       'scale_variations': None,
                                       'aspect_ratio_variations': None,
                                       'illumination_variations': None,
                                       'motion_blur': None,
                                       'complexity': None,
                                       'size': None,
                                       'length': None})
        
        object_class = re.sub(r'_\d+$', '', os.path.basename(seq_path))
        object_attributes["object_class_name"] = object_class
        return object_attributes

    def _build_seq_per_class(self):
        seq_per_class = {}

        for i, s in enumerate(self.sequence_list):
            object_class = self.sequence_attributes[s]['object_class_name']
            if object_class in seq_per_class:
                seq_per_class[object_class].append(i)
            else:
                seq_per_class[object_class] = [i]

        return seq_per_class

    def get_sequences_in_class(self, class_name):
        return self.seq_per_class[class_name]

    def _get_sequence_list(self):
        sequence_list = os.listdir(self.root)
        return sequence_list

    def _read_bb_anno(self, seq_path):
        bb_anno_file = os.path.join(seq_path, "groundtruth_rect.txt")
        gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
        return torch.tensor(gt)

    def _read_target_visible(self, seq_path):
        # Read full occlusion and out_of_view
        occlusion_file = os.path.join(seq_path, "absent.txt")

        with open(occlusion_file, 'r', newline='') as f:
            occlusion = torch.ByteTensor([int(v[0]) for v in csv.reader(f)])
        visible = ~occlusion
        return visible

    def _read_scenario(self, seq_path):
        sce_file = os.path.join(seq_path, "scenario.txt")
        with open(sce_file, 'r', newline='') as f:
            sce = torch.ByteTensor(v for v in csv.reader(f))
        return sce

    def _get_sequence_path(self, seq_id):
        return os.path.join(self.root, self.sequence_list[seq_id])

    def get_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_path)

        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = self._read_target_visible(seq_path)
        visible = visible & valid.byte()
        # sce = self._read_scenario(seq_path)

        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def _get_frame_path(self, seq_path, frame_id):
        return os.path.join(seq_path, 'img' ,'{:06}.jpg'.format(frame_id+1))    # frames start from 1

    def _get_frame(self, seq_path, frame_id):
        return self.image_loader(self._get_frame_path(seq_path, frame_id))

    def get_class_name(self, seq_id):
        obj_attributes = self.sequence_attributes[self.sequence_list[seq_id]]

        return obj_attributes['object_class_name']

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)
        obj_attributes = self.sequence_attributes[self.sequence_list[seq_id]]

        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        return frame_list, anno_frames, obj_attributes
    
