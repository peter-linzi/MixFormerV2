import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
import os


class WebUAV3MDataset(BaseDataset):
    """WebUAV-3M Dataset.

    Publication:
        ``WebUAV-3M: A Benchmark for Unveiling the Power of Million-Scale Deep UAV Tracking``,
        Chunhui Zhang, Guanjie Huang, Li Liu, Shan Huang, Yinan Yang, Xiang Wan,
        Shiming Ge, Dacheng Tao. arXiv 2022.

    Download dataset from https://github.com/flyers/drone-tracking
    """
    def __init__(self, split):
        super().__init__()
        # Split can be test, val, or ltrval (a validation split consisting of videos from the official train set)
        if split == 'test':
            self.base_path = os.path.join(self.env_settings.webuav3m_path, 'Test')
        elif split == 'val':
            self.base_path = os.path.join(self.env_settings.webuav3m_path, 'Val')
        else:
            self.base_path = os.path.join(self.env_settings.webuav3m_path, 'Train')

        self.sequence_list = self._get_sequence_list()
        self.split = split

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        anno_path = '{}/{}/groundtruth_rect.txt'.format(self.base_path, sequence_name)

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)

        frames_path = '{}/{}/img'.format(self.base_path, sequence_name)
        frame_list = [frame for frame in os.listdir(frames_path) if frame.endswith(".jpg")]
        frame_list.sort(key=lambda f: int(f[:-4]))
        frames_list = [os.path.join(frames_path, frame) for frame in frame_list]

        return Sequence(sequence_name, frames_list, 'webuav3m', ground_truth_rect.reshape(-1, 4))

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        sequence_list = os.listdir(self.base_path)
        return sequence_list
