import numpy as np
import os
import time

from skimage.io import imread
from torch.utils import data


class YouTubeFacesSetsDataset(data.Dataset):
    def __init__(self, data_dir, split='train', n_frames_per_set=5):
        self.n_frames_per_set = n_frames_per_set
        self.path = data_dir
        splits = {
            'test': slice(100),
            'valid': slice(100, 200),
            'train': slice(200, 1595)
        }
        self.people = os.listdir(self.path)[splits[split]]
        cutoff = None if split == 'train' else 1
        self.videos = [sorted(os.listdir(os.path.join(self.path, person)))[:cutoff]
                       for person in self.people]

        self.n = len(self.people)

    def __getitem__(self, item):
        # person is chosen by iteration
        person = self.people[item]

        # hack to solve multiprocessing rng issue
        seed = int(str(time.time()).split('.')[1])
        np.random.seed(seed=seed)

        # choose a video
        video = np.random.choice(self.videos[item])
        video_path = os.path.join(self.path, person, video)
        all_frames = os.listdir(video_path)

        # choose frames from video
        choice_frames = np.random.choice(all_frames, size=self.n_frames_per_set,
                                         replace=False)

        return np.array([imread(os.path.join(video_path, frame)).transpose(2, 0, 1)
                         for frame in choice_frames]).astype(np.float32) / 255

    def __len__(self):
        return self.n
