"""ASVspoof 2019 LA dataset loader."""
import os
import random

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

import audio_utils as A
import config as C

SPLITS = {
    "train": ("ASVspoof2019_LA_train", "ASVspoof2019.LA.cm.train.trn.txt"),
    "dev": ("ASVspoof2019_LA_dev", "ASVspoof2019.LA.cm.dev.trl.txt"),
    "eval": ("ASVspoof2019_LA_eval", "ASVspoof2019.LA.cm.eval.trl.txt"),
}


def read_protocol(root, split):
    """root = the LA folder. Returns a list of (flac path, label, attack id). Label: 0 real, 1 fake."""
    folder, protocol = SPLITS[split]
    items = []
    with open(os.path.join(root, "ASVspoof2019_LA_cm_protocols", protocol)) as f:
        for line in f:
            _speaker, utt, _, attack, key = line.split()
            items.append((os.path.join(root, folder, "flac", utt + ".flac"), 0 if key == "bonafide" else 1, attack))
    return items


class ASVspoofLA(Dataset):
    def __init__(self, root, split, augment=False, mp3_prob=0.3):
        self.items = read_protocol(root, split)
        self.augment, self.mp3_prob = augment, mp3_prob

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label, _ = self.items[i]
        y, sr = sf.read(path, dtype="float32")
        y = A.peak_normalize(A.resample(y, sr))
        if self.augment and random.random() < self.mp3_prob:        # compression
            y = A.mp3_roundtrip(y, C.SAMPLE_RATE, random.choice([64, 96, 128]))
        seg = C.SEGMENT_SECONDS * C.SAMPLE_RATE
        if len(y) < seg:
            y = np.tile(y, int(np.ceil(seg / len(y))))[:seg]
        start = random.randint(0, len(y) - seg) if self.augment else 0
        y = y[start:start + seg]
        if self.augment and random.random() < 0.5:                  # noise
            y = A.add_noise(y, random.uniform(10, 30))
        x = torch.from_numpy(A.log_mel(y)).unsqueeze(0)
        return x, torch.tensor(float(label))
