import os
import numpy as np
from torch.utils.data import Dataset
import torch

class SCDataset(Dataset):
    def __init__(self, root, label_list):
        self.samples = []
        self.label_map = {label: idx for idx, label in enumerate(label_list)}

        for label in label_list:
            folder = os.path.join(root, label)
            if not os.path.isdir(folder):
                raise FileNotFoundError(f"Missing folder: {folder}")
            for fname in os.listdir(folder):
                if fname.endswith(".npy"):
                    path = os.path.join(folder, fname)
                    self.samples.append((path, self.label_map[label]))

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            mfcc_array = np.load(path)
        except Exception as e:
            raise IOError(f"Failed to load {path}: {e}")
        mfcc_tensor = torch.from_numpy(mfcc_array).float()
        return mfcc_tensor, label

    def __len__(self):
        return len(self.samples)
