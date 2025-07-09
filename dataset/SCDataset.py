import os
import numpy as np
import torchaudio
from torch.utils.data import Dataset

class SCDataset(Dataset):
    def __init__(self, root, label_list):
        self.samples = []
        self.label_map = {label: idx for idx, label in enumerate(label_list)}

        for label in label_list:
            folder = os.path.join(root, label)
            if not os.path.isdir(folder):
                raise ValueError(f"Missing folder: {folder}")
            for fname in os.listdir(folder):
                if fname.endswith(".wav") or fname.endswith(".npy"):
                    path = os.path.join(folder, fname)
                    self.samples.append((path, self.label_map[label]))

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        ext = os.path.splitext(path)[1].lower()

        if ext == ".wav":
            waveform, sr = torchaudio.load(path)
            waveform = waveform.mean(dim=0).numpy()
            return {
                "audio": {"array": waveform, "sampling_rate": sr},
                "label": label,
                "filename": os.path.basename(path)
            }

        elif ext == ".npy":
            mfcc_array = np.load(path)
            return {
                "audio": {"array": mfcc_array, "sampling_rate": 16000},  # fake sr for consistency
                "label": label,
                "filename": os.path.basename(path)
            }

        else:
            raise ValueError(f"Unsupported file type: {path}")

    def __len__(self):
        return len(self.samples)
