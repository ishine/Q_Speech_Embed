import torchaudio
import torch
import torch.nn.functional as F

def convert_to_contiguous(lst):
    return {value: index for index, value in enumerate(sorted(set(lst)))}

def preprocess_audio_batch(batch):
    audio_arrays = [example["audio"]["array"] for example in batch]
    sampling_rates = [example["audio"]["sampling_rate"] for example in batch]
    labels = [example["label"] for example in batch]

    batch_mfccs = []

    for audio, sr in zip(audio_arrays, sampling_rates):
        # Check if input is already MFCC.
        tensor = torch.tensor(audio, dtype=torch.float32)
        if tensor.dim() == 3 and tensor.shape[0] == 1 and tensor.shape[1] == 12:
            # Already MFCC → skip processing
            mfccs = tensor
        else:
            # Treat as raw waveform
            waveform = tensor

            if sr != 16000:
                waveform = waveform.unsqueeze(0)
                waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=16000)
                waveform = waveform.squeeze(0)
                sr = 16000

            target_length = sr
            if waveform.size(0) > target_length:
                waveform = waveform[:target_length]
            else:
                waveform = F.pad(waveform, (0, target_length - waveform.size(0)))

            waveform = waveform.unsqueeze(0)

            hop_length = int(sr * 0.01)
            win_length = min(int(sr * 0.03), 1024)

            mfccs = torchaudio.transforms.MFCC(
                sample_rate=sr,
                n_mfcc=12,
                melkwargs={
                    "n_fft": 1024,
                    "hop_length": hop_length,
                    "win_length": win_length,
                    "n_mels": 23,
                    "center": False
                }
            )(waveform)

        batch_mfccs.append(mfccs)

    batch_mfccs = torch.stack(batch_mfccs)
    labels = torch.tensor(labels, dtype=torch.long)

    return batch_mfccs, labels
