import os
import sys
import torch
import torch.nn.functional as F
import soundfile as sf

# === Path fix ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tiny_speech.TinySpeech import TinySpeechQuantized
from tiny_speech.Preprocess import preprocess_audio_batch

# === Config ===
WAV_PATH = os.path.join("me", "why.wav")  # <- Change this per sample
LABELS = ["yes", "no", "on", "off", "stop", "go", "_unknown_", "_silence_"]
DEVICE = "cpu"
QUANTSCALE = 0.25
MODEL_DIR = "models"

# === Load and fix audio ===
try:
    waveform_np, sr = sf.read(WAV_PATH)
    if waveform_np.ndim > 1:
        waveform_np = waveform_np.mean(axis=1)  # Force mono
    waveform = torch.tensor(waveform_np, dtype=torch.float32).unsqueeze(0)  # [1, T]
except Exception as e:
    print(f"Failed to load WAV: {e}")
    exit(1)

# === Resample manually if needed (Preprocess expects 16kHz)
if sr != 16000:
    import torchaudio
    waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=16000)
    sr = 16000

# === Create fake batch for preprocess_audio_batch
sample_dict = [{
    "audio": {"array": waveform.squeeze(0).numpy(), "sampling_rate": sr},
    "label": 0
}]

mfcc_tensor, _ = preprocess_audio_batch(sample_dict)
print("MFCC tensor shape before model input:", mfcc_tensor.shape)

# [1, 12, T] → [1, 1, 12, T] to match Conv2D input
mfcc = mfcc_tensor.to(DEVICE)

# === Run through all models
model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pth") and ("best_model" in f or f == "last_model_epoch36.pth")]
model_files.sort()

print(f"\n🧠 Inference results for: {WAV_PATH}\n")
for fname in model_files:
    model_path = os.path.join(MODEL_DIR, fname)

    model = TinySpeechQuantized(
        num_classes=len(LABELS),
        quantscale=QUANTSCALE,
        test=0
    ).to(DEVICE)

    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    except Exception as e:
        print(f"{fname:<35} → Load error: {e}")
        continue

    model.eval()

    with torch.no_grad():
        outputs = model(mfcc)
        probs = F.softmax(outputs, dim=1).squeeze()
        pred_idx = outputs.argmax(dim=1).item()
        pred_label = LABELS[pred_idx]
        confidence = probs[pred_idx].item()

    print(f"{fname:<35} → {pred_label:<10} (conf: {confidence:.4f})")
