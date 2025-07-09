import os
import torch
import torch.nn as nn
import yaml
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from utils.log_tools import LoggerUnit
from dataset.SCDataset import SCDataset
from model.BitGateNet import BitGateNet
from Preprocess import preprocess_audio_batch

# === Logger setup ===
logger = LoggerUnit("Tester").get_logger()

# === Load config ===
config_path = "config.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

torch.manual_seed(config["seed"])
device = config["device"]

# === Load dataset ===
logger.info(f"Loading test set from {config['dataset']['test_path']}")
test_dataset = SCDataset(config["dataset"]["test_path"], config["labels"])
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=config["batch_size"],
    shuffle=False,
    collate_fn=preprocess_audio_batch
)

# === Model ===
model = BitGateNet(
    num_classes=len(config["labels"]),
    quantscale=config["quant_scale"],
    test=config.get("test", 0)
).to(device)

model_path = os.path.join(config["save_pth"], "best_model_0.8128.pth")
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
logger.info(f"Loaded model from {model_path}")

# === Evaluation ===
criterion = nn.CrossEntropyLoss()
test_loss = 0.0
correct = 0
total = 0
misclassified_samples = []
all_preds = []
all_targets = []

with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        print(f"Processing batch {batch_idx + 1}...")
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

        for i in range(len(inputs)):
            if preds[i] != targets[i]:
                path, _ = test_dataset.samples[batch_idx * config["batch_size"] + i]
                misclassified_samples.append({
                    "filename": path,
                    "pred": config["labels"][preds[i].item()],
                    "true": config["labels"][targets[i].item()]
                })

test_loss /= total
accuracy = correct / total

logger.info(f"Test complete: loss={test_loss:.4f}, acc={accuracy:.4f}")
logger.info(f"Misclassified {len(misclassified_samples)} / {total}")

# Print misclassifications
for item in misclassified_samples:
    print(f"[WRONG] {item['filename']} → pred: {item['pred']}, true: {item['true']}")

# Save misclassified
os.makedirs("logs_misclassified", exist_ok=True)
with open("logs_misclassified/misclassified.json", "w") as f:
    json.dump(misclassified_samples, f, indent=2)
logger.info("Misclassified samples saved.")

# === Metrics ===
print("\n=== Classification Report ===")
print(classification_report(all_targets, all_preds, target_names=config["labels"]))

# === Confusion Matrix ===
cm = confusion_matrix(all_targets, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, xticklabels=config["labels"], yticklabels=config["labels"], fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("logs_misclassified/confusion_matrix.png")
plt.show()

# === Per-Class Accuracy ===
all_preds = np.array(all_preds)
all_targets = np.array(all_targets)
print("\n=== Per-Class Accuracy ===")
for i, label in enumerate(config["labels"]):
    mask = all_targets == i
    class_correct = np.sum((all_preds == i) & mask)
    class_total = np.sum(mask)
    acc = class_correct / class_total if class_total > 0 else 0.0
    print(f"{label:>10}: {acc:.4f}")
