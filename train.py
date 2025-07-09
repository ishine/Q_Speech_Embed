import os
import torch
import torch.nn as nn
import yaml

from utils.log_tools import LoggerUnit
from dataset.SCDataset import SCDataset
from model.BitGateNet import BitGateNet
from Preprocess import preprocess_audio_batch

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    logger = LoggerUnit("Trainer").get_logger()
    os.makedirs("logs", exist_ok=True)

    # Load config
    config_path = "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Reproducibility + device setup
    torch.manual_seed(config["seed"])
    device = config["device"]
    os.makedirs(config["save_pth"], exist_ok=True)

    # Dataset setup
    train_dataset = SCDataset(config["dataset"]["train_path"], config["labels"])
    val_dataset = SCDataset(config["dataset"]["val_path"], config["labels"])

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=preprocess_audio_batch
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=preprocess_audio_batch
    )

    # Model init — quantized and configured
    model = BitGateNet(
        num_classes=len(config["labels"]),
        quantscale=config["quant_scale"],
        test=config.get("test", 0)
    ).to(device)

    logger.info(f"Model initialized: {model}")
    total_params = sum(p.numel() for p in model.parameters())
    print("Total parameters:", total_params)
    logger.info(f"Total parameters: {total_params}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config["lr"],
        momentum=config["momentum"],
        weight_decay=1e-4
    )

    best_val_loss = float("inf")
    patience = 0

    for epoch in range(config["epochs"]):
        print(f"\n=== Epoch {epoch+1}/{config['epochs']} ===")
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        print(f"\nEpoch {epoch+1}")
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            print(f"  Batch {batch_idx+1}/{len(train_loader)}", end="\r")
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0 or batch_idx == len(train_loader) - 1:
                print(f"  → Batch {batch_idx+1}: loss={loss.item():.4f}, acc={(outputs.argmax(1) == targets).float().mean():.4f}")


            total_loss += loss.item() * inputs.size(0)
            total_correct += (outputs.argmax(1) == targets).sum().item()
            total_samples += targets.size(0)

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        print(f"  → Training: loss={avg_loss:.4f}, acc={accuracy:.4f}")
        logger.info(f"Epoch {epoch+1}: loss={avg_loss:.4f}, acc={accuracy:.4f}")

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item() * inputs.size(0)
                val_correct += (outputs.argmax(1) == targets).sum().item()
                val_total += targets.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total
        logger.info(f"Validation: loss={val_loss:.4f}, acc={val_acc:.4f} (best: {best_val_loss:.4f})")
        print(f"  → Validation: loss={val_loss:.4f}, acc={val_acc:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 0
            best_model_path = os.path.join(config["save_pth"], f"best_model_{val_acc:.4f}.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved to: {best_model_path}")

            logger.info(f"Best model saved: {best_model_path}")
        else:
            patience += 1
            logger.info(f"No improvement. Patience = {patience}/6")
            if patience >= 6:
                logger.warning("Early stopping.")
                break

    # Save last model
    final_path = os.path.join(config["save_pth"], f"last_model_epoch{epoch+1}.pth")
    torch.save(model.state_dict(), final_path)
    logger.info(f"Final model saved: {final_path}")
