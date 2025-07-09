import os
import torch
import torch.nn as nn
import yaml

from dataset import SCDataset
from model import BitGateNet
from utils import LoggerUnit, preprocess_audio_batch
from utils import train_epoch, validate_epoch

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    logger = LoggerUnit("Trainer").get_logger()
    os.makedirs("logs", exist_ok=True)

    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Reproducibility + device setup
    torch.manual_seed(config["seed"])
    device = config["device"]
    os.makedirs(config["save_pth"], exist_ok=True)

    # Dataset
    train_dataset = SCDataset(config["dataset"]["train_path"], config["labels"])
    val_dataset = SCDataset(config["dataset"]["val_path"], config["labels"])

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True,
        num_workers=4, collate_fn=preprocess_audio_batch,
        prefetch_factor=2, persistent_workers=True 
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config["batch_size"], shuffle=False,
        num_workers=4, collate_fn=preprocess_audio_batch,
        prefetch_factor=2, persistent_workers=True
    )
    

    # Model
    model = BitGateNet(
        num_classes=len(config["labels"]),
        quantscale=config["quant_scale"],
    ).to(device)

    logger.info(f"Model initialized: {model}")
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params}")

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config["lr"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"]
    )

    # Training loop
    best_val_loss = float("inf")
    pat = 0
    max_pat= config["patience"]
    for epoch in range(config["epochs"]):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, logger, epoch
        )

        val_loss, val_acc = validate_epoch(
            model, val_loader, criterion, device, logger, epoch, best_val_loss
        )

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            pat = 0
            best_model_path = os.path.join(config["save_pth"], f"b_m_{val_acc:.4f}.pth")
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"New best model saved: {best_model_path}")
        else:
            pat += 1
            logger.info(f"No improvement. Patience = {pat}/{max_pat}")
            if pat >= max_pat:
                logger.warning("Early stopping.")
                break

    # Save last model
    final_path = os.path.join(config["save_pth"], f"last_model_epoch{epoch+1}.pth")
    torch.save(model.state_dict(), final_path)
    logger.info(f"Final model saved: {final_path}")
