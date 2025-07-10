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
    device = torch.device(config["device"])
    os.makedirs(config["save_pth"], exist_ok=True)

    # Opt: Save config used for training
    with open(os.path.join(config["save_pth"], "used_config.yaml"), "w") as f_out:
        yaml.safe_dump(config, f_out)

    # Dataset
    train_dataset = SCDataset(config["dataset"]["train_path"], config["labels"])
    val_dataset = SCDataset(config["dataset"]["val_path"], config["labels"])

    # Set worker count based on device
    num_workers = 0 if config["device"] == "cpu" else 4

    # Dynamically build loader kwargs
    loader_kwargs = {
        "batch_size": config["batch_size"],
        "num_workers": num_workers,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
        loader_kwargs["persistent_workers"] = True

    # DataLoaders (with shuffle specified separately)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, **loader_kwargs
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, shuffle=False, **loader_kwargs
    )

    # Model
    model = BitGateNet(
        num_classes=len(config["labels"]),
        quantscale=config["quant_scale"],
        q_en=config["q_en"]
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

    # Training loop with safety
    best_val_loss = float("inf")
    pat = 0
    max_pat = config["patience"]
    final_path = os.path.join(config["save_pth"], f"last_model_epochX.pth")

    try:
        for epoch in range(config["epochs"]):
            logger.info(f"Epoch {epoch+1}/{config['epochs']} | LR: {optimizer.param_groups[0]['lr']:.6f}")

            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device, logger, epoch
            )

            val_loss, val_acc = validate_epoch(
                model, val_loader, criterion, device, logger, epoch, best_val_loss
            )

            # Save best model
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

            # Update last model path for final save
            final_path = os.path.join(config["save_pth"], f"last_model_epoch{epoch+1}.pth")

    finally:
        # Always save final model
        torch.save(model.state_dict(), final_path)
        logger.info(f"Final model saved: {final_path}")
