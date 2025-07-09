import torch


from tqdm import tqdm

def train_epoch(model, train_loader, criterion, optimizer, device, logger, epoch):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    logger.info(f"--- Epoch {epoch+1} (Train) ---")
    
    pbar = tqdm(train_loader, desc=f"[Train] Epoch {epoch+1}", dynamic_ncols=True)

    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # Metrics
        batch_loss = loss.item()
        batch_acc = (outputs.argmax(1) == targets).float().mean().item()

        total_loss += batch_loss * inputs.size(0)
        total_correct += (outputs.argmax(1) == targets).sum().item()
        total_samples += targets.size(0)

        # Update progress bar display
        pbar.set_postfix(loss=f"{batch_loss:.4f}", acc=f"{batch_acc:.4f}")

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    logger.info(f"→ Training: loss={avg_loss:.4f}, acc={accuracy:.4f}")
    return avg_loss, accuracy



@torch.no_grad()
def validate_epoch(model, val_loader, criterion, device, logger, epoch, best_val_loss):
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0

    logger.info(f"--- Epoch {epoch+1} (Validation) ---")

    for inputs, targets in val_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        val_loss += loss.item() * inputs.size(0)
        val_correct += (outputs.argmax(1) == targets).sum().item()
        val_total += targets.size(0)

    val_loss /= val_total
    val_acc = val_correct / val_total
    logger.info(f"→ Validation: loss={val_loss:.4f}, acc={val_acc:.4f} (best: {best_val_loss:.4f})")

    return val_loss, val_acc
