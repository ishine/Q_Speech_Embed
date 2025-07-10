import time
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
        batch_start = time.time()

        if device.type != "cpu":
            inputs, targets = inputs.to(device), targets.to(device)

        # # Profile and exit after batch 0, but update stats before exiting
        # if batch_idx == 0:
        #     with torch.autograd.profiler.profile(use_cpu=True) as prof:
        #         outputs = model(inputs)
        #         loss = criterion(outputs, targets)
        #     print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=30))
        #
        #     batch_loss = loss.item()
        #     batch_acc = (outputs.argmax(1) == targets).float().mean().item()
        #
        #     total_loss += batch_loss * inputs.size(0)
        #     total_correct += (outputs.argmax(1) == targets).sum().item()
        #     total_samples += targets.size(0)
        #
        #     pbar.set_postfix(loss=f"{batch_loss:.4f}", acc=f"{batch_acc:.4f}")
        #     logger.info(f"[Profiler Timing] Batch 1 took {time.time() - batch_start:.3f} seconds")
        #     logger.info(f"[Profiler Batch] loss={batch_loss:.4f}, acc={batch_acc:.4f}")
        #     break

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        batch_loss = loss.item()
        batch_acc = (outputs.argmax(1) == targets).float().mean().item()

        total_loss += batch_loss * inputs.size(0)
        total_correct += (outputs.argmax(1) == targets).sum().item()
        total_samples += targets.size(0)

        pbar.set_postfix(loss=f"{batch_loss:.4f}", acc=f"{batch_acc:.4f}")

        #batch_end = time.time()
        #if batch_idx < 5:
        #    logger.info(f"[Timing] Batch {batch_idx+1} took {batch_end - batch_start:.3f} seconds")

    # Prevent division by zero
    if total_samples > 0:
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
    else:
        avg_loss, accuracy = 0.0, 0.0

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
        if device.type != "cpu":
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
