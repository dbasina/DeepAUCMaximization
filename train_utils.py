import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from models import create_ensemble
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

def lr_lambda(current_iter):
    if current_iter < 2000:
        return 1.0
    elif current_iter < 8000:
        return 1/3
    else:
        return 1/9

def compute_AUC(val_labels, val_predictions):
    try:
        per_class = roc_auc_score(val_labels, val_predictions, average=None)
        val_AUC_mean = float(np.nanmean(per_class))  # ignore classes that are NaN
    except ValueError:
        # happens if a class has only one label present; treat as NaN then mean later
        per_class = []
        for c in range(val_labels.shape[1]):
            y = val_labels[:, c]
            p = val_predictions[:, c]
            try:
                per_class.append(roc_auc_score(y, p))
            except ValueError:
                per_class.append(np.nan)
        val_AUC_mean = float(np.nanmean(per_class))
    return val_AUC_mean, per_class

def train_one_epoch(dataloader, val_loader, model, optimizer, scheduler, loss, bestAUC, stage, epoch, log_file, device):
    criterion = loss
    model.train()
    total_loss = 0.0

    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for step, (images, labels) in enumerate(progress_bar):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        if stage == 2:
            scheduler.step()
        total_loss += loss.item() * images.size(0)

        progress_bar.set_postfix({"step": step + 1})

        #validation segment
        if (step + 1) % 50 == 0:
            model.eval()
            val_predictions = []
            val_labels = []
            with torch.no_grad():

                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    val_outputs = model(images)
                    val_predictions.append(torch.sigmoid(val_outputs).cpu())
                    val_labels.append(labels.cpu())
            val_predictions = torch.cat(val_predictions).numpy()
            val_labels = torch.cat(val_labels).numpy()
            val_AUC_mean, per_class = compute_AUC(val_labels, val_predictions)
            
            # Save the best model
            if val_AUC_mean > bestAUC:
                bestAUC = val_AUC_mean
                torch.save(model.state_dict(), f"Outputs/models/best_model.pth")
            
            # Log Results
            log_file.write(f"Stage {stage} Epoch {epoch}, Step {step+1}, Validation AUC: {val_AUC_mean:.4f}, Current Best AUC: {bestAUC:.4f}, Per-Class AUC: {per_class}\n")
            log_file.flush()
            model.train()
        
        #Learning rate scheduler step can be added here if needed
        

    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss, bestAUC