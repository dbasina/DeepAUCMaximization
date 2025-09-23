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
            val_AUC_mean = np.mean(roc_auc_score(val_labels, val_predictions, average=None))
            log_file.write(f"Stage {stage} Epoch {epoch}, Step {step+1}, Validation AUC: {val_AUC_mean:.4f}, Current Best AUC: {bestAUC:.4f}\n")
            log_file.flush()
            # Save the best model
            if val_AUC_mean > bestAUC:
                bestAUC = val_AUC_mean
                torch.save(model.state_dict(), f"Outputs/models/best_model.pth")
            model.train()
        
        #Learning rate scheduler step can be added here if needed
        

    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss, bestAUC