import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from models import create_ensemble
import numpy as np
from libauc.metrics import auc_roc_score
from tqdm import tqdm


def train_one_epoch(dataloader, val_loader,test_loader, model, optimizer, loss_fn, bestAUC, stage, epoch, log_file, device):
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for step, (images, labels) in enumerate(progress_bar):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        outputs = torch.sigmoid(outputs)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        progress_bar.set_postfix({"step": step + 1})

        #validation segment
        if (step + 1) % 50 == 0:

            val_mAUC = evaluate_AUC(model, val_loader, device)
            test_mAUC = evaluate_AUC(model, test_loader, device)
            # Save the best model
            if val_mAUC > bestAUC:
                bestAUC = val_mAUC
                torch.save(model.state_dict(), f"Outputs/models/best_model.pth")
                
            # Log Results
            log_file.write(f"Stage {stage}, Epoch {epoch}, Step {step+1}, Val AUC: {val_mAUC:.4f}, Test AUC: {test_mAUC:.4f}, Best Val AUC: {bestAUC:.4f}\n")
            log_file.flush()
            model.train()
        
        #Learning rate scheduler step can be added here if needed
        

    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss, bestAUC

def evaluate_AUC(model, loader, device):
    model.eval()
    eval_predictions = []
    eval_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            eval_predictions.append(torch.sigmoid(outputs).cpu())
            eval_labels.append(labels.cpu())
    eval_predictions = torch.cat(eval_predictions).numpy()
    eval_labels = torch.cat(eval_labels).numpy()
    AUC_mean = np.mean(auc_roc_score(eval_labels, eval_predictions))
    return AUC_mean