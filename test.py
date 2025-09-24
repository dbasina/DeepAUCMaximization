import torch
import libauc
from libauc.losses import MultiLabelAUCMLoss
from libauc.optimizers import PESG
import torch.nn as nn
from torch.utils.data import DataLoader
from train_utils import train_one_epoch
import os
from dataloader import CheXpert
from models import create_ensemble
from tqdm import trange,tqdm
from torch.utils.data import Subset
from train_utils import compute_AUC

# Define paths, create necessary directories

test_csv_path = "/scratch/dbasina/CheXpert-v1.0/CheXpert-v1.0/test.csv"
test_images_path = "/scratch/dbasina/CheXpert-v1.0/CheXpert-v1.0/test"
image_root_path = "/scratch/dbasina/CheXpert-v1.0/CheXpert-v1.0"

# Hyperparameters, Devices, Datasets, DataLoaders
batch_size = 4
num_epochs = 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


test_ds = CheXpert(
    csv_path=test_csv_path,
    image_root_path=image_root_path,
    image_size=320,
    mode="valid",
    use_frontal=True,      # set False if test.csv lacks the column
    use_upsampling=False,
)

# Create subset dataloaders for quick testing
subset_test_indices = list(range(10))
test_subset_ds = Subset(test_ds, subset_test_indices)
test_subset_loader = DataLoader(test_subset_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# Define running AUC
bestAUC = 0.0

def test(model, test_loader, device):
    test_log_file = open('./Outputs/logs/test_log.txt', 'w')
    model.eval()
    test_predictions = []
    test_labels = []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            test_predictions.append(outputs.cpu())
            test_labels.append(labels.cpu())

    test_predictions = torch.cat(test_predictions).numpy()
    test_labels = torch.cat(test_labels).numpy()
    test_AUC_mean, per_class = compute_AUC(test_labels, test_predictions)
    test_log_file.write(f"Test AUC: {test_AUC_mean:.4f}\n")
    for i, auc in enumerate(per_class):
        test_log_file.write(f"  Class {i}: AUC = {auc:.4f}\n")

    test_log_file.close()


model = create_ensemble().to(device)
model = nn.DataParallel(model)
checkpoint = torch.load('./Outputs/models/best_model.pth', map_location=device)
model.load_state_dict(checkpoint)
test(model, test_loader, device)