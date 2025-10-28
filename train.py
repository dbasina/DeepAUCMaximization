import torch
import libauc
import timm
from libauc.losses import MultiLabelAUCMLoss, AUCMLoss
from libauc.optimizers import PESG
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from train_utils import train_one_epoch
import os
from dataset import CheXpert
from models import create_ensemble
from tqdm import trange,tqdm
from torch.utils.data import Subset
from torch.optim.lr_scheduler import LambdaLR

# Define paths, create necessary directories

train_csv_path = "./CheXpert-mini/train.csv"
val_csv_path = "./CheXpert-mini/valid.csv"
test_csv_path = "./CheXpert-mini/test.csv"
train_images_path = "./CheXpert-mini/train"
val_images_path = "./CheXpert-mini/valid"
test_images_path = "./CheXpert-mini/test"
image_root_path = "./CheXpert-mini/"

# train_csv_path = "/scratch/dbasina/CheXpert-v1.0/CheXpert-v1.0/train.csv"
# val_csv_path = "/scratch/dbasina/CheXpert-v1.0/CheXpert-v1.0/valid.csv"
# test_csv_path = "/scratch/dbasina/CheXpert-v1.0/CheXpert-v1.0/test.csv"
# train_images_path = "/scratch/dbasina/CheXpert-v1.0/CheXpert-v1.0/train"
# val_images_path = "/scratch/dbasina/CheXpert-v1.0/CheXpert-v1.0/valid"
# test_images_path = "/scratch/dbasina/CheXpert-v1.0/CheXpert-v1.0/test"
# image_root_path = "/scratch/dbasina/CheXpert-v1.0/CheXpert-v1.0"
os.makedirs("Outputs", exist_ok=True)
os.makedirs("Outputs/models", exist_ok=True)
os.makedirs("Outputs/logs", exist_ok=True)
log_file = open("Outputs/logs/training_log.log", "w")

# Hyperparameters, Devices, Datasets, DataLoaders
batch_size = 4
num_epochs = 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Checkpoint loading (if any)
# checkpoint = torch.load('./Outputs/models/run2_best_model.pth', map_location=device)
# model.load_state_dict(checkpoint)


train_ds = CheXpert(
    csv_path=train_csv_path,
    image_root_path=image_root_path,
    image_size=320,
    mode="train",
    use_frontal=True,
    use_upsampling=True,
    class_index=-1,
)
val_ds = CheXpert(
    csv_path=val_csv_path,
    image_root_path=image_root_path,
    image_size=320,
    mode="valid",
    use_frontal=True,      # set False if valid.csv lacks the column
    use_upsampling=False,
    class_index=-1,
)
test_ds = CheXpert(
    csv_path=test_csv_path,
    image_root_path=image_root_path,
    image_size=320,
    mode="valid",
    use_frontal=False,      # set False if test.csv lacks the column
    use_upsampling=False,
    class_index=-1,
)

subset_train_indices = list(range(100))
subset_val_indices = list(range(50))

train_subset_ds = Subset(train_ds, subset_train_indices)
train_subset_loader = DataLoader(train_subset_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_subset_ds = Subset(val_ds, subset_val_indices)
val_subset_loader = DataLoader(val_subset_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# Define running AUC
bestAUC = 0.0


model = timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes = 5, in_chans = 3)
model = model.to(device)
model = nn.DataParallel(model)

# # Train Stage 1: Using BCEWithLogitsLoss and Adam optimizer for initial training
# stage = 1
# imratio_list = getattr(train_ds, "imratio_list", None)
# # loss = nn.BCEWithLogitsLoss()
# loss = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
# log_file.write(f"Starting Stage {stage} Training\n")
# log_file.flush()
# for epoch in trange(num_epochs, desc = "Epochs"):
#     train_loss, bestAUC = train_one_epoch(train_subset_loader, val_subset_loader, model, optimizer, loss, bestAUC, stage, epoch, log_file, device, milestones)
#     # train_loss, bestAUC = train_one_epoch(train_loader,test_loader, model, optimizer, loss, bestAUC, stage, epoch, log_file, device, milestones)


#Train Stage 2: Using MultiLabelAUCMLoss and PESG optimizer for finetuning, reset the final fc layer
stage = 2
imratio_list = getattr(train_ds, "imratio_list", None)
learning_rate = 0.1
weight_decay = 1e-5
epoch_decay = 2e-3
margin = 1.0

# loss = AUCMLoss(num_labels=5, device = device)

loss = MultiLabelAUCMLoss(num_labels=5, device = device, imratio = imratio_list)
optimizer = PESG(model.parameters(),loss_fn = loss, lr=learning_rate, margin=margin, weight_decay=weight_decay, epoch_decay = epoch_decay, device=device)
log_file.write(f"Starting Stage {stage} Training\n")
log_file.flush()
for epoch in trange(num_epochs, desc="Epochs"):
    # train_loss, bestAUC = train_one_epoch(train_subset_loader,val_subset_loader, model, optimizer, loss, bestAUC, stage, epoch, log_file, device)
    train_loss, bestAUC = train_one_epoch(train_loader,val_loader,test_loader, model, optimizer, loss, bestAUC, stage, epoch, log_file, device)
    optimizer.update_regularizer(decay_factor=10)
        

# Save the final model
torch.save(model.state_dict(), "Outputs/models/final_model.pth")

# Final log statements
log_file.write(f"Training complete. Best AUC: {bestAUC:.4f}\n")
log_file.write("Model saved to Outputs/models/final_model.pth\n")
log_file.flush()


# # Evaluate on the test set
# test(model, test_loader, device)

# Close the log file
log_file.close()



