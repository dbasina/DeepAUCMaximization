import torch
import libauc
from libauc.losses import MultiLabelAUCMLoss
from libauc.optimizers import PESG
import torch.nn as nn
from torch.utils.data import DataLoader
from train_utils import train_one_epoch
import os
from dataloader import CheXpert, build_ts_transformations, build_eval_transformations
from models import create_ensemble
from tqdm import trange,tqdm

# Define paths, create necessary directories
train_csv_path = "./CheXpert/train.csv"
val_csv_path = "./CheXpert/valid.csv"
test_csv_path = "./CheXpert/test.csv"
train_images_path = "./CheXpert/train"
val_images_path = "./CheXpert/valid"
test_images_path = "./CheXpert/test"
os.makedirs("Outputs", exist_ok=True)
os.makedirs("Outputs/models", exist_ok=True)
os.makedirs("Outputs/logs", exist_ok=True)
log_file = open("Outputs/logs/training_log.log", "w")

# Hyperparameters, Devices, Datasets, DataLoaders
batch_size = 4
num_epochs = 20

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = create_ensemble().to(device)
model = nn.DataParallel(model)

# train_ds = CheXpert(train_images_path, train_csv_path, augment=build_ts_transformations())
# val_ds   = CheXpert(val_images_path, val_csv_path,   augment=build_eval_transformations())
# test_ds  = CheXpert(test_images_path, test_csv_path,  augment=build_eval_transformations())

train_ds = CheXpert(
    csv_path="./CheXpert/train.csv",
    image_root_path="./CheXpert/",
    image_size=320,
    use_frontal=True,
    use_upsampling=True,
    mode="train",
    class_index=-1,  # 5-class multilabel
)

val_ds = CheXpert(
    csv_path="./CheXpert/valid.csv",
    image_root_path="./CheXpert/",
    image_size=320,
    use_frontal=True,
    use_upsampling=False,  # IMPORTANT: never upsample val/test
    mode="valid",
    class_index=-1,
)

test_ds = CheXpert(
    csv_path="./CheXpert/test.csv",
    image_root_path="./CheXpert/",
    image_size=320,
    use_frontal=True,
    use_upsampling=False,
    mode="valid",
    class_index=-1,
)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# Define running AUC
bestAUC = 0.0


#Train Stage 1: Using BCEWithLogitsLoss and Adam optimizer for warm-up
stage = 1
loss = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)
scheduler = None
log_file.write(f"Starting Stage {stage} Training\n")
log_file.flush()
for epoch in trange(num_epochs, desc = "Epochs"):
    train_loss, bestAUC = train_one_epoch(train_loader,val_loader, model, optimizer, scheduler, loss, bestAUC, stage, epoch, log_file, device)


#Train Stage 2: Using MultiLabelAUCMLoss and PESG optimizer for finetuning, reset the final fc layer
stage = 2
model.module.fc.reset_parameters()
imratio_list = getattr(train_ds, "imratio_list", None)
loss = MultiLabelAUCMLoss(num_labels=5, device = device, imratio = imratio_list)
optimizer = PESG(model.parameters(),loss_fn = loss, lr=0.1, weight_decay=0)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.1)

log_file.write(f"Starting Stage {stage} Training\n")
log_file.flush()
for epoch in trange(num_epochs, desc="Epochs"):
    train_loss, bestAUC = train_one_epoch(train_loader,val_loader, model, optimizer, scheduler,loss, bestAUC, stage, epoch, device)
    log_file.write(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}\n")

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



