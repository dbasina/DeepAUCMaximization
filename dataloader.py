
import argparse, os, random, sys, csv, copy
import numpy as np
import os
import torch.distributed as dist
import libauc
from torch.nn.parallel import DistributedDataParallel as DDP

from pathlib import Path
from PIL import Image, ImageFile, UnidentifiedImageError
ImageFile.LOAD_TRUNCATED_IMAGES = True
import timm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import roc_auc_score
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Subset
from libauc.sampler import DualSampler


def build_ts_transformations():
    """Build transformations for CheXpert dataset with hardcoded 320x320 size"""
    from torchvision import transforms
    
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    train_transform = transforms.Compose([
        transforms.Resize((320, 320)),
        # Random rotation, translation, and scaling as used in the reproduced work
        transforms.RandomAffine(
            degrees=15,                    # Random rotation: -15 to +15 degrees
            translate=(0.1, 0.1),         # Random translation: up to 10% of image size
            scale=(0.9, 1.1),             # Random scaling: 90% to 110%
        ),
        transforms.ToTensor(),
        normalize
    ])
    
    return train_transform

def build_eval_transformations():
    """Eval/validation transforms for CheXpert: resize and normalize only"""
    from torchvision import transforms
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    eval_transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        normalize
    ])
    return eval_transform
 
class CheXpert(Dataset):  
  def __init__(self, images_path, file_path, augment, num_class=5,
               uncertain_label="LSR-Ones", unknown_label=0, annotation_percent=100):

    self.img_list = []
    self.img_label = []
    self.augment = augment
    self.train_augment = build_ts_transformations()
    
    # Define the 5 target diseases and their column indices in the CSV
    self.disease_columns = {
        'Cardiomegaly': 7,
        'Edema': 10,
        'Consolidation': 11, 
        'Atelectasis': 13,
        'Pleural Effusion': 15
    }
    self.target_indices = [7, 10, 11, 13, 15]  # Column indices for the 5 diseases
    
    assert uncertain_label in ["Ones", "Zeros", "LSR-Ones", "LSR-Zeros"]
    self.uncertain_label = uncertain_label

    with open(file_path, "r") as fileDescriptor:
      csvReader = csv.reader(fileDescriptor)
      next(csvReader, None)
      for line in csvReader:
        imagePath = os.path.join(images_path, line[0])
        
        # Extract only the 5 target disease labels
        full_labels = line[5:]  # All labels starting from column 5
        target_labels = []
        
        for col_idx in self.target_indices:
          label_idx = col_idx - 5  # Adjust for 0-based indexing after skipping first 5 columns
          if label_idx < len(full_labels) and full_labels[label_idx]:
            a = float(full_labels[label_idx])
            if a == 1:
              target_labels.append(1)
            elif a == 0:
              target_labels.append(0)
            elif a == -1: # uncertain label
              if self.uncertain_label == "LSR-Ones":
                target_labels.append(random.uniform(0.55, 0.85))
              elif self.uncertain_label == "LSR-Zeros":
                target_labels.append(random.uniform(0, 0.3))
          else:
            target_labels.append(unknown_label) # unknown label

        self.img_list.append(imagePath)
        self.img_label.append(target_labels)

    indexes = np.arange(len(self.img_list))
    if annotation_percent < 100:
      random.Random(99).shuffle(indexes)
      num_data = int(indexes.shape[0] * annotation_percent / 100.0)
      indexes = indexes[:num_data]

      _img_list, _img_label = copy.deepcopy(self.img_list), copy.deepcopy(self.img_label)
      self.img_list = []
      self.img_label = []

      for i in indexes:
        self.img_list.append(_img_list[i])
        self.img_label.append(_img_label[i])

  def __getitem__(self, index):
    imagePath = self.img_list[index]
    try:
        imageData = Image.open(imagePath).convert('RGB')
    except (UnidentifiedImageError, OSError):
        imageData = Image.new('RGB', (320, 320), (0, 0, 0))
    imageLabel = torch.FloatTensor(self.img_label[index])

    if self.augment is not None:
        img = self.augment(imageData)
    else:
        img = self.train_augment(imageData)

    return img, imageLabel

  def __len__(self):

    return len(self.img_list)
