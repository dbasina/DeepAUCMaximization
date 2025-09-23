
import argparse, os, random, sys, csv, copy
import numpy as np
import pandas as pd
import cv2
import torchvision.transforms as tfs
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
 
# class CheXpert(Dataset):  
#   def __init__(self, images_path, file_path, augment, num_class=5,
#                uncertain_label="LSR-Ones", unknown_label=0, annotation_percent=100):

#     self.img_list = []
#     self.img_label = []
#     self.augment = augment
#     self.train_augment = build_ts_transformations()
    
#     # Define the 5 target diseases and their column indices in the CSV
#     self.disease_columns = {
#         'Cardiomegaly': 7,
#         'Edema': 10,
#         'Consolidation': 11, 
#         'Atelectasis': 13,
#         'Pleural Effusion': 15
#     }
#     self.target_indices = [7, 10, 11, 13, 15]  # Column indices for the 5 diseases
    
#     assert uncertain_label in ["Ones", "Zeros", "LSR-Ones", "LSR-Zeros"]
#     self.uncertain_label = uncertain_label

#     with open(file_path, "r") as fileDescriptor:
#       csvReader = csv.reader(fileDescriptor)
#       next(csvReader, None)
#       for line in csvReader:
#         imagePath = os.path.join(images_path, line[0])
        
#         # Extract only the 5 target disease labels
#         full_labels = line[5:]  # All labels starting from column 5
#         target_labels = []
        
#         for col_idx in self.target_indices:
#           label_idx = col_idx - 5  # Adjust for 0-based indexing after skipping first 5 columns
#           if label_idx < len(full_labels) and full_labels[label_idx]:
#             a = float(full_labels[label_idx])
#             if a == 1:
#               target_labels.append(1)
#             elif a == 0:
#               target_labels.append(0)
#             elif a == -1: # uncertain label
#               if self.uncertain_label == "LSR-Ones":
#                 target_labels.append(random.uniform(0.55, 0.85))
#               elif self.uncertain_label == "LSR-Zeros":
#                 target_labels.append(random.uniform(0, 0.3))
#           else:
#             target_labels.append(unknown_label) # unknown label

#         self.img_list.append(imagePath)
#         self.img_label.append(target_labels)

#     indexes = np.arange(len(self.img_list))
#     if annotation_percent < 100:
#       random.Random(99).shuffle(indexes)
#       num_data = int(indexes.shape[0] * annotation_percent / 100.0)
#       indexes = indexes[:num_data]

#       _img_list, _img_label = copy.deepcopy(self.img_list), copy.deepcopy(self.img_label)
#       self.img_list = []
#       self.img_label = []

#       for i in indexes:
#         self.img_list.append(_img_list[i])
#         self.img_label.append(_img_label[i])

#   def __getitem__(self, index):
#     imagePath = self.img_list[index]
#     try:
#         imageData = Image.open(imagePath).convert('RGB')
#     except (UnidentifiedImageError, OSError):
#         imageData = Image.new('RGB', (320, 320), (0, 0, 0))
#     imageLabel = torch.FloatTensor(self.img_label[index])

#     if self.augment is not None:
#         img = self.augment(imageData)
#     else:
#         img = self.train_augment(imageData)

#     return img, imageLabel

#   def __len__(self):

#     return len(self.img_list)

class CheXpert(Dataset):
    '''
    Reference: 
        @inproceedings{yuan2021robust,
            title={Large-scale Robust Deep AUC Maximization: A New Surrogate Loss and Empirical Studies on Medical Image Classification},
            author={Yuan, Zhuoning and Yan, Yan and Sonka, Milan and Yang, Tianbao},
            booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
            year={2021}
            }
    '''
    def __init__(self, 
                 csv_path, 
                 image_root_path='',
                 image_size=320,
                 class_index=0, 
                 use_frontal=True,
                 use_upsampling=True,
                 flip_label=False,
                 shuffle=True,
                 seed=123,
                 verbose=True,
                 upsampling_cols=['Cardiomegaly', 'Consolidation'],
                 train_cols=['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis',  'Pleural Effusion'],
                 mode='train'):
        
    
        # load data from csv
        self.df = pd.read_csv(csv_path)
        self.df['Path'] = self.df['Path'].str.replace('CheXpert-v1.0-small/', '')
        self.df['Path'] = self.df['Path'].str.replace('CheXpert-v1.0/', '')
        if use_frontal:
            self.df = self.df[self.df['Frontal/Lateral'] == 'Frontal']  
            
        # upsample selected cols
        if use_upsampling:
            assert isinstance(upsampling_cols, list), 'Input should be list!'
            sampled_df_list = []
            for col in upsampling_cols:
                print ('Upsampling %s...'%col)
                sampled_df_list.append(self.df[self.df[col] == 1])
            self.df = pd.concat([self.df] + sampled_df_list, axis=0)


        # impute missing values 
        for col in train_cols:
            if col in ['Edema', 'Atelectasis']:
                self.df[col].replace(-1, 1, inplace=True)  
                self.df[col].fillna(0, inplace=True) 
            elif col in ['Cardiomegaly','Consolidation',  'Pleural Effusion']:
                self.df[col].replace(-1, 0, inplace=True) 
                self.df[col].fillna(0, inplace=True)
            else:
                self.df[col].fillna(0, inplace=True)
        
        self._num_images = len(self.df)
        
        # 0 --> -1
        if flip_label and class_index != -1: # In multi-class mode we disable this option!
            self.df.replace(0, -1, inplace=True)   
            
        # shuffle data
        if shuffle:
            data_index = list(range(self._num_images))
            np.random.seed(seed)
            np.random.shuffle(data_index)
            self.df = self.df.iloc[data_index]
        
        
        assert class_index in [-1, 0, 1, 2, 3, 4], 'Out of selection!'
        assert image_root_path != '', 'You need to pass the correct location for the dataset!'

        if class_index == -1: # 5 classes
            print ('Multi-label mode: True, Number of classes: [%d]'%len(train_cols))
            self.select_cols = train_cols
            self.value_counts_dict = {}
            for class_key, select_col in enumerate(train_cols):
                class_value_counts_dict = self.df[select_col].value_counts().to_dict()
                self.value_counts_dict[class_key] = class_value_counts_dict
        else:       # 1 class
            self.select_cols = [train_cols[class_index]]  # this var determines the number of classes
            self.value_counts_dict = self.df[self.select_cols[0]].value_counts().to_dict()
        
        self.mode = mode
        self.class_index = class_index
        self.image_size = image_size
        
        self._images_list =  [image_root_path+path for path in self.df['Path'].tolist()]
        if class_index != -1:
            self._labels_list = self.df[train_cols].values[:, class_index].tolist()
        else:
            self._labels_list = self.df[train_cols].values.tolist()
    
        if verbose:
            if class_index != -1:
                print ('-'*30)
                if flip_label:
                    self.imratio = self.value_counts_dict[1]/(self.value_counts_dict[-1]+self.value_counts_dict[1])
                    print('Found %s images in total, %s positive images, %s negative images'%(self._num_images, self.value_counts_dict[1], self.value_counts_dict[-1] ))
                    print ('%s(C%s): imbalance ratio is %.4f'%(self.select_cols[0], class_index, self.imratio ))
                else:
                    self.imratio = self.value_counts_dict[1]/(self.value_counts_dict[0]+self.value_counts_dict[1])
                    print('Found %s images in total, %s positive images, %s negative images'%(self._num_images, self.value_counts_dict[1], self.value_counts_dict[0] ))
                    print ('%s(C%s): imbalance ratio is %.4f'%(self.select_cols[0], class_index, self.imratio ))
                print ('-'*30)
            else:
                print ('-'*30)
                imratio_list = []
                for class_key, select_col in enumerate(train_cols):
                    imratio = self.value_counts_dict[class_key][1]/(self.value_counts_dict[class_key][0]+self.value_counts_dict[class_key][1])
                    imratio_list.append(imratio)
                    print('Found %s images in total, %s positive images, %s negative images'%(self._num_images, self.value_counts_dict[class_key][1], self.value_counts_dict[class_key][0] ))
                    print ('%s(C%s): imbalance ratio is %.4f'%(select_col, class_key, imratio ))
                    print ()
                self.imratio = np.mean(imratio_list)
                self.imratio_list = imratio_list
                print ('-'*30)
            
    @property       
    def class_counts(self):
        return self.value_counts_dict
    
    @property
    def imbalance_ratio(self):
        return self.imratio

    @property
    def num_classes(self):
        return len(self.select_cols)
       
    @property  
    def data_size(self):
        return self._num_images 
    
    def image_augmentation(self, image):
        img_aug = tfs.Compose([tfs.RandomAffine(degrees=(-15, 15), translate=(0.05, 0.05), scale=(0.95, 1.05), fill=128)]) # pytorch 3.7: fillcolor --> fill
        image = img_aug(image)
        return image
    
    def __len__(self):
        return self._num_images
    
    def __getitem__(self, idx):

        image = cv2.imread(self._images_list[idx], 0)
        image = Image.fromarray(image)
        if self.mode == 'train':
            image = self.image_augmentation(image)
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # resize and normalize; e.g., ToTensor()
        image = cv2.resize(image, dsize=(self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)  
        image = image/255.0
        __mean__ = np.array([[[0.485, 0.456, 0.406]]])
        __std__ =  np.array([[[0.229, 0.224, 0.225]  ]]) 
        image = (image-__mean__)/__std__
        image = image.transpose((2, 0, 1)).astype(np.float32)
        if self.class_index != -1: # multi-class mode
            label = np.array(self._labels_list[idx]).reshape(-1).astype(np.float32)
        else:
            label = np.array(self._labels_list[idx]).reshape(-1).astype(np.float32)
        return image, label


if __name__ == '__main__':
    root = '../chexpert/dataset/CheXpert-v1.0-small/'
    traindSet = CheXpert(csv_path=root+'train.csv', image_root_path=root, use_upsampling=True, use_frontal=True, image_size=320, mode='train', class_index=0)
    testSet =  CheXpert(csv_path=root+'valid.csv',  image_root_path=root, use_upsampling=False, use_frontal=True, image_size=320, mode='valid', class_index=0)
    trainloader =  torch.utils.data.DataLoader(traindSet, batch_size=32, num_workers=2, drop_last=True, shuffle=True)
    testloader =  torch.utils.data.DataLoader(testSet, batch_size=32, num_workers=2, drop_last=False, shuffle=False)
