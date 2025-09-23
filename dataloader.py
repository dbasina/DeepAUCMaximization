import pandas as pd
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as tfs
import torch
from torch.utils.data import Dataset

FIVE_COLS = ['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion']

class CheXpert(Dataset):
    def __init__(self,
                 csv_path,
                 image_root_path,
                 image_size=320,
                 mode='train',                 # 'train' | 'valid'
                 use_frontal=True,
                 use_upsampling=True,          # True ONLY for train
                 train_cols=FIVE_COLS,
                 upsampling_cols=['Cardiomegaly', 'Consolidation'],
                 verbose=True):
        self.mode = mode
        self.image_size = image_size
        self.select_cols = list(train_cols)

        df = pd.read_csv(csv_path)

        # normalize relative paths
        df['Path'] = (df['Path']
                      .str.replace('CheXpert-v1.0-small/', '', regex=False)
                      .str.replace('CheXpert-v1.0/', '', regex=False))

        # filter frontal views if available
        if use_frontal:
            if 'Frontal/Lateral' in df.columns:
                df = df[df['Frontal/Lateral'].fillna('') == 'Frontal']
            elif 'View Position' in df.columns:
                # accept PA or AP as frontal
                df = df[df['View Position'].fillna('').isin(['PA', 'AP'])]
            else:
                print('[warn] No view column found; proceeding without frontal-only filter.')

        # OPTIONAL: upsample positives for selected columns (train only)
        if self.mode == 'train' and use_upsampling:
            sampled = []
            for col in upsampling_cols:
                if col in df.columns:
                    print(f'Upsampling {col}...')
                    sampled.append(df[df[col] == 1])
            if sampled:
                df = pd.concat([df] + sampled, axis=0, ignore_index=True)

        # impute uncertainties/missing (hard labels; sklearn AUC expects 0/1)
        for col in self.select_cols:
            if col not in df.columns:
                raise ValueError(f'Missing column {col} in CSV.')
            if col in ['Edema', 'Atelectasis']:
                # map -1 -> 1, NaN -> 0
                df.loc[df[col] == -1, col] = 1
                df[col] = df[col].fillna(0)
            elif col in ['Cardiomegaly', 'Consolidation', 'Pleural Effusion']:
                # map -1 -> 0, NaN -> 0
                df.loc[df[col] == -1, col] = 0
                df[col] = df[col].fillna(0)
            else:
                df[col] = df[col].fillna(0)

            # ensure numeric
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(np.float32)

        # compute class ratios (for libauc imratio)
        self.value_counts = {}
        for i, col in enumerate(self.select_cols):
            vc = df[col].value_counts().to_dict()
            zeros = int(vc.get(0.0, 0))
            ones  = int(vc.get(1.0, 0))
            self.value_counts[i] = {0: zeros, 1: ones}

        self.imratio_list = []
        for i in range(len(self.select_cols)):
            zeros, ones = self.value_counts[i][0], self.value_counts[i][1]
            denom = max(1, zeros + ones)
            self.imratio_list.append(ones / denom)

        if verbose:
            print('Multi-label mode: True, Number of classes:', len(self.select_cols))
            print('-' * 30)
            n = len(df)
            for i, col in enumerate(self.select_cols):
                zeros, ones = self.value_counts[i][0], self.value_counts[i][1]
                print(f'Found {n} images in total, {ones} positive, {zeros} negative')
                print(f'{col}(C{i}): imbalance ratio is {ones/max(1,zeros+ones):.4f}\n')
            print('-' * 30)

        self.images = [f'{image_root_path.rstrip("/")}/{p}' for p in df['Path'].tolist()]
        self.labels = df[self.select_cols].values.astype(np.float32)

        # transforms
        self.train_tf = tfs.Compose([
            tfs.RandomAffine(degrees=(-15, 15), translate=(0.05, 0.05), scale=(0.95, 1.05), fill=128),
        ])
        self.normalize = tfs.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # read as grayscale then to RGB
        img = cv2.imread(self.images[idx], cv2.IMREAD_GRAYSCALE)
        if img is None:
            # fallback blank if corrupted/missing
            img = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        pil = Image.fromarray(img)
        if self.mode == 'train':
            pil = self.train_tf(pil)

        arr = np.array(pil)
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        arr = cv2.resize(arr, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        
        arr = (arr / 255.0).astype(np.float32)          # <-- ensure float32
        x = torch.from_numpy(arr.transpose(2, 0, 1)).float()
        x = self.normalize(x)                            # torchvision Normalize (keeps float32)
        y = torch.from_numpy(self.labels[idx]).float()
        return x, y

