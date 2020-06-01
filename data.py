"""
Dataset and DataLoaders for Airbus Ship Detection Challenge
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
from skimage.io import imread
import numpy as np
from .utility import rle_decode
from sklearn.model_selection import train_test_split
from torchvision.transforms import ToTensor, Compose

def masks_as_img(masks):
    combined_masks = np.zeros((768, 768), dtype=np.float)
    for mask in masks:
        if isinstance(mask, str):
            combined_masks += rle_decode(mask, (768, 768))
    return np.expand_dims(combined_masks, -1)

class ShipDataset(Dataset):

    def __init__(self, df, mode='train'):
        grp = list(df.groupby('ImageId'))
        self.image_ids = [id for id, _ in grp]
        self.image_masks = [m['EncodedPixels'].values for _, m in grp]
        self.img_transform = Compose([
          ToTensor()
        ])
        if mode == 'train':
            self.image_path = 'train_v2'
        else:
            self.image_path = 'test_v2'

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_file_name = self.image_ids[idx]
        img = imread(f'../input/airbus-ship-detection/{self.image_path}/{img_file_name}')
        img = img.astype(float)
        mask = masks_as_img(self.image_masks[idx])
        return self.img_transform(img), torch.from_numpy(np.moveaxis(mask, -1, 0)).float()


def get_datasets(top=None):
    """
    Get an instance of training and test dataset
    """
    df = pd.read_csv('../input/airbus-ship-detection/train_ship_segmentations_v2.csv')
    unique_image_ids = df.groupby('ImageId').size().reset_index(name='counts')
    stratification = unique_image_ids['counts']

    if top is not None:
        unique_image_ids = unique_image_ids.head(top)
        stratification = None

    train_ids, valid_ids = train_test_split(unique_image_ids, test_size=0.2, stratify=stratification)
    train_df = pd.merge(df, train_ids)
    valid_df = pd.merge(df, valid_ids)

    ds_train = ShipDataset(train_df)
    ds_valid = ShipDataset(valid_df)

    return ds_train, ds_valid
