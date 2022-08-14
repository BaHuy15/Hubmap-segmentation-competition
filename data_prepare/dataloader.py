import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
from torch.utils.data import Dataset,DataLoader
import os
import cv2
import glob
from torch.autograd import Variable
import albumentations as albu

train_data='data/train/*'
image_dir='data/masks/*'
class Hubmapdataset(Dataset):
    def __init__(self, image_dir, mask_dir,mode):
        #Get image from directory and sort them 
        # join image folder and  image then save in a list 
        #for ex :['../input/hubmaphacking2022-comp-256x256/train/10044_0000.png','__/10044_0001.png'...]
        self.image_dir = sorted(glob.glob('../input/hubmaphacking2022-comp-256x256/train/*'))
        # join mask folder and mask then save in a list 
        #for ex :['../input/hubmaphacking2022-comp-256x256/mask/10044_0000.png','__/10044_0001.png'...]
        self.mask_dir=sorted(glob.glob('../input/hubmaphacking2022-comp-256x256/masks/*'))
        self.mode=mode
        assert mode in ['train','test']
        #Creat mode train and test to do augmentation
        if self.mode=='train':
            self.augmentation=get_training_augmentation()                  
            
    def __getitem__(self, i):
        # read image file
        image = cv2.imread(self.image_dir[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # read mask file
        mask = cv2.imread(self.mask_dir[i],0)
        mask=np.expand_dims(mask, axis=2)
        # apply augmentations
        if self.mode=='train':
            #do augmentation for image and mask
            sample = self.augmentation(image=image, mask=mask)
            image = sample['image'].astype(np.float32)/255
            image=image.transpose(2, 0, 1)
            mask=sample['mask'].astype(np.float32).transpose(2,1,0)
            return torch.tensor(image), torch.tensor(mask)
        elif self.mode=='test':
            image = image.astype(np.float32)/255
            image=image.transpose(2, 0, 1)
            mask=mask.transpose(2,1,0)
            return torch.tensor(image),torch.tensor(mask)
    
    def __len__(self):
        return len(self.image_dir)

def get_training_augmentation(): 
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        #albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,),
        albu.OneOf(
            [
                albu.RandomContrast(p=0.5),
                albu.HueSaturationValue(p=0.5),
            ],
            p=0.9,),
         albu.OneOf([
            albu.OpticalDistortion(p=0.3),
            albu.GridDistortion(p=.1),
            albu.IAAPiecewiseAffine(p=0.3),
        ], p=0.3),]
    return albu.Compose(train_transform)
