import torch
from torch.utils import data
from torchvision import transforms

import numpy as np
import random
from PIL import Image
import glob
import os

class CAGAN_Dataset(data.Dataset):
    def __init__(self, dataroot='data', mode='train', img_size=(256, 192)):
        super(CAGAN_Dataset, self).__init__()
        self.data_dir = os.path.join(dataroot, 'imgs_'+mode)
        self.mode = mode
        self.human_names = self.get_filenames(os.path.join(self.data_dir, '1/*'))
        self.cloth_names = self.get_filenames(os.path.join(self.data_dir, '5/*'))
        self.height, self.width = img_size
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        if self.mode == 'train':
            self.transform = transforms.Compose([
                transforms.RandomCrop(img_size),
            ]) 
            self.random_flip = transforms.RandomHorizontalFlip()
        
    def joint_flip(self, x, y):
        seed = random.randint(0,2**32)
        random.seed(seed)
        x = self.random_flip(x)
        random.seed(seed)
        y = self.random_flip(y)
        return x, y

    def get_filenames(self, file_pattern):
        filenames = glob.glob(file_pattern)
        filenames.sort()
        return filenames
    
    def __len__(self):
        return len(self.human_names)
      
    def __getitem__(self, index):
        h_name = self.human_names[index]
        c1_name = self.cloth_names[index]
        # Load article picture y_j randomly
        temp = self.cloth_names.copy()
        temp.remove(c1_name)
        c2_name = np.random.choice(temp)
        
        img_c1 = Image.open(c1_name).resize((self.width, self.height))
        img_c2 = Image.open(c2_name).resize((self.width, self.height))
        if self.mode == 'train':
            img_h = Image.open(h_name).resize((self.width+30, self.height+40))
            img_h = self.transform(img_h)
            img_h, img_c1 = self.joint_flip(img_h, img_c1)
        else:
            img_h = Image.open(h_name).resize((self.width, self.height))
        xi = self.normalize(img_h)
        yi = self.normalize(img_c1)
        yj = self.normalize(img_c2)
        sample = torch.cat([xi, yi, yj], 0)
        return sample