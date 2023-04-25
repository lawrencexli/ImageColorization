"""
General dataset load class for pytorch dataset
"""

from glob import glob
import os

import numpy as np
import torch
from torchvision import transforms

from PIL import Image
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
from sklearn.model_selection import train_test_split

"""
Normalize ab channel from range [-128, 127] to [-1, 1]
"""
def func(x):
    return (2*(x + 128) / 255) - 1

"""
Denormalize ab channel from range [-1, 1] to [-128, 127] 
"""
def ifunc(x):
    return 255*(x+1) / 2 - 128

"""
Build dataset by specifying a dataset folder containing all .jpg images
"""
class Mirflickr:
    
    def __init__(self, data_dir='mirflickr25k', split_size=0.2):
        self.img_list = glob(os.path.join(data_dir, '*.jpg'))
        
        # Use random_state to make the training and evaluation set the same (Needed it for UNet pretrain and GAN train)
        self.train_list, self.eval_list = train_test_split(self.img_list, test_size = split_size, random_state=628)
        
    def build_dataset(self, size=256, batch_size=32):
        train_set = ImageData(self.train_list, ImageTransform(size), 'train')
        eval_set = ImageData(self.eval_list, ImageTransform(size), 'eval')

        train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
        eval_dataloader = torch.utils.data.DataLoader(eval_set, batch_size=1, shuffle=True)
        
        return train_dataloader, eval_dataloader
    

"""
ImageTransform class is called for all images for image transformation. 
Used for data augmentation, input image resizing, and converting images data into tensor floats. 
Note that data augmentation is only used in training data. Not validation and testing data.
"""
class ImageTransform:
    
    def __init__(self, size):
        self.interpolation_mode = transforms.InterpolationMode.BICUBIC
        self.data_transform = {
            'train': transforms.Compose([
                transforms.Resize((size, size), interpolation=self.interpolation_mode),
                transforms.RandomHorizontalFlip()
            ]),
            'eval': transforms.Compose([
                transforms.Resize((size, size), interpolation=self.interpolation_mode)
            ])
        }
        
    def __call__(self, img, phase):
        return self.data_transform[phase](img)
    
    
"""
Get the images via pytorch dataloader class for model training
"""
class ImageData(torch.utils.data.Dataset):
    
    def __init__(self, file_list, transform, mode):
        assert mode == 'train' or mode == 'eval', "[ERROR] mode should be either 'train' or 'eval'"
        self.file_list = file_list
        self.transform = transform
        self.mode = mode
        
    # self has 3 attributes: file_list, transform, and mode
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        img = self.transform(img, self.mode)
        img = np.array(img)
        try:
            img_lab = rgb2lab(img).astype("float32")
            
        except ValueError:
            img = gray2rgb(img)
            img_lab = rgb2lab(img).astype("float32")
            
        img_lab = transforms.ToTensor()(img_lab)
        
        # Extract the color channels
        # Normalize all channels to range [-1, 1]
        L = img_lab[[0], ...] / 50. - 1.
        ab = func(img_lab[[1, 2], ...])
        
        return {'L': L, 'ab': ab}