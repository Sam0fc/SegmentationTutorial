#imports
import torch
from torch import nn
import torch.utils.data as TD
import os
from os import path
import torchvision
import torchvision.transforms as T
import torchvision.io as TIO
from typing import Sequence
from torchvision.transforms import functional as F
import numbers 
import random
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import torchmetrics as TM
import pandas as pd

#rename and define working
t2img = T.ToPILImage()
img2t = T.ToTensor()
def trimap2f(trimap):
    return ((trimap) -1) / 2

def concat_h(im1,im2):
    comb = Image.new('RGB',(im1.width + im2.width, im1.height))
    comb.paste(im1,(0,0))
    comb.paste(im2,(im1.width,0))
    return comb

working_dir = "working"
dataset_dir = "Dataset"
labels_file = "annotations/list.txt"
seperator = " "
img_dir = "images"
target_dir = "annotations/trimaps"

full_img_dir = path.join(dataset_dir,img_dir)
full_labels_file = path.join(dataset_dir,labels_file)
full_target_dir = path.join(dataset_dir,target_dir)

#helpers 
def save_model_checkpoint(model, cp_name):
    torch.save(model.state_dict(), os.path.join(working_dir, cp_name))

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def load_model_from_checkpoint(model,cp_path):
    return model.load_state_dict(torch.load(
        cp_path,
        map_location=get_device(),
        )
    )

#for input tensor or model get from device
def to_device(x):
    if torch.cuda.is_available():
        return x.cuda()
    else:
        return x.cpu()

def get_model_parameters(m):
    total_params = sum(
            param.numel() for param in m.parameters()
    )
    return total_params

def print_model_parameters(m):
    num_model_parameters = get_model_parameters(m)
    print(f"Model has {num_model_parameters/1e6:.2f}M parameters")

def close_figures():
    while len(plt.get_fignums()) > 0:
        plt.close()

def args_to_dict(**kwargs):
    return kwargs

print(f"CUDA: {torch.cuda.is_available()}")

## Turning Dataset into pytorch
class CustomImageDataset(TD.Dataset):
    def __init__(self,
                 label_file,
                 target_dir,
                 img_dir, 
                 pre_transform = None, 
                 post_transform = None, 
                 pre_target_transform = None, 
                 post_target_transform = None, 
                 common_transform = None
                 ):


        self.target_dir = target_dir
        self.img_dir = img_dir
        self.pre_transform = pre_transform
        self.post_transform = post_transform
        self.pre_target_transform = pre_target_transform
        self.post_target_transform = post_target_transform
        self.common_transform = common_transform

        self.img_labels = pd.read_csv(label_file,header=6,delimiter=seperator)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self,idx):
        
        #grab the two iamges
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx,0]) + ".jpg"
        image = TIO.read_image(img_path)

        target_path = os.path.join(self.target_dir, self.img_labels.iloc[idx,0]) + ".png"
        target = (TIO.read_image(target_path))

        #apply transforms

        if self.pre_transform:
            image = self.pre_transform(image)
        if self.pre_target_transform:
            target = self.pre_target_transform(target)

        if self.common_transform:
            #magic to combine->transform->split out (4 channel image)
            both = torch.cat([image,target],dim=0)
            both = self.common_transform(both)
            (image, target) = torch.split(both, 3, dim=0)

        if self.post_transform: 
            image = self.post_transform(image)
        if self.post_target_transform:
            target = self.post_target_transform(target)


        return (image.float()/256,target.float())


#define some transforms:
# Torchvision transform to send to a specific device.
class ToDevice(torch.nn.Module):
    def __init__(self,device):
        super().__init__()
        self.device = device

    def forward(self, img):
        return img.to(self.device)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(device={device})"

#convert the trimap to longs with 0 for the pet. (1 bg, 2 border)
def tensor_trimap(t):
    x=t.to(torch.long)
    return x

#composition of some transforms
transform_dict = args_to_dict(
        common_transform = T.Compose([
            ToDevice(get_device()),
            T.Resize((128,128), interpolation=T.InterpolationMode.NEAREST),
            T.RandomHorizontalFlip(p=0.5),
    ]),
        post_transform=T.Compose([
            T.ColorJitter(contrast=0.3),
        ]),
        post_target_transform=T.Compose([
            T.Lambda(tensor_trimap),
        ]),
)

## Fetching Dataset
full_dataset = CustomImageDataset(full_labels_file, full_target_dir, full_img_dir,**transform_dict)

#define loader
dataset_loader = TD.DataLoader(full_dataset,batch_size=64,shuffle=True)
(inputs,targets) = next(iter(dataset_loader))

#spotcheck grid
t2img(torchvision.utils.make_grid(inputs,nrow=8))
#have to convert trimap (0,1,2) to floats from 0->1
t2img(trimap2f(torchvision.utils.make_grid(targets,nrow=8)))



#split data
train_data, test_data = TD.random_split(full_dataset,[0.7,0.3])

pets_train_loader = TD.DataLoader(
    train_data,
    batch_size=32,
    shuffle=True,
)

pets_test_loader = TD.DataLoader(
    test_data,
    batch_size=21,
    shuffle=True,
)

train_pets_inputs, train_pets_targets = next(iter(pets_train_loader))
test_pets_inputs, test_pets_targets = next(iter(pets_test_loader))
train_pets_inputs.shape, train_pets_targets.shape

## THE ACTUAL MODEL!
class DownConv2(nn.Module):
    def __init__(self, chin, chout, kernel_size):
        super().__init__()
        self.seq = nn.Sequential(
                nn.Conv2d(in_channels=chin, out_channels=chout, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
                nn.BatchNorm2d(chout),
                nn.ReLU(),
                nn.Conv2d(in_channels=chout, out_channels=chout, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
                nn.BatchNorm2d(chout),
                nn.ReLU(),
        )
        self.mp=nn.MaxPool2d(kernel_size=2, return_indices=True)

    def forward(self,x):
        y = self.seq(x)
        pool_shape = y.shape
        y, indices = self.mp(y)
        return y, indices, pool_shape

class DownConv3(nn.Module):
    def __init__(self, chin, chout, kernel_size):
        super().__init__()
        self.seq = nn.Sequential(
                nn.Conv2d(in_channels=chin, out_channels=chout, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
                nn.BatchNorm2d(chout),
                nn.ReLU(),
                nn.Conv2d(in_channels=chout, out_channels=chout, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
                nn.BatchNorm2d(chout),
                nn.ReLU(),
                nn.Conv2d(in_channels=chout, out_channels=chout, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
                nn.BatchNorm2d(chout),
                nn.ReLU(),
        )
        self.mp = nn.MaxPool2d(kernel_size=2, return_indices=True)

    def forward(self, x):
        y = self.seq(x)
        pool_shape = y.shape
        y, indices = self.mp(y)
        return y, indices, pool_shape


class UpConv2(nn.Module):
    def __init__(self, chin, chout, kernel_size):
        super().__init__()
        self.seq = nn.Sequential(
                nn.Conv2d(in_channels=chin, out_channels=chin, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
                nn.BatchNorm2d(chin),
                nn.ReLU(),
                nn.Conv2d(in_channels=chin, out_channels=chout, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
                nn.BatchNorm2d(chout),
                nn.ReLU(),
        )
        self.mup = nn.MaxUnpool2d(kernel_size=2)

    def forward(self, x, indices, output_size):
        y = self.mup(x, indices, output_size=output_size)
        y = self.seq(y)
        return y

class UpConv3(nn.Module):
    def __init__(self, chin, chout, kernel_size):
        super().__init__()
        self.seq = nn.Sequential(
                nn.Conv2d(in_channels=chin, out_channels=chin, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
                nn.BatchNorm2d(chin),
                nn.ReLU(),
                nn.Conv2d(in_channels=chin, out_channels=chin, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
                nn.BatchNorm2d(chin),
                nn.ReLU(),
                nn.Conv2d(in_channels=chin, out_channels=chout, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
                nn.BatchNorm2d(chout),
                nn.ReLU(),
        )
        self.mup = nn.MaxUnpool2d(kernel_size=2)
    
    def forward(self, x, indices, output_size):
        y = self.mup(x, indices, output_size=output_size)
        y = self.seq(y)
        return y

class ImageSegmentation(torch.nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.out_channels = 3 
        self.bn_input = nn.BatchNorm2d(3)
        
        self.dc1 = DownConv2(3,64, kernel_size=kernel_size)
        self.dc2 = DownConv2(64,128, kernel_size=kernel_size)
        self.dc3 = DownConv3(128,256, kernel_size=kernel_size)
        self.dc4 = DownConv3(256,512, kernel_size=kernel_size)

        self.uc4 = UpConv3(512,256, kernel_size=kernel_size)
        self.uc3 = UpConv3(256,128, kernel_size=kernel_size)
        self.uc2 = UpConv2(128,64, kernel_size=kernel_size)
        self.uc1 = UpConv2(64,3, kernel_size=kernel_size)

    def forward(self, batch: torch.Tensor):
        x = self.bn_input(batch)

        x, mp1_indices, shape1 = self.dc1(x)
        x, mp2_indices, shape2 = self.dc2(x)
        x, mp3_indices, shape3 = self.dc3(x)
        x, mp4_indices, shape4 = self.dc4(x)

        x = self.uc4(x, mp4_indices, output_size=shape4)
        x = self.uc3(x, mp3_indices, output_size=shape3)
        x = self.uc2(x, mp2_indices, output_size=shape2)
        x = self.uc1(x, mp1_indices, output_size=shape1)

        return x

# DONE

torch.cuda.empty_cache()

#single batch checker
m = ImageSegmentation(kernel_size=3)
m.eval()
to_device(m)
m(to_device(train_pets_inputs)).shape





