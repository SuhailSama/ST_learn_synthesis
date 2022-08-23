# Import Packages
import random 
import numpy as np 
import sys
import os
sys.path.insert(0, 'src')
import matplotlib.pyplot as plt
import importlib
import pickle
from os.path import exists
import copy

# stl stuff for creating stl formulae and calculating robustness
import stlcg
import stlviz as viz
from stlcg import Expression
from utils import print_learning_progress

# for clustering, dimension reduction, feature selection
from random import randint
import pandas as pd
import pickle
import time
import timeit
import scipy
from mpl_toolkits.mplot3d import Axes3D
import regex
from scipy.io import savemat
print(os.getcwd())
## 
import torch
import cv2
import torch.nn as nn 
from torch.utils.data import DataLoader,Dataset, TensorDataset
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
import time
import torch.nn.functional as F
import torch.optim as optim 
import torchvision 

# clustering package
from clustimage import Clustimage


if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu" 
device = torch.device(dev)

"""
import classes and functions
"""
from learn_backend import train_model 
from utils import save2pkl,load_pkl

# train_model(model, criterion, optimizer, scheduler, num_epochs=25):

    
# upload labeled data 
path = "D:\Projects\ST_learn_synthesis\output\\"

# raw_traj = load_pkl(path+"traj_raw")
# print(raw_traj.shape)
# raw_images =  load_pkl(path+"img_raw")
# print(raw_images.shape)

num_traj = 300 
num_times = 40
num_images = num_traj*num_times
traj_raw = np.random.rand(num_traj,64,64,3,num_times)
img_dim = traj_raw.shape[2]
traj_raw = np.transpose(traj_raw, (0,4,1,2,3))
num_traj = traj_raw.shape[0]
num_time_steps = traj_raw.shape[1]
img_raw =  traj_raw.reshape((-1,img_dim,img_dim,3))
img_raw = np.transpose(img_raw, (0,3,1,2))
n_img_clust = 6
n_traj_clust = 5

traj_y = np.random.randint(n_traj_clust, size=(num_traj,1))
img_y  = np.random.randint(n_img_clust, size=(num_images,1))


img_x= torch.Tensor(img_raw) # transform to torch tensor
img_y = torch.Tensor(img_y)

img_dataset = TensorDataset(img_x,img_y) # create your datset
img_loader = DataLoader(img_dataset) # create your dataloader

traj_x= torch.Tensor(traj_raw) # transform to torch tensor
traj_y = torch.Tensor(traj_y)
traj_dataset = TensorDataset(traj_x,traj_y) # create your datset
traj_loader = DataLoader(traj_dataset) # create your dataloader

"""
Spatial learning
""" 

# upload CNN model 
# Importing and training the model
import torchvision.models as models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
VGG16 = models.vgg16(pretrained=True)

for param in VGG16.parameters():
    param.requires_grad = False
num_ftrs = VGG16.classifier[6].out_features
VGG16.classifier.append(nn.Linear(num_ftrs, n_img_clust))
# VGG16.classifier.append(nn.Tanh())
# print(VGG16)

# #model_ft = model_ft.to(device)
criterion = nn.L1Loss()
# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(VGG16.parameters(), lr=0.001, momentum=0.9)
# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

VGG16 = train_model(VGG16,img_loader, criterion, optimizer_ft, exp_lr_scheduler,num_epochs=25)
# torch.save(VGG16,'spatial_classifier.pth')


# temporal Learning (Erfan)




# Save formulae and data




# plot and save performance metrics  

