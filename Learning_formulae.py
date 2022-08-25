# Import Packages
import numpy as np 
import sys
import os
sys.path.insert(0, 'src')
import matplotlib.pyplot as plt
import importlib
import pickle

import datetime

import torch
import torch.nn as nn 
from torch.utils.data import DataLoader,TensorDataset
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
import torch.optim as optim 


if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu" 
device = torch.device(dev)

"""
import classes and functions
"""
from learn_backend import train_model 
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
img_dim = traj_raw.shape[1:3]
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
torch.save(VGG16,'spatial_classifier.pth')


"""
temporal Learning (Erfan)
"""

traj_stl = np.zeros([num_traj, num_time_steps,n_img_clust])
for t in range(num_times): 
    X_t = traj_raw[:,t,:,:,:]
    traj_stl[:,t,:] = VGG16(X_t)


# Boosted decision trees 


def BDT_learn (traj_stl,traj_y, args):
    pass


start   = datetime.datetime.now()

formula = BDT_learn (traj_stl,traj_y)

end = datetime.datetime.now()
learning_time = end - start
print('\n learning time: ', learning_time)

# Save formulae and data


from utils import save2pkl 

path = os.getcwd()+"\\output\\"
os.makedirs(path, exist_ok=True)

spatial_pred_file = path + 'sp_pred.pkl'
temp_formula_file = path + 'img_feat.pkl'

save2pkl(spatial_pred_file,VGG16)
save2pkl(temp_formula_file,formula)


traj_stl_file = path + 'traj_stl.pkl'
save2pkl(traj_stl_file,traj_stl)


# plot and save performance metrics  



