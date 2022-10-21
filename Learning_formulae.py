# Import Packages
import numpy as np 
import sys
import os
sys.path.insert(0, 'src')
import matplotlib.pyplot as plt
import importlib
import pickle

import time

import torch
import torch.nn as nn 
from torch.utils.data import DataLoader,TensorDataset
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
import torch.optim as optim 

import pandas as pd

from utils import load_pkl, save2pkl

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
import classes and functions
"""
from learn_backend import train_model 
# train_model(model, criterion, optimizer, scheduler, num_epochs=25):

    
# upload labeled data 

# raw_traj = load_pkl(path+"traj_raw")
# print(raw_traj.shape)
# raw_images =  load_pkl(path+"img_raw")
# print(raw_images.shape)


# case = "rand"
# case = "rand"
case = "abm"
# case = "turing1"
# case = "turing2"

print("#################################")
print("Loading data for case  " + case)
cwd = os.getcwd()

path = r"D:/Projects/ST_learn_synthesis/output/"


tic = time.time()

if case == "rand": 
        
    num_traj = 300 
    num_times = 40
    num_images = num_traj*num_times
    traj_X = np.random.rand(num_traj,64,64,3,num_times)
    img_dim = traj_X.shape[1:3]
    traj_X = np.transpose(traj_X, (0,4,1,2,3))
    num_traj = traj_X.shape[0]
    num_time_steps = traj_X.shape[1]
    img_X=  traj_X.reshape((-1,img_dim[0],img_dim[1],3))
    img_X = np.transpose(img_X, (0,3,1,2))
    
    n_img_clust = 6
    n_traj_clust = 5
    
    traj_y = np.random.randint(n_traj_clust, size=(num_traj,1))
    img_y  = np.random.randint(n_img_clust, size=(num_images,1))

else: 
    os.chdir(path)
    img_index_file  = path +case+ '_img_in.pkl'
    img_X_file      = path +case+ '_img_X.pkl'
    img_feat_file   = path +case+ '_img_feat.pkl'
    img_y_file      = path +case+ '_img_y.pkl'

    traj_index_file = path +case+ '_traj_ind.pkl'
    traj_X_file     = path +case+ '_traj_X.pkl'
    traj_feat_file  = path +case+ '_traj_feat.pkl'
    traj_y_file     = path +case+ '_traj_y.pkl'

    # index_img   = load_pkl(img_index_file)
    img_X       = load_pkl(img_X_file)
    img_feat    = load_pkl(img_feat_file)
    img_y       = load_pkl(img_y_file)

    index_traj  = load_pkl(traj_index_file)
    traj_X      = load_pkl(traj_X_file)
    traj_feat   = load_pkl(traj_feat_file)
    traj_y      = load_pkl(traj_y_file)
  
n_img_clust = img_y.max().item()+1
shape  = (img_y.shape[0], n_img_clust)
img_one_hot = -np.ones(shape)
rows = np.arange(img_y.shape[0])
img_one_hot[rows, img_y] = 1

n_traj_clust = traj_y.max()+1
shape  = (traj_y.shape[0],n_traj_clust )
traj_one_hot = -np.ones(shape)
rows = np.arange(traj_y.shape[0])
traj_one_hot[rows, traj_y] = 1

toc = time.time()

os.chdir(cwd)
print("#################################")
print("Finished loading data in " , toc-tic, " s")
print("traj_raw.shape : ", traj_X.shape)

# input("continue?")
# traj_raw = np.transpose(traj_raw, (0,4,1,2,3))

img_x= torch.Tensor(np.transpose(img_X, (0,3,1,2))).to(device) # transform to torch tensor
img_y = torch.Tensor(img_one_hot).to(device)
img_dataset = TensorDataset(img_x,img_y) # create your datset
img_loader = DataLoader(img_dataset,batch_size=1000) # create your dataloader

traj_x= torch.Tensor(traj_X) # transform to torch tensor
traj_y = torch.Tensor(traj_one_hot)
traj_dataset = TensorDataset(traj_x,traj_y) # create your datset
traj_loader = DataLoader(traj_dataset,batch_size=100) # create your dataloader

"""
Spatial learning
""" 

# upload CNN model 
# Importing and training the model
import torchvision.models as models

vgg16_file = path +case+ 'spatial_classifier.pkl'

is_new_model = True #input("Train new model? [True, False]")
if is_new_model: 
    VGG16 = models.vgg16(pretrained=True)
    
    for param in VGG16.parameters():
        param.requires_grad = False
        
    num_ftrs = VGG16.classifier[6].out_features
    VGG16.classifier.append(nn.Linear(num_ftrs, n_img_clust))
    VGG16.classifier.append(nn.Tanh())
    
    VGG16.to(device)
else:
    VGG16 = load_pkl(vgg16_file)
    
print(VGG16)
# #model_ft = model_ft.to(device)
criterion = nn.L1Loss()
# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(VGG16.parameters(), lr=0.001, momentum=0.9)
# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

VGG16 = train_model(VGG16,img_loader, criterion, optimizer_ft, exp_lr_scheduler,
                    num_epochs=20)


save2pkl(vgg16_file, VGG16)

"""
temporal Learning (Erfan)
"""
if len(traj_X.shape)==4:
    print("this was one channels dataset")
    traj_X      = np.repeat(traj_X[:, :,:, np.newaxis,:], 3, axis=3)
    
# traj_X = torch.Tensor(np.transpose(traj_X, (0,4,3,1,2))).to(device) # transform to torch tensor

num_traj, num_time_steps = traj_X.shape[:2]
traj_stl = torch.zeros([num_traj, num_time_steps,n_img_clust]).to(device)
for t in range(num_time_steps): 
    X_t = traj_X[:,t,:,:,:]
    traj_stl[:,t,:] = VGG16(X_t)

# Boosted decision trees 
# def BDT_learn (traj_stl,traj_y, args):
#     pass

# start   = time.time()

# formula = BDT_learn (traj_stl,traj_y)

# end = time.time()
# learning_time = end - start
# print('\n learning time: ', learning_time)

# Save formulae and data



spatial_pred_file = path + case +'_sp_pred.pkl'
st_traj_file     = path +case+ '_st_traj.pkl'

save2pkl(spatial_pred_file,VGG16)
save2pkl(st_traj_file,traj_stl)


# traj_stl_file = path + 'traj_stl.pkl'
# save2pkl(traj_stl_file,traj_stl)


# plot and save performance metrics  
import itertools
fig, axs = plt.subplots(3, 2)
axs = list(itertools.chain(*axs))
xt = np.arange(5,5+traj_stl.shape[1])

LINE_STYLES = ['solid','solid','solid','solid','solid', 
               'dashed','dashed','dashed','dashed','dashed']
NUM_STYLES = len(LINE_STYLES)
clrs = ['r','b','k','g','c','r','b','k','g','c']

for i in range(6): # spatial classes
    ax = axs[i]
    ax.set_title('')
    for ii in range(10): # temporal classes
        y = traj_stl[traj_y[:,ii]==1,:,i].cpu().detach().numpy()
        x = np.repeat(xt[np.newaxis,:], y.shape[0], axis=0)
        ax.plot(x[1],y[1])
        ax.set_prop_cycle(clrs[i])
        ax.set_linestyle(LINE_STYLES[i%NUM_STYLES])
        



