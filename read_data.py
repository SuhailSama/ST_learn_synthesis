# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 17:08:33 2022

@author: suhail

This script has code to load data and visualize movies/photos. 

TODO: 
    - the color map is not correct (e.g., some are RGB other are BGR ,etc..)
    - 
"""
import numpy as np 
import sys
import cv2
import os, glob

import matplotlib.pyplot as plt
import pickle
import mat73
import matplotlib.animation as animation
from itertools import chain
import time 
import pickle as pkl

def save2pkl(fileName,arrayInput):
    
    fileObject = open(fileName, 'wb')
    pkl.dump(arrayInput, fileObject)
    fileObject.close()

def load_pkl(fileName):
    fileObject2 = open(fileName, 'rb')
    modelInput = pkl.load(fileObject2)
    fileObject2.close()
    return modelInput

def plot_canvas(img_arr,num_images = 9):
    fig = plt.figure(figsize=(50, 50))  # width, height in inches

    for i in range(max(num_images,img_arr.shape[0])):
        # print(i)
        sub = fig.add_subplot(3,3, i+1)
        frame = img_arr[i].astype(np.uint8) 
        sub.imshow(frame)

def animate_canvas(vid_arr,classes = [1,2,3],num_vids = 9,num_time_steps = 20):
    names = []
    for i in range(min(num_vids,vid_arr.shape[0])):
        fig, ax = plt.subplots()
        # flatten_axes = list(chain.from_iterable(axes))
        vids = []
        # ax = flatten_axes[i]
        for t in range(num_time_steps):
            vid = ax.imshow(vid_arr[i,:,:,:,t], animated=True)
            if t == 0:
                ax.imshow(vid_arr[i,:,:,:,t])  # show an initial one first
            vids.append([vid])
            # print(i)
        ani = animation.ArtistAnimation(fig, vids, interval=500, blit=True,
                                    repeat_delay=1000)
        plt.show()
        name = ["movie"+str(i)+".mp4"]
        names.append(name)
        print(name[0])
        ani.save(name[0])
    window_titles = names
    cap = [cv2.VideoCapture(i) for i in names]
    frames = [None] * len(names);
    gray = [None] * len(names);
    ret = [None] * len(names);    
    while True:

        for i,c in enumerate(cap):
            if c is not None:
                ret[i], frames[i] = c.read();
        for i,f in enumerate(frames):
            if ret[i] is True:
                gray[i] = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                cv2.imshow(window_titles[i], gray[i]);
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
           break
    
    
    for c in cap:
        if c is not None:
            c.release();
    
    cv2.destroyAllWindows()
    
       
if __name__ == "__main__": 
    
    np.random.default_rng().standard_normal(size=1, dtype='float32')
    """
    upload data 
    """
    path = r"D:/Projects/ST_learn_synthesis/output/" 
    os.makedirs(path, exist_ok=True)
    
    # case = "rand"
    # case = "abm"
    case = "turing1"
    # case = "turing2"
    print("#################################")
    print("Loading data for case  " + case)    
    gen_new = False#input("Generate new data for case "+ case+ "? [True, False]")
    tic = time.time()
    traj_raw_file       = path +case+ '_traj_raw.pkl'
    traj_input_par_file = path +case+ '_traj_input_par.pkl'
    traj_raw            = load_pkl(traj_raw_file) *(1+249* (case=="turing1")) # scale channels to 255
    # traj_input_par      = load_pkl(traj_input_par_file)
    toc = time.time()   
    print("#################################")
    print("Finished loading data in " , toc-tic, " s")
    
    """
    sample data (the datasets are very big)
    """
    print("Sampling data")
    n_samples_traj = 1000
    n_samples_img  = 20000 # set to -1 to process all data
    t1,t2,dt       = 5,55,2 # sample over time from trajectories
        
    num_traj        = traj_raw.shape[0]
    n_samples_traj  = min(n_samples_traj,num_traj)
    index_traj      = np.random.choice(traj_raw.shape[0],
                                       n_samples_traj, replace=True) # sample over images
    traj_X_3d       = traj_raw[index_traj]
    traj_X          = traj_X_3d[:,:,:,:,t1:t2:dt]
    
    if len(traj_raw.shape)==4:
        print("this was one channels dataset")
        traj_X      = np.repeat(traj_X[:, :,:, np.newaxis,:], 3, axis=3)
    
    num_traj,img_h,img_w,img_ch,num_time_steps = traj_X.shape
    n_samples_img   = min(n_samples_img,num_traj*num_time_steps) # for 2 random indices
    img_dim         = [img_h,img_w,img_ch]
    traj_X          = np.transpose(traj_X, (0,4,1,2,3)) # [num_traj,num_time_steps,img_h,img_w,img_ch]
    img_X_3d        =  traj_X.reshape((-1,img_h,img_w,img_ch))
    
    index_img       = np.random.choice(img_X_3d.shape[0],n_samples_img, replace=True)
    img_X_3d        = img_X_3d[index_img]
    img_X           = img_X_3d.reshape((n_samples_img,-1))
    
    print("traj_raw.shape :",traj_raw.shape)
    print("Sampled img_raw.shape : ", img_X.shape)
    
        
    # del traj_raw # uncomment to free space
    # del img_raw
        
    print("#################################")
    print("saved raw data and cleared memory")
    
    """
    Visualize data
    """
    
    img_arr = traj_X[:9,10,:,:,:]
    plot_canvas(img_arr,num_images = 9)
    
    
    vid_arr = traj_raw[:10]
    animate_canvas(vid_arr)
