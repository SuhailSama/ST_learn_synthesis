""" 
Import Packages
"""
import random 
import numpy as np 
import sys
import os
sys.path.insert(0, 'src')
import matplotlib.pyplot as plt
import importlib
import platform
import subprocess
import pickle
from os.path import exists

# stl stuff for creating stl formulae and calculating robustness
import torch
import stlcg
import stlviz as viz
from stlcg import Expression
from utils import print_learning_progress

# for clustering, dimension reduction, feature selection
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
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

# 
from keras.preprocessing.image import load_img 
from keras.preprocessing.image import img_to_array 
from keras.applications.vgg16 import preprocess_input 
from keras.models import Model
import tensorflow as tf
print(tf.__version__)

if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu" 
device = torch.device(dev)
print("Running torch on: ", torch.cuda.get_device_name(0))


"""
import classes and functions
"""

from opt_backend import Particle, Space


"""
load spatial predicate
"""

# load pre-trained CNN model



"""
define specification formula
"""

trace_len = 5
x = torch.tensor(np.random.rand(trace_len,1) ).float() 
y = torch.tensor(np.random.rand(trace_len,1) ).float() 
# define the stl formula 
a = torch.as_tensor(0.48).float()
b = torch.as_tensor(0.52).float()
x_exp = stlcg.Expression('x_exp',x)
y_exp = stlcg.Expression('y_exp',y)
ϕ1 = x_exp >a
ϕ2 = y_exp <b
formula = stlcg.Always(subformula=ϕ1& ϕ2, interval=[2,3])
print('formula is :' ,formula)
#visualize the formula
# viz.make_stl_graph(formula)


""" 
hyper parameters
"""
n_iterations = 5    # int(input("Inform the number of iterations: "))
target_error = 0.1   # float(input("Inform the target error: "))
n_particles = 4      # int(input("Inform the number of particles: "))
target = 1           # 
simulation_time = 500*(trace_len-1)
search_space = Space(target, target_error, n_particles, simulation_time,CNN_model,SVM_model, selected_features,formula)
particles_vector = [Particle() for _ in range(search_space.n_particles)]
search_space.particles = particles_vector
search_space.print_particles()



"""
MAIN optimization loop 

"""
iteration = 0
start_time = time.time()
while(True):
    print("########################################################### ")
    print("############# Iteration # ",iteration,"####################" )
    print("########################################################### ")
    iter_time = time.time()
    print("#############  finding best local ####################" )
    search_space.print_particles()
    search_space.set_pbest()  
    print('positions after move')
    search_space.print_particles()
    print("#############  finding best global ####################" )
    current_best_rob, sim_id = search_space.set_gbest()
    if(abs(search_space.gbest_value - search_space.target) <= search_space.target_error):
        break
    print("#############  iteration result ####################" )
    print("run time = " ,(time.time() - iter_time),",global best solution: ",current_best_rob,"from sim",sim_id)
    search_space.move_particles()
    if iteration < n_iterations:
        iteration += 1
    else: 
        break 

print("Total time %s seconds ---" % (time.time() - start_time))
print("The best solution is: ", search_space.gbest_position, " in n_iterations: ", iteration)


# Save formulae and data



# plot and save performance metrics  



