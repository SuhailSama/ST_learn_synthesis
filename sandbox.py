# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 18:11:00 2022

@author: suhai
"""

import numpy as np
import matplotlib.pyplot as plt

def discrete_laplacian(M):
    """Get the discrete Laplacian of matrix M"""
    L = -4*M
    L += np.roll(M, (0,-1), (0,1)) # right neighbor
    L += np.roll(M, (0,+1), (0,1)) # left neighbor
    L += np.roll(M, (-1,0), (0,1)) # top neighbor
    L += np.roll(M, (+1,0), (0,1)) # bottom neighbor
    
    return L

def gray_scott_update(A, B, DA, DB, f, k, delta_t):
    """
    Updates a concentration configuration according to a Gray-Scott model
    with diffusion coefficients DA and DB, as well as feed rate f and
    kill rate k.
    """
    
    # Let's get the discrete Laplacians first
    LA = discrete_laplacian(A)
    LB = discrete_laplacian(B)
    
    # Now apply the update formula
    diff_A = (DA*LA - A*B**2 + f*(1-A)) * delta_t
    diff_B = (DB*LB + A*B**2 - (k+f)*B) * delta_t
    
    A += diff_A
    B += diff_B
    
    return A, B

def get_initial_configuration(N, random_influence=0.2):
    """
    Initialize a concentration configuration. N is the side length
    of the (N x N)-sized grid.
    `random_influence` describes how much noise is added.
    """
    
    # We start with a configuration where on every grid cell 
    # there's a lot of chemical A, so the concentration is high
    A = (1-random_influence) * np.ones((N,N)) + random_influence * np.random.random((N,N))
    
    # Let's assume there's only a bit of B everywhere
    B = random_influence * np.random.random((N,N))
    
    # Now let's add a disturbance in the center
    N2 = N//2
    radius = r = int(N/10.0)
    
    A[N2-r:N2+r, N2-r:N2+r] = 0.50
    B[N2-r:N2+r, N2-r:N2+r] = 0.25
    
    return A, B


def draw(A,B):
    """draw the concentrations"""
    fig, ax = plt.subplots(1,2,figsize=(5.65,4))
    ax[0].imshow(A, cmap='Greys')
    ax[1].imshow(B, cmap='Greys')
    ax[0].set_title('A')
    ax[1].set_title('B')
    ax[0].axis('off')
    ax[1].axis('off')
    


def turing_model(parameters, is_plot = True):
    """
    parameters: 
        DA - ,
        DB - , 
        f  - , 
        k  - . 
    """
    # DA, DB, f, k = 0.16, 0.08, 0.060, 0.062 # intestins?
    # DA, DB, f, k = 0.14, 0.06, 0.035, 0.065 # bacteria
    DA, DB, f, k = parameters
    delta_t = 1.0 # update in time
    N_simulation_steps = 10000 # simulation steps
    N = 100 # grid size
    
    A = np.ones((3,3))
    A[1,1] = 0
    
    A0, B0 = get_initial_configuration(N, random_influence=0.2)
    
    A_T = np.empty(N,N,0)
    B_T = np.empty(N,N,0)
    for t in range(N_simulation_steps):
        A, B = gray_scott_update(A0, B0, DA, DB, f, k, delta_t)
        A_T = np.append(A_T,np.expand_dims(A, axis = 2))
        B_T = np.append(B_T,np.expand_dims(B, axis = 2))
    if is_plot: 
        draw(A0,B0)    
        draw(A,B)
    



def plot_traj(traj_X,traj_y):
    num_clust =np.max(traj_y)+1
    fig = plt.figure(figsize=(50, 50))  # width, height in inches
    colors = ['r','b','g','k','r','b','g','k']
    for i in range(num_clust):
        print(i)
        sub = fig.add_subplot(math.ceil(num_clust**0.5),
                              math.ceil(num_clust**0.5), i+1)
        Xt = traj_X[traj_y == i]
        sub.plot(Xt,colors[i], alpha=0.3)
        


if __name__ == "__main__": 
    from Label_data import Temporal_cluster
    import math
    from numpy import genfromtxt
    num_traj = 200
    num_times = 150
    n_clusters = 8
    my_data = genfromtxt('all_compiled_flux_trunc.csv', delimiter=',')
    traj_X = my_data[1:,6:]
    traj_y = Temporal_cluster(traj_X,metric = "softdtw", n_clusters=n_clusters,
                         isplot =True,seed = 0,n_init =2,n_jobs = None )

    # traj_X = np.random.rand(num_traj,num_times)
    # traj_y = np.random.randint(n_clusters, size=(num_traj,))
    plot_traj(traj_X,traj_y)
    plt.figure()
    plt.hist(traj_y, density=False, bins=max(traj_y)+1,
             facecolor='g', alpha=0.75)  # density=False would make counts
    plt.ylabel('count')
    plt.xlabel('bins');