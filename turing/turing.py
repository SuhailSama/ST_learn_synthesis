# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 16:27:22 2022

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
    


def turing_model(parameters,num_sim_steps = 10000,grid_size = 100, 
                 delta_t = 1.0,reg_freq =200, is_plot = True):
    """
    parameters = [DA, DB, f, k]: 
        DA - ,
        DB - , 
        f  - , 
        k  - . 
        examples: 
        DA, DB, f, k = 0.16, 0.08, 0.060, 0.062 # intestins?
        DA, DB, f, k = 0.14, 0.06, 0.035, 0.065 # bacteria
    """
    
    DA, DB, f, k = parameters
    
    A = np.ones((3,3))
    A[1,1] = 0
    
    A0, B0 = get_initial_configuration(grid_size, random_influence=0.2)
    
    A_T = np.empty((grid_size,grid_size,0))
    B_T = np.empty((grid_size,grid_size,0))
    for t in range(num_sim_steps):
        A, B = gray_scott_update(A0, B0, DA, DB, f, k, delta_t)
        if not t % reg_freq: 
            A_T = np.append(A_T,np.expand_dims(A, axis = 2), axis = 2)
            B_T = np.append(B_T,np.expand_dims(B, axis = 2), axis = 2)
    if is_plot: 
        draw(A0,B0)    
        draw(A,B)
    return A_T, B_T


