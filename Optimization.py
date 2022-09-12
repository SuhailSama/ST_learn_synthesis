""" 
Import Packages
"""
import numpy as np 
import sys
import os
sys.path.insert(0, 'src')

# stl stuff for creating stl formulae and calculating robustness
import torch
import stlcg
import stlviz as viz

# for clustering, dimension reduction, feature selection
import time

print(os.getcwd())

# 
if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu" 
device = torch.device(dev)
print("Running torch on: ", device)


"""
import classes and functions
"""
from opt_backend import Space

"""
load spatial predicate
"""
# spatial predicate

path = os.getcwd()
spat_pred_path = os.path.join(path,'output','spatial_classifier.pth')
spat_pred = torch.load(spat_pred_path)

# temporal robustness 

"""
define specification formula
"""

trace_len = 5
x = torch.tensor(np.random.rand(trace_len,1) ).float() 
y = torch.tensor(np.random.rand(trace_len,1) ).float() 
## define the stl formula 

# thresholds
a = torch.as_tensor(0.48).float() 
b = torch.as_tensor(0.52).float()

# predicate "function"
x_exp = stlcg.Expression('x_exp',x)
y_exp = stlcg.Expression('y_exp',y)

# predicates
ϕ1 = x_exp >a
ϕ2 = y_exp <b

#formula

formula = stlcg.Always(subformula=ϕ1& ϕ2, interval=[2,3])
print('formula is :' ,formula)

#visualize the formula
viz.make_stl_graph(formula)


""" 
hyper parameters
"""
n_iterations = 2   
n_particles = 4  
simulation_time = 500*(trace_len-1)

search_space = Space(simulation_time,formula,spat_pred,n_particles=n_particles)
search_space.print_particles()

"""
MAIN optimization loop 

"""
iteration = 0
start_time = time.time()
min_change_rate = 0.0001
while(True):
    print("########################################################### ")
    print("############# Iteration # ",iteration,"####################" )
    print("########################################################### ")
    iter_time = time.time()
    # search_space.print_particles()
    search_space.set_pbest()
    # print('positions after move')
    # search_space.print_particles()
    current_best_rob, sim_id = search_space.set_gbest()
    # if(abs(search_space.gbest_value - current_best_rob) <= min_change_rate):
    #     break
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



