# Import Packages

import  os
import sys
from abm import model
from turing import turing_model
import numpy as np
import datetime
from multiprocessing import Pool



def parameter_sweep_abm(yaml_file):
    print("Started parameter_sweep_abm")
    directory = os.getcwd()
    os.chdir(os.path.join(directory, "yaml_parameters"))
    model.TestSimulation.start_sweep(directory + '/outputs', yaml_file, f"{yaml_file[:-5]}", 0)
    print("Finished parameter_sweep_abm")

def parameter_sweep_morpheus(parameters):
    print("Started parameter_sweep_morpheus")
    simulation_time = 24000
    pp_chemo = parameters[0]     
    mn_chemo = parameters[1]
    me_chemo = parameters[2]
    en_chemo = parameters[3]
    rn_seed  = parameters[4]
    sim_id   = parameters[5]
    run_file = 'python MorpheusSetup.py \" %s " " %s " " %s " " %s " " %s " " %s " " %s "' % (
        str(pp_chemo),str(mn_chemo),str(me_chemo),str(en_chemo),
        str(rn_seed),str(sim_id),str(simulation_time))
    print(run_file, flush=True)
    os.system(run_file)
    print("Finished parameter_sweep_morpheus")
def parameter_sweep_turing(par_batch,num_sim_steps ,grid_size , 
                 delta_t ,reg_freq, is_plot): 
    batch_size  = par_batch.shape[0]
    T           = num_sim_steps//reg_freq
    A_T_batch = np.empty((0,grid_size,grid_size,T))
    B_T_batch = np.empty((0,grid_size,grid_size,T))
    print("Started parameter_sweep_turing")
    for par in par_batch:  
        A_T,B_T = turing_model(par,num_sim_steps ,grid_size, 
                         delta_t,reg_freq, is_plot)
        # print("A_T.shape : ",A_T.shape,"B_T.shape : ",B_T.shape,
        #       "\n A_T_batch ", A_T_batch.shape)
        A_T_batch = np.append(A_T_batch, np.expand_dims(A_T, axis=0),axis=0) 
        B_T_batch = np.append(B_T_batch, np.expand_dims(B_T, axis=0),axis=0)
    print("Finished parameter_sweep_turing")  
    print("YAY! Generated ",batch_size , " trajectories!")
    return A_T_batch # ,B_T_batch

    
if __name__=="__main__": 
    np.seterr(all="ignore")


    # model_name = "morpheus"
    # model_name = "abm" 
    model_name = "turing" 
    if model_name =="morpheus":
        directory = os.getcwd()
        
        step_size = 100
        pp_chemo_strength = np.arange(start=0, stop=500, step=step_size)
        mn_chemo_strength = np.arange(start=-500, stop=0, step=step_size)
        me_chemo_strength = np.arange(start=-500, stop=0, step=step_size)
        en_chemo_strength = np.arange(start=-500, stop=0, step=step_size)
        random_seed = np.arange(start=1, stop=9, step=10)
        
        parameters = np.array(np.meshgrid(pp_chemo_strength,mn_chemo_strength,me_chemo_strength,en_chemo_strength,random_seed)).T.reshape(-1,5)
        num_sim = parameters.shape[0]
        sim_id = np.expand_dims(np.arange(start=0, stop=num_sim, step=1), axis = 1)

        parameters = np.concatenate((parameters, sim_id), axis=1)
        num_processes = 6
        os.chdir(os.path.join(directory, "morpheus"))
        start   = datetime.datetime.now()

        with Pool(processes=num_processes) as pool:
            pool.map(parameter_sweep_morpheus, parameters)

        end = datetime.datetime.now()
        simulation_time = end - start
        print('simulation time: ')
        print(simulation_time)
        print('simulation time per run:')
        print(simulation_time/num_sim)

    elif model_name =="abm":
        num_processes = 10 # int(sys.argv[2])
        os.chdir(os.path.join(os.getcwd(), "abm"))
        directory = os.getcwd()
        yaml_array = os.listdir( os.path.join(directory, "yaml_parameters"))
        yaml_array = [s for s in yaml_array if ".yaml" in s]
        
        start   = datetime.datetime.now()
        parameter_sweep_abm(yaml_array[0])
        # with Pool(processes=num_processes) as pool:
        #     pool.map(parameter_sweep_abm, yaml_array[:10])
        
        end = datetime.datetime.now()
        simulation_time = end - start
        print('simulation time: ')
        print(simulation_time)
        print('simulation time per run:')
        print(simulation_time/num_sim)
        
    elif model_name =="turing":
        num_processes = 10 # int(sys.argv[2])
        DA = np.arange(0.01, 2,   0.1,  dtype=float)
        DB = np.arange(0.01, 2,   0.1,  dtype=float)
        f  = np.arange(0.01, 0.1, 0.01, dtype=float)
        k   = np.arange(0.01,0.1, 0.01, dtype=float)
        par_batch = np.array(np.meshgrid(DA, DB, f, k)).T.reshape(-1,4)
        
        start   = datetime.datetime.now()
        A_T_batch = parameter_sweep_turing(par_batch,num_sim_steps = 10000,
                               grid_size = 100, delta_t = 1.0,
                               reg_freq =200, is_plot = False)
        # with Pool(processes=num_processes) as pool:
        #     pool.map(turing, yaml_array[:10])
        num_sim = A_T_batch.shape[0]
        end = datetime.datetime.now()
        simulation_time = end - start
        print('simulation time: ')
        print(simulation_time)
        print('simulation time per run:')
        print(simulation_time/num_sim)
    else: 
        print("Invalid model name, please enter a valid value (\"abm\" or \"morpheus\") ")