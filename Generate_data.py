# Import Packages

import  os
# import sys
from abm import model
import numpy as np
import datetime
from multiprocessing import Pool

def parameter_sweep(yaml_file):
    directory = os.getcwd()
    os.chdir(os.path.join(directory, "yaml_parameters"))
    model.TestSimulation.start_sweep(directory + '/outputs', yaml_file, f"{yaml_file[:-5]}", 0)
    print("Finished")

def parameter_sweep_morpheus(parameters):
    print("Started")
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

if __name__=="__main__": 
    # model_name = "morpheus"
    model_name = "abm" 
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
        # parameter_sweep(yaml_array[0])
        with Pool(processes=num_processes) as pool:
            pool.map(parameter_sweep, yaml_array[:10])
    else: 
        print("Invalid model name, please enter a valid value (\"abm\" or \"morpheus\") ")