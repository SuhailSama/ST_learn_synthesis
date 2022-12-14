import os
import sys
import model
import numpy as np


def parameter_sweep(yaml_file, path="D:\\Projects\\ST_learn_synthesis\\abm"):
    print(yaml_file + " ...")
    os.chdir(path + '\\yaml_parameters')
    model.TestSimulation.start_sweep(path + '\\outputs', yaml_file, f"{yaml_file[:-5]}", 0)
    print("Finished")

# 
if __name__ == "__main__":
    path = "D:\\Projects\\ST_learn_synthesis\\abm"
    process_tag = int(1)
    #replicate_tag = int(sys.argv[2])
    #cells_tag = int(sys.argv[3])
    os.chdir(path)
    yaml_array = os.listdir(path + "\\yaml_parameters")
    #parameter_sweep(f'{process_tag}_dox_aba_{replicate_tag}_{cells_tag}_cells.yaml')
    parameter_sweep(yaml_array[0])

