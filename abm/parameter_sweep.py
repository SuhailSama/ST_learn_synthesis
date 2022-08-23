import os
import sys
import model
from multiprocessing import Pool


def parameter_sweep(yaml_file, path=r'/storage/scratch1/5/salsalehi3'):
    print(yaml_file + " ...")
    os.chdir(path + '/yaml_parameters')
    model.TestSimulation.start_sweep(path + '/outputs', yaml_file, f"{yaml_file[:-5]}", 0)
    print("Finished")


if __name__ == "__main__":
    directory = '/storage/scratch1/5/salsalehi3'#os.getcwd()
    num_processes = 128 # int(sys.argv[2])
    # print("directory: ", directory)
    os.chdir(directory)
    yaml_array = os.listdir( os.path.join(directory, "yaml_parameters"))
    yaml_array = [s for s in yaml_array if ".yaml" in s]
    # print(yaml_array)
    # parameter_sweep(yaml_array[0])
    with Pool(processes=num_processes) as pool:
        pool.map(parameter_sweep, yaml_array)


