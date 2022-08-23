"""
  @suhail
  python3 MorpheusSetup.py  num_mesendo me_chemo_strength m__chemo_strength e__chemo_strength sim_id simulation_time

example: python3 MorpheusSetup.py  100 100 100 0 1 1200
        python3 MorpheusSetup.py  "100" "100" "100" "0" "1" "1200"
"""
import sys, os, errno
import argparse
import re
from shutil import copyfile
from xml.dom import minidom

import numpy as np


def main():
    args = read_args()                                # read command line input

    model_file = 'morpheus.xml'
    configure_XML_model(args, model_file)  # create folder, and modify XML file
    run_model(model_file)                             # runs simulation, saves stderr/stdout to file

def make_sure_path_exists(path):
    """ Function for safely making output folders """

    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

def read_args():

    # Read parameters from the command line
    parser = argparse.ArgumentParser(description=\
    "This is a morpheus model .\n")

    #Set command line arguments

    parser.add_argument('pp_chemo_strength',
                        help='pluripotent  Chemotaxis strength []',
                        type=str)
    parser.add_argument('mn_chemo_strength',
                        help='Mesoendo Chemotaxis strength []',
                        type=str)
    parser.add_argument('me_chemo_strength',
                        help='Mesoderm Chemotaxis strength []',
                        type=str)
    parser.add_argument('en_chemo_strength',
                        help='endoderm Chemotaxis strength []',
                        type=str)
    parser.add_argument('num_mesendo',
                        help='Number of mesendoderm cells',
                        type=str)
    parser.add_argument('rand_seed',
                        help='random seed',
                        type=str)
    parser.add_argument('sim_id',
                        help='Unique identifier for a simulations so replicates can be run.',
                        type=str)
    parser.add_argument('simulation_time',
                        help='Number of hours the model will simulate. Default is 96 (units are hours)',
                        type=str)
    # Process arguments
    args = parser.parse_args()
    print('args succesfully uploaded', args)
    return args


##############################################################################
def configure_XML_model(args, model_file):
    output_folder ='%s'%(args.sim_id)

    output_folder_prefix = "simulations"
    output_folder = os.path.join(output_folder_prefix,output_folder) #prefix folder
    print(output_folder)
    #create new directory for sim and chdir
    make_sure_path_exists(output_folder)
    copyfile(model_file, os.path.join(output_folder,model_file))
    os.chdir(output_folder)

    # Modify new XML file with model parameters
    change_XML(args,model_file)

def change_XML(args,model_file):
    """
        Function: Modifies the XML file used to run simulations.
            1. change number of cells, properties of mesoendo, meso  and endo cells
            2. change the random seed so simulations are truly stochastic
            3. change the simulation stop time.

    Note. All variables and properties defined in this file must match the
          corresponding variable name in the Morpheus XML file exactly!
    """

    # 1. change number of cells, properties of mesoendo, meso  and endo cells
    doc = minidom.parse(model_file)
    variables = doc.getElementsByTagName("Variable")

    for var in variables:
        symbol_name = var.attributes["symbol"].value

        # set initial number of cells
        if symbol_name == 'num_mesendo':
            var.setAttribute("value",str(args.num_mesendo))
            #print (symbol_name,args.num_mesendo)

        # Set cell ratios
        elif symbol_name == 'simulation_time':
            var.setAttribute("value",str(args.simulation_time))
            #print (symbol_name,args.simulation_time)

        # set chemotactic strength for mesoendo cells
        elif symbol_name.startswith('mn'):
            if 'chemo_strength' in symbol_name:
                #print (symbol_name,args.mn_chemo_strength)
                var.setAttribute("value",str(args.mn_chemo_strength))

        # set chemotactic strength for mesoderm cells
        elif symbol_name.startswith('me'):
            if 'chemo_strength' in symbol_name:
                #print (symbol_name,args.me_chemo_strength)
                var.setAttribute("value",str(args.me_chemo_strength))

        # set chemotactic strength for endoderm cells
        elif symbol_name.startswith('en'):
            if 'chemo_strength' in symbol_name:
                #print (symbol_name,args.en_chemo_strength)
                var.setAttribute("value",str(args.en_chemo_strength))


    # 2. Change the random seed to a random number not dependent on time.
    random_integer = np.random.randint(0,10)
    RandomSeed_variable = doc.getElementsByTagName("RandomSeed")[0]
    RandomSeed_variable.setAttribute("value",str(random_integer))

    # 3. Change simulation stop time in XML file
    simulation_stop_time = args.simulation_time
    if simulation_stop_time == None:
        simulation_stop_time = 100
    stop_time = doc.getElementsByTagName("StopTime")[0]
    stop_time.setAttribute("value",str(simulation_stop_time))

    # Write XML file
    file_handle = open(model_file,"w")
    doc.writexml(file_handle)
    file_handle.close()
    #sys.exit()

######################################## running the model ####################
def run_model(model_file):
    status = os.system("morpheus -file %s > model_output.txt 2>&1" % model_file)
    return status

##########################################
if __name__ == "__main__":
    main()
