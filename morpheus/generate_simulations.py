# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 11:45:17 2020

@author: suhail

To generate one run of the morpheus model, use the following command
  python3 MorpheusSetup.py pp_chemo_strength mn_chemo_strength me_chemo_strength en_chemo_strength num_mesendo sim_id simulation_time
  example: python3 MorpheusSetup.py  100 100 100 0 1 1200
"""
import  os
import numpy as np
import datetime


file = 'MorpheusSetup.py'

simulation_time = 24000
step_size = 500
pp_chemo_strength = np.arange(start=0, stop=500, step=step_size)
mn_chemo_strength = np.arange(start=-500, stop=0, step=step_size)
me_chemo_strength = np.arange(start=-500, stop=0, step=step_size)
en_chemo_strength = np.arange(start=-500, stop=0, step=step_size)
random_seed = np.arange(start=1, stop=9, step=10)
print(pp_chemo_strength)

start   = datetime.datetime.now()
sim_id = 0

for i, pp  in enumerate(pp_chemo_strength):
  pp_chemo = pp
  for ii, mn in enumerate(mn_chemo_strength):
      mn_chemo = mn
      for iii, me in enumerate(me_chemo_strength):
          me_chemo = me
          for iiii, en in enumerate(en_chemo_strength):
              en_chemo = en
              for iiiii, rn_seed in enumerate(random_seed):
                  sim_id = sim_id +1
                  sim_par = [pp_chemo,mn_chemo,me_chemo,en_chemo,
                             rn_seed,sim_id,simulation_time]
                  print (sim_par)
                  run_file = 'python3 MorpheusSetup.py \" %s " " %s " " %s " " %s " " %s " " %s " " %s "' % (
                     str(pp_chemo),str(mn_chemo),str(me_chemo),str(en_chemo),
                     str(rn_seed),str(sim_id),str(simulation_time))
                  print('run number: ', sim_id )
                  os.system(run_file)

end = datetime.datetime.now()
simulation_time = end - start
num_sim = sim_id
print('simulation time: ')
print(simulation_time)
print('simulation time per run:')
print(simulation_time/num_sim)

 # run MorpheusSetup.py "pp_chemo_strength" "mn_chemo_strength" "me_chemo_strength" "en_chemo_strength" "num_mesendo" "sim_id" "simulation_time"
