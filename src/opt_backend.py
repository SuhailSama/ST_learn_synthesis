import numpy as np 
import os 
import random 

import torch
import torchvision
import torchvision.transforms as T

# import torchvision.transforms as T

class Particle():
    def __init__(self,p_id):
        self.position = np.random.uniform(-10, 10,(4,1)) # initialization
        self.pbest_position = self.position
        self.pbest_value = - float('inf')
        self.velocity = np.ones((4,1))
        self.id = p_id
        
    def __str__(self):
        return print("\n I am # ", self.id ,"at ", np.transpose(self.position))
    
    def move(self):
        self.position = self.position + self.velocity
        
class Space():  
    def __init__(self, simulation_time,formula,spat_pred,n_img_clust = 6, n_particles=30, rn_seed = 1):
        # print("##### initialized ",n_particles ,  " particles ##### ")
        self.rn_seed = rn_seed # for reproducipility
        self.sim_id = int(0)
        self.simulation_time = simulation_time
        self.particles = [Particle(i) for i in range(n_particles)]
        self.gbest_value = -float('inf')
        self.gbest_position = np.random.uniform(-10, 10,(4,1)) #np.array([])
        self.n_time_steps = int(self.simulation_time/500+1)
        self.traces = []
        self.global_best_sim = 0
        self.formula = formula
        self.spat_pred = spat_pred
        self.n_img_clust = n_img_clust
        self.path = r'D:\Projects\ST_learn_synthesis'
        self.sim_path = os.path.join(self.path,"morpheus","sims")
        self.img_dim = [3,64,64]
        self.transform = T.Resize((64,64))

        
    def print_particles(self):
        print("##### Meet particles! ##### ")
        for particle in self.particles:
            particle.__str__()
   
        
    def extract_st_traj(self, sim_path):
        # print("##### extracting traj_from images ##### ")
        images   = []
        X_t      = torch.zeros([self.n_time_steps,self.img_dim[0],self.img_dim[1],self.img_dim[2]])
        st_traj  = torch.zeros([self.n_time_steps,self.n_img_clust])
        while not images and len(images)<self.n_time_steps:
            images = [os.path.join(sim_path,file) for file in os.listdir(sim_path) if file.endswith('.png')]
        
        for t, image_path in enumerate (sorted(images)):
            if t >= self.n_time_steps: 
                break
            img  = torchvision.io.read_image(image_path)
            img_resized  = self.transform(img)[:3]
            X_t[t,:,:,:] = img_resized 
        st_traj = self.spat_pred(X_t)
        return st_traj
            
    def fitness(self, particle_id): # compute robustness
        # print("##### Calculating robustness ##### ")
        sim_path = os.path.join(self.sim_path, "sim"+str(particle_id))
        st_traj = self.extract_st_traj(sim_path)
        x = st_traj[0].reshape([1, -1, 1]).float()
        y = st_traj[1].reshape([1, -1, 1]).float()
        signal = (x , y)
        robustness = self.formula.robustness(signal, scale=0.5)
        return robustness.item()    
    
    def run_sim(self):
        # print("##### running simulations ##### ")
        for ind, particle in enumerate(self.particles):
            row = np.append(particle.position,np.array(particle.id).astype(int).reshape(1,1), axis=0)
            if ind == 0:
                data = row
            else:
                data = np.append(data,row, axis =1)
        data = data.transpose()
        os.makedirs(self.sim_path, exist_ok=True)
        os.chdir(self.sim_path)
        myfile = "test%s.jobs" %self.sim_id
        if not os.path.exists(myfile):
            f= open(myfile,"x")
        f= open(myfile,"w")
        for x in data:
            x= np.round(x,2)
            run_file = 'python MorpheusSetup.py \" %s " " %s " " %s " " %s " " %s " " %s " " %s "\r\n' % ( \
                str(x[0]),str(x[1]),str(x[2]),str(x[3]),\
                str(self.rn_seed),str(int(x[4])),str(self.simulation_time))
            f.write(run_file)
            # os.system(run_file) # run in comand line (NOT parallel jobs)
        f.close()
        # print('finished writing to file')
        terminal_command = "pace-gnu-job -G " +myfile+" -q embers -A GT-mkemp6"
        # print(" ###### Running parallel jobs  ")
        # os.system(terminal_command)
            
    def set_pbest(self):
        self.run_sim()
        # Serial processing
        particles_updated = []
        for particle in self.particles:
            fitness_cadidate = self.fitness(particle.id)
            if(particle.pbest_value < fitness_cadidate):
                particle.pbest_value = fitness_cadidate
                particle.pbest_position = particle.position
            particles_updated.append(particle)
        self.particles = particles_updated
        
    def set_gbest(self):
        for particle in self.particles:
            best_fitness_cadidate = self.fitness(particle.id)
            if(self.gbest_value < best_fitness_cadidate):
                self.gbest_value = best_fitness_cadidate
                self.gbest_position = particle.position
                self.global_best_sim = particle.id
                
        return self.gbest_value, self.global_best_sim

    def move_particles(self):
        w = 0.5
        c1 = 0.5
        c2 = 0.9
        particles_updated = []
        for particle in self.particles:
            new_velocity = (w*particle.velocity) + (c1*random.random()) * (particle.pbest_position - particle.position) + \
                            (random.random()*c2) * (self.gbest_position - particle.position)
            print()
            particle.velocity = new_velocity
            particle.move()
            particles_updated.append(particle)
        self.particles = particles_updated