# define PSO particles and search space, fitness function 
class Particle():
    def __init__(self):
        self.position = np.random.uniform(-10, 10,(4,1)) # initialization
        self.pbest_position = self.position
        self.pbest_value = - float('inf')
        self.velocity = np.zeros((4,1))
        self.id = None
        
    def __str__(self):
        return print("I am # ", self.id ,"at ", np.transpose(self.position))
    
    def move(self):
        self.position = self.position + self.velocity
        
class Space():  
    def __init__(self, target, target_error, n_particles, simulation_time,CNN_model,SVM_model, selected_features,formula):
        self.rn_seed = 1 # for reproducipility
        self.sim_id = int(0)
        self.simulation_time = simulation_time
        self.target = target
        self.target_error = target_error
        self.n_particles = n_particles
        self.particles = []
        self.gbest_value = -float('inf')
        self.gbest_position = np.random.uniform(-10, 10,(4,1)) #np.array([])
        self.n_time_steps = int(self.simulation_time/500+1)
        self.traces = []
        self.global_best_sim = 0
        self.CNN_model = CNN_model
        self.SVM_model = SVM_model
        self.selected_features = selected_features
        self.formula = formula

    def print_particles(self):
        for particle in self.particles:
            particle.__str__()
   
    def extract_features(self, image, CNN_model):
        '''
        Extracts features from images using a pretrained CNN 
        
        '''
        # load the image as a 224x224 array
        img = load_img(image, target_size=(224,224))
        # convert from 'PIL.Image.Image' to numpy array
        img = np.array(img) 
        reshaped_img = img.reshape(1,224,224,3) 

        # prepare image for model
        imgx = preprocess_input(reshaped_img)
        # get the feature vector
        features = CNN_model.predict(imgx, use_multiprocessing=True)
        return features
    
    def extract_trace(self, path_to_images, CNN_model,SVM_model, selected_features):
        '''
        Stacks spatial features over time into one temporal signal 
        inputs: 
        path_to_images: path to folder containing images with time stamp
        CNN_model: pretrained CNN model to extract features
        SVM_model: pretrained SVM model for spatial classification 
        selected_features: indecies of selected features (created using logistic regression)
        '''
        os.chdir(path_to_images)
        images = []
        while not images:
            with os.scandir(path_to_images) as files:
              # loops through each file in the directory
                for file in files:
                    if file.name.endswith('.png'): #or file.name.endswith('.jpg'):
                      # adds only the image files to the images list
                        images.append(file.name)

        dic_img = {}
        p = r"dump"
        labels = []
        # loop through each image in the dataset
        only_class_labels = []
        for index, image in enumerate (sorted(images)):
            if index>=45:
                break
            # try to extract the features and update the dictionary
            image_name= os.path.splitext(image)[0]
            print('index', index, 'image_name : ', image_name)
            for ii in range(10): 
                try:
                    feat = self.extract_features(image,CNN_model)
                    feat = feat[:,selected_features]
                    label = SVM_model.predict(feat.reshape(1,-1))
                    arr = SVM_model.decision_function(feat.reshape(1,-1)).reshape(1,-1)
                    if index ==0 :
                        traj_dist = arr
                    else:
                        print("index > 0")
                        traj_dist = np.concatenate((traj_dist, arr), axis = 0)
                    print('traj_dist added : ',arr)
                    labels.append(int(label[0]))
                    break

                except Exception as e: 
                    print(e)
                    print('failed')
                    with open(p,'wb') as file:
                        pickle.dump(images,file)
                    time.sleep(10)

        # get a list of the filenames
        filenames = np.array(list(dic_img.keys()))
        labels =np.array(labels)
        return traj_dist, labels
    
    def convert_imgs_to_traces(self, paths_to_simulations):
        first_time = True
        for path_to_images in paths_to_simulations:
            spatio_temp, labels = self.extract_trace(path_to_images, self.CNN_model,self.SVM_model, self.selected_features)
            spatio_temp= np.expand_dims(spatio_temp, axis=0)
            if first_time:
                self.traces = spatio_temp
                
                self.traces_labels = labels
                first_time = False
            else:
                self.traces = np.concatenate((self.traces, spatio_temp), axis = 0)
                self.traces_labels = np.concatenate((self.traces_labels, labels), axis = 0)
            print ('self.traces added : ', spatio_temp )
    def fitness(self, particle_id): # compute robustness
        spatio_temp = self.traces[particle_id]
        # Calculating STL robustness
        # spatio_temp has time in axis =0 and signal in axis =1
        x1= torch.tensor(spatio_temp[:,0].reshape([1, -1, 1]), requires_grad=False).float()
        x2= torch.tensor(spatio_temp[:,1].reshape([1, -1, 1]), requires_grad=False).float()
#         x3= torch.tensor(spatio_temp[:,2].reshape([1, -1, 1]), requires_grad=False)
#         x4= torch.tensor(spatio_temp[:,3].reshape([1, -1, 1]), requires_grad=False)
#         x5= torch.tensor(spatio_temp[:,4].reshape([1, -1, 1]), requires_grad=False)
        print('x1: ', x1)
        print('x2: ', x2)
        signal = (x1,x2)
        
        #calculate robustness
        robustness = formula.robustness(signal, scale=0.5)
        return robustness.item()    
    
                
    def create_simulation_folder(self,sim_id):
        return r'/storage/coda1/p-mkemp6/0/salsalehi3/Organoid_Suhail/simulations/sim %s ' % int(sim_id)
        
        
    def set_pbest(self):
        # Serial processing
        for ind, particle in enumerate(self.particles):
            particle.id = int(self.sim_id%self.n_particles)
            row = np.append(particle.position,np.array(self.sim_id).astype(int).reshape(1,1), axis=0)
            if ind == 0:
                data = row
            else:
                data = np.append(data,row, axis =1)
            self.sim_id += 1
        data = data.transpose()
        paths_to_simulations = [self.create_simulation_folder(x[4]) for x in data]
        path_to_run_sim = r"/storage/coda1/p-mkemp6/0/salsalehi3/Organoid_Suhail/"
        os.chdir(path_to_run_sim)
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
            os.system(run_file) # run in comand line (NOT parallel jobs)
        f.close()
#         print('finished writing to file')
#         terminal_command = "pace-gnu-job -G " +myfile+" -q embers -A GT-mkemp6"
#         print("Terminal command is : ", terminal_command )
#         os.system(terminal_command)
        for sim_folder in paths_to_simulations:
            while not os.path.isdir(sim_folder):
                print('taking a nap!',sim_folder)
                time.sleep(5)
                time.sleep(5)
#         time.sleep(5)
        self.convert_imgs_to_traces(paths_to_simulations)
        for particle in self.particles:
            fitness_cadidate = self.fitness(particle.id)
            print (' Evaluated Particle ',particle.id, ' with fitness',fitness_cadidate, ' and location ', np.transpose(particle.position))
            if(particle.pbest_value < fitness_cadidate):
                particle.pbest_value = fitness_cadidate
                particle.pbest_position = particle.position
        return fitness_cadidate

    def set_gbest(self):
        for particle in self.particles:
            best_fitness_cadidate = self.fitness(particle.id)
            print('best_fitness_cadidate : ',best_fitness_cadidate, 'for particle # : ', particle.id)
            if(self.gbest_value < best_fitness_cadidate):
                self.gbest_value = best_fitness_cadidate
                self.gbest_position = particle.position
                self.global_best_sim = particle.id
        return self.gbest_value, self.global_best_sim

    def move_particles(self):
        w = 0.5
        c1 = 0.5
        c2 = 0.9
        for particle in self.particles:
            new_velocity = (w*particle.velocity) + (c1*random.random()) * (particle.pbest_position - particle.position) + \
                            (random.random()*c2) * (self.gbest_position - particle.position)
         
            particle.velocity = new_velocity
            particle.move()