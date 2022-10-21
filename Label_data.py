"""
Packages 
"""
import numpy as np 
import sys
import cv2
import os, glob
import math
import matplotlib.pyplot as plt
import pickle
import mat73
# for clustering, dimension reduction, feature selection
from clustimage import Clustimage
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesResampler
import pandas as pd
import re
from utils import save2pkl, load_pkl
import time 
import matplotlib.animation as animation
from itertools import chain

sys.path.insert(0, 'src')
seed = 0
np.random.seed(seed)


"""
define functions

TODO: 
    - define fucntin for embedding + feature selection [use old function]
    - 
    - 
"""
def read_mp4(my_dir,img_dim = 300,T = 61):
    os.chdir(my_dir)
    
    img_dim_red = 64
    # trajectories = np.empty((0,img_dim, img_dim, 3,T)).astype(np.float16)
    trajectories = np.empty((0,img_dim_red, img_dim_red, 3,T//2+1)).astype(np.float16)
    parameters = np.empty((0,5)).astype(np.float16)
    for file in glob.glob("*.mp4")[:-1:10]:
        floats = []
        p_int = re.compile(r'\d+')
        p_float = re.compile(r'\d+\.\d+')  # Compile a pattern to capture float values
        floats = [float(i) for i in p_int.findall(file[:file.find('dox')])]
        for i in p_float.findall(file):
            floats.append(float(i))  # Convert strings to float
        # print(file ,"", floats)
        parameters = np.append(parameters,np.expand_dims(floats,axis=0), axis=0).astype(np.float16)
        print("loading : ", file)
        cap = cv2.VideoCapture(file)
        # Check if camera opened successfully
        if (cap.isOpened()== False):
          print("Error opening video stream or file")
          cap.release()
          continue 
        # trajectory = np.empty((img_dim, img_dim, 3,0))
        trajectory = np.empty((img_dim_red, img_dim_red, 3,0))
        # Read until video is completed
        count = 0 
        while (cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret == True:
                count += 1
                frame = cv2.resize(frame, (img_dim_red,img_dim_red),interpolation = cv2.INTER_NEAREST)
                frame = np.expand_dims(frame.astype(np.float16),axis=3)
                if count%2:
                    trajectory = np.append(trajectory,frame, axis=3)
                
            # Break the loop
            else:
                break
        
        trajectories = np.append(trajectories,np.expand_dims(trajectory,axis=0), axis=0)
        cap.release() # release the video capture object
        
    return trajectories,parameters

def spatial_cluster(img_X,min_clust= 5,dim=[], max_clust= 15, 
                    method='pca-hog',
                    verbose = 60,isplot = True):
    """
    
    literature and packages used
    - https://erdogant.github.io/clustimage/pages/html/index.html
    - 
    
    To-do: 
        - modify pca to accept images with 3 channels
    """
    assert method in ["pca","hog","pca-hog"]
    print(method + " based spatial clustering")
    cl = Clustimage(method = method,verbose = verbose,dim=dim)
    # Import data. This can either by a list with pathnames or NxM array.
    X = cl.import_data(img_X)
    # Extract features with the initialized method (PCA in this case)
    img_feat = cl.extract_feat(X)
    # Embedding using tSNE
    xycoord = cl.embedding(img_feat)
    
    # Cluster with all default settings
    img_y = cl.cluster(cluster='agglomerative',
                        evaluate='silhouette',
                        metric='euclidean',
                        linkage='ward',
                        min_clust = min_clust,
                        max_clust = max_clust,
                        cluster_space='high')
    
    # Return
    results = cl.results
    img_y = results['labels']
    
    num_feat = img_feat.shape[1]
    print("number of images = ", img_X.shape )
    print("number of features = ", num_feat)
    print("labels of 1st 10 images", results['labels'][:10]) 
    
    # plot stuff 
    if isplot:
        results.keys()
        
        # Silhouette plots
        cl.clusteval.plot()
        cl.clusteval.scatter(xycoord)
        
        # PCA explained variance plot
        # cl.pca.plot()
        
        # Dendrogram
        # cl.dendrogram()
        
        # Plot unique image per cluster
        cl.plot_unique(img_mean=False)
        
        # # Scatterplot
        cl.scatter(zoom=1, img_mean=False)
        cl.scatter(zoom=None, img_mean=False)
        
        # # Plot images per cluster or all clusters
        # cl.plot(cmap='binary', labels=[1,2])
        cl.plot(cmap='binary')
    return cl, img_feat, img_y


def Temporal_cluster(traj_feat, metric = "euclidean", n_clusters=3,
                     isplot =True,seed = 0,n_init =2,n_jobs = None ):
    """
    based on  https://tslearn.readthedocs.io/en/stable/auto_examples/clustering/plot_kmeans.html#sphx-glr-auto-examples-clustering-plot-kmeans-py

    """
    traj_feat_norm = TimeSeriesScalerMeanVariance().fit_transform(traj_feat)
    # traj_feat_norm = TimeSeriesResampler(sz=40).fit_transform(traj_feat_norm)
    # numpy.random.shuffle(traj_feat_norm)
    # Keep only 50 time series
    traj_feat_norm = TimeSeriesScalerMeanVariance().fit_transform(traj_feat_norm)
    # Make time series shorter
    # traj_feat_norm = TimeSeriesResampler(sz=40).fit_transform(traj_feat_norm)
    sz = traj_feat_norm.shape[1]
    
    # Euclidean k-means
    assert metric in ["euclidean","dtw","softdtw"]
    print(metric + " k-means")
    print(n_clusters , " clusters ")
    
    if metric == "softdtw" :
        metric_params={"gamma": .01}
        max_iter_barycenter=10
    elif metric == "dtw": 
        metric_params=None # {"gamma": .01}
        max_iter_barycenter=10
    else: 
        metric_params = None
        max_iter_barycenter = 100
        
    km = TimeSeriesKMeans(n_clusters=n_clusters,
                          n_init = n_init,
                          verbose=True, 
                          metric = metric,
                          metric_params = metric_params,
                          max_iter_barycenter= max_iter_barycenter, # only for dtw,soft dtw
                          random_state=seed)
    
    traj_y = km.fit_predict(traj_feat_norm)
    
    if isplot: 
        plt.figure()
        for yi in range(n_clusters):
            plt.subplot(math.ceil(n_clusters**0.5),
                                  math.ceil(n_clusters**0.5),
                                  yi+1)
            for xx in traj_feat_norm[traj_y == yi]:
                plt.plot(xx.ravel(), "k-", alpha=.2)
            plt.plot(km.cluster_centers_[yi].ravel(), "r-", alpha=1)
            plt.xlim(0, sz)
            plt.ylim(-4, 4)
            plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
                     transform=plt.gca().transAxes)
            plt.grid()
            if yi == 1:
                plt.title(metric + " $k$-means")

        plt.tight_layout()
        plt.show()
        
    return traj_y


def extract_feat():
    
    
    pass

def plot_canvas(img_arr,num_images = 9):
    fig = plt.figure(figsize=(50, 50))  # width, height in inches

    for i in range(max(num_images,img_arr.shape[0])):
        # print(i)
        sub = fig.add_subplot(3,3, i+1)
        frame = img_arr[i].astype(np.uint8) 
        sub.imshow(frame)

def animate_canvas(vid_arr,classes = [1,2,3],num_vids = 9,num_time_steps = 20):
    names = []
    for i in range(min(num_vids,vid_arr.shape[0])):
        fig, ax = plt.subplots()
        # flatten_axes = list(chain.from_iterable(axes))
        vids = []
        # ax = flatten_axes[i]
        for t in range(num_time_steps):
            vid = ax.imshow(vid_arr[i,:,:,:,t], animated=True)
            if t == 0:
                ax.imshow(vid_arr[i,:,:,:,t])  # show an initial one first
            vids.append([vid])
        
            # print(i)
        ani = animation.ArtistAnimation(fig, vids, interval=500, blit=True,
                                    repeat_delay=1000)
        plt.show()
        name = ["movie"+str(i)+".mp4"]
        names.append(name)
        ani.save(name)
    window_titles = names
    cap = [cv2.VideoCapture(i) for i in names]
    frames = [None] * len(names);
    gray = [None] * len(names);
    ret = [None] * len(names);    
    while True:

        for i,c in enumerate(cap):
            if c is not None:
                ret[i], frames[i] = c.read();
    
    
        for i,f in enumerate(frames):
            if ret[i] is True:
                gray[i] = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                cv2.imshow(window_titles[i], gray[i]);
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
           break
    
    
    for c in cap:
        if c is not None:
            c.release();
    
    cv2.destroyAllWindows()

def plot_traj(traj_feat,traj_y):
    
    fig = plt.figure(figsize=(50, 50))  # width, height in inches

    for i in range(max(traj_y)):
        # print(i)
        sub = fig.add_subplot(3,3, i+1)
        Xt = traj_feat[traj_y == i]
        sub.plot(Xt, alpha=1)
    
       
if __name__ == "__main__": 
    
    np.random.default_rng().standard_normal(size=1, dtype='float32')
    """
    upload data 
    """
    path = r"D:/Projects/ST_learn_synthesis/output/"
    os.makedirs(path, exist_ok=True)
    
    # case = "rand"
    case = "abm"
    # case = "turing1"
    # case = "turing2"
    print("#################################")
    print("Loading data for case  " + case)    
    gen_new = False#input("Generate new data for case "+ case+ "? [True, False]")
    tic = time.time()
    if gen_new: 
        if case == "rand": 
            traj_raw = np.random.rand(17000,300,300,3,60)
        
        elif case == "abm":
            data_folder = r"D:/Projects/ST_learn_synthesis/data/abm/cell_sorting_videos"
            traj_raw,traj_input_par = read_mp4(data_folder)
            
        elif case == "turing1": 
            data_dir        = r"D:/Projects/ST_learn_synthesis/data/traj_raw_Turing.mat"
            mat             = mat73.loadmat(data_dir) # insert your filename here
            traj_input_par  = mat["parameters"]
            print("traj_input_par.shape : ", traj_input_par.shape)
            traj_raw        = mat["traj_raw"]
            print("traj_raw.shape : ", traj_raw.shape)
            
        elif case == "turing2": 
            data_dir        = r"D:\Projects\ST_learn_synthesis\data\turing\*.pkl"
            
            files           =  glob.glob(data_dir)
            num_traj,img_dim,_,T = pd.read_pickle(files[0])["A_T_batch"].shape
            num_par         = len(pd.read_pickle(files[0])["par_batch"][0])
            traj_raw        = np.empty((0,img_dim,img_dim,T))
            traj_input_par  = np.empty((0,num_par))
            for file in files:
                traj_info   = pd.read_pickle(file)
                traj_par    = np.expand_dims(np.array(traj_info["par_batch"][0]),0)
                traj_input_par = np.append(traj_input_par,traj_par,axis=0)
                traj_raw    = np.append(traj_raw,traj_info["A_T_batch"],axis=0)
            print(traj_raw.shape)
        
    else:
        traj_raw_file       = path +case+ '_traj_raw.pkl'
        traj_input_par_file = path +case+ '_traj_input_par.pkl'
        traj_raw            = load_pkl(traj_raw_file) *(1+249* (case=="turing1"))
        # traj_input_par      = load_pkl(traj_input_par_file)
    toc = time.time()   
    print("#################################")
    print("Finished loading data in " , toc-tic, " s")
    
    """
    sample data
    """
    print("Sampling data")
    n_samples_traj = 1000
    n_samples_img  = 20000 # set to -1 to process all data
    t1,t2,dt       = 5,55,2
        
    num_traj        = traj_raw.shape[0]
    n_samples_traj  = min(n_samples_traj,num_traj)
    index_traj      = np.random.choice(traj_raw.shape[0],n_samples_traj, replace=True)
    traj_X_3d       = traj_raw[index_traj]
    traj_X          = traj_X_3d[:,:,:,:,t1:t2:dt]
    
    if len(traj_raw.shape)==4:
        print("this was one channels dataset")
        traj_X      = np.repeat(traj_X[:, :,:, np.newaxis,:], 3, axis=3)
    
    num_traj,img_h,img_w,img_ch,num_time_steps = traj_X.shape
    n_samples_img   = min(n_samples_img,num_traj*num_time_steps) # for 2 random indices
    img_dim         = [img_h,img_w,img_ch]
    traj_X          = np.transpose(traj_X, (0,4,1,2,3)) # [num_traj,num_time_steps,img_h,img_w,img_ch]
    img_X_3d        =  traj_X.reshape((-1,img_h,img_w,img_ch))
    
    index_img       = np.random.choice(img_X_3d.shape[0],n_samples_img, replace=True)
    img_X_3d        = img_X_3d[index_img]
    img_X           = img_X_3d.reshape((n_samples_img,-1))
    
    print("traj_raw.shape :",traj_raw.shape)
    print("Sampled img_raw.shape : ", img_X.shape)
    
        
    # save raw data and free memory
    if gen_new: 
        input("Do you really want to overwrite the files?")
        # img_raw_file = path + case+ '_img_raw.pkl'
        # save2pkl(img_raw_file,img_raw)
        traj_raw_file = path +case+ '_traj_raw.pkl'
        save2pkl(traj_raw_file,traj_raw)
        
    del traj_raw
    # del img_raw
        
    print("#################################")
    print("saved raw data and cleared memory")
    
    img_arr = traj_X[:9,10,:,:,:]
    plot_canvas(img_arr,num_images = 9)
    
    
    """
    SPATIAL CLUSTERING
    
    """
    print("#################################")
    print("Started spatial clustering")
    tic = time.time()
    cl, img_feat, img_y = spatial_cluster(img_X, 
                                          dim = img_dim,
                                          method='pca',
                                          min_clust= 10,
                                          max_clust= 30,
                                          isplot = True)
    
    toc = time.time()
    print("#################################")
    print("Finished spatial clustering in ",toc-tic , " s")
    num_feat = img_feat.shape[1]
    plt.hist(img_y, density=False, bins=max(img_y)+1)  # density=False would make counts
    plt.ylabel('count')
    plt.xlabel('bins');   
    """
     TEMPORAL CLUSTERING
     
    """
    print("Started temporal clustering")
    
    traj_feat = np.zeros([num_traj, num_time_steps, num_feat])
    traj_X    = traj_X.reshape((num_traj, num_time_steps, -1))
    # embed ST trajectories 
    for t in range(num_time_steps):
        X_t = cl.import_data(traj_X[:,t,:])
        X_temp = cl.extract_feat(X_t)
        
        traj_feat[:,t,:] = X_temp
    
    tic = time.time()
    traj_y = Temporal_cluster(traj_feat, n_clusters=10,isplot =True )
    
    toc = time.time()
    plt.hist(traj_y, density=False, bins=max(traj_y)+1)  # density=False would make counts
    plt.ylabel('count')
    plt.xlabel('bins');
    
    # vid_arr = traj_raw[:10]
    # animate_canvas(vid_arr)
    print("#################################")
    print("Finished temporal clustcamering in ",toc-tic , " s")
    
    plot_traj(traj_feat,traj_y)
    
    """ 
    process and save data 
    """
    print("#################################")
    print("Saving results")
    

    # save sampled data and results 
    img_index_file = path +case+ '_img_in.pkl'
    img_X_file = path +case+ '_img_X.pkl'
    img_feat_file = path +case+ '_img_feat.pkl'
    img_y_file = path +case+ '_img_y.pkl'
    
    traj_index_file = path +case+ '_traj_ind.pkl'
    traj_X_file = path +case+ '_traj_X.pkl'
    traj_feat_file = path +case+ '_traj_feat.pkl'
    traj_y_file = path + case+ '_traj_y.pkl'
    
    save2pkl(img_index_file,index_img)
    save2pkl(img_X_file,img_X_3d)
    save2pkl(img_feat_file,img_feat)
    save2pkl(img_y_file,img_y)
    
    save2pkl(traj_index_file,index_traj)
    save2pkl(traj_X_file,traj_X_3d)
    save2pkl(traj_feat_file,traj_feat)
    save2pkl(traj_y_file,traj_y)
    
    # plot and save performance metrics 


