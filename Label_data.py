"""
Packages 
"""
import numpy as np 
import sys
import cv2
import os, glob

import matplotlib.pyplot as plt
import pickle
import mat73
# for clustering, dimension reduction, feature selection
from clustimage import Clustimage
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesResampler

sys.path.insert(0, 'src')
seed = 0
np.random.seed(seed)


"""
define functions

To do: 
    - define fucntin for embedding + feature selection [use old function]
    - 
"""
def read_mp4(my_dir):
    os.chdir(my_dir)
    
    trajectories = np.empty((0,500, 500, 3,41))
    
    for file in glob.glob("*.mp4"):
        print(file)
        cap = cv2.VideoCapture("loading : ", file)
        # Check if camera opened successfully
        if (cap.isOpened()== False):
          print("Error opening video stream or file")
          cap.release()
          continue 
        trajectory = np.empty((500, 500, 3,0))
        # Read until video is completed
        while (cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret == True:
                trajectory = np.append(trajectory,np.expand_dims(frame,axis=3), axis=3)
            # Break the loop
            else:
                break
        trajectories = np.append(trajectories,np.expand_dims(trajectory,axis=0), axis=0)
        cap.release() # release the video capture object
        
    return trajectories

def spatial_cluster(Xraw,min_clust= 5,dim=[], max_clust= 15, method='pca-hog',verbose = 60,isplot = False):
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
    X = cl.import_data(Xraw)
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
    print("number of images = ", Xraw.shape )
    print("number of features = ", num_feat)
    print("labels of 1st 10 images", results['labels'][:10]) 
    
    # plot stuff 
    if isplot:
        results.keys()
        # Silhouette plots
        cl.clusteval.plot()
        cl.clusteval.scatter(xycoord)
        
        # PCA explained variance plot
        cl.pca.plot()
        
        # Dendrogram
        cl.dendrogram()
        
        # Plot unique image per cluster
        cl.plot_unique(img_mean=False)
        
        # # Scatterplot
        cl.scatter(zoom=3, img_mean=False)
        cl.scatter(zoom=None, img_mean=False)
        
        # # Plot images per cluster or all clusters
        cl.plot(cmap='binary', labels=[1,2])
        cl.plot(cmap='binary')
    return cl, img_feat, img_y


def Temporal_cluster(traj_feat, metric = "euclidean", n_clusters=3,isplot =True,seed = 0,n_init =2,n_jobs = None ):
    """
    based on  https://tslearn.readthedocs.io/en/stable/auto_examples/clustering/plot_kmeans.html#sphx-glr-auto-examples-clustering-plot-kmeans-py

    """
    traj_feat_norm = TimeSeriesScalerMeanVariance().fit_transform(traj_feat)
    traj_feat_norm = TimeSeriesResampler(sz=40).fit_transform(traj_feat_norm)
    # numpy.random.shuffle(traj_feat_norm)
    # Keep only 50 time series
    traj_feat_norm = TimeSeriesScalerMeanVariance().fit_transform(traj_feat_norm[:50])
    # Make time series shorter
    traj_feat_norm = TimeSeriesResampler(sz=40).fit_transform(traj_feat_norm)
    sz = traj_feat_norm.shape[1]
    
    # Euclidean k-means
    assert metric in ["euclidean","dtw","softdtw"]
    print(metric + " k-means")
    print(n_clusters , " clusters ")
    
    if metric == "softdtw" :
        metric_params={"gamma": .01}
        # max_iter_barycenter=10
    elif metric == "dtw": 
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
            plt.subplot(1,n_clusters , yi+1)
            for xx in traj_feat_norm[traj_y == yi]:
                plt.plot(xx.ravel(), "k-", alpha=.2)
            plt.plot(km.cluster_centers_[yi].ravel(), "r-")
            plt.xlim(0, sz)
            plt.ylim(-4, 4)
            plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
                     transform=plt.gca().transAxes)
            if yi == 1:
                plt.title(metric + " $k$-means")

        plt.tight_layout()
        plt.show()
        
    return traj_y


if __name__ == "__main__": 
    
    """
    upload data 
    """
    
    # case = "rand"
    # case = "turing"
    case = "abm"
    
    if case == "rand": 
        traj_raw = np.random.rand(1000,6,6,3,10)
    
    elif case == "turing": 
        data_dir = ""
        mat = mat73.loadmat('traj_raw_Turing.mat') # insert your filename here
        traj_input_par = mat["parameters"]
        print(traj_input_par.shape)
        traj_raw = mat["traj_raw"]
    
    elif case == "abm":
        data_folder = "D:/Projects/ST_learn_synthesis/data/abm"
        
        traj_raw = read_mp4(data_folder)
    
    num_traj,img_h,img_w,img_ch,num_time_steps = traj_raw.shape
    img_dim = [img_h,img_w,img_ch]
    traj_raw = np.transpose(traj_raw, (0,4,1,2,3)) # [num_traj,num_time_steps,img_h,img_w,img_ch]
    img_raw =  traj_raw.reshape((-1,img_h,img_w,img_ch))
    
    n_samples = min(10000,num_traj*num_time_steps) # for 2 random indices
    index = np.random.choice(img_raw.shape[0],n_samples, replace=True)
    Xraw = img_raw[index].reshape((n_samples,-1))
    print("traj_raw.shape :",traj_raw.shape)
    print("img_raw.shape : ", img_raw.shape)
    print("Sampled img_raw.shape : ", Xraw.shape)
    
    """
    Spatial clustering
    
    """
    cl, img_feat, img_y = spatial_cluster(Xraw, dim = img_dim,method='hog',max_clust= 10)
    num_feat = img_feat.shape[1]
    """
     temporal clustering
     
    """
    traj_feat = np.zeros([num_traj, num_time_steps, num_feat])
    traj_raw = traj_raw.reshape((num_traj, num_time_steps, -1))
    # embed ST trajectories 
    for t in range(num_time_steps):
        X_t = cl.import_data(traj_raw[:,t,:])
        X_temp = cl.extract_feat(X_t)
        traj_feat[:,t,:] = X_temp
    
    traj_y = Temporal_cluster(traj_feat, n_clusters=3,isplot =True )
    """ 
    process and save data 
    """
    
    from utils import save2pkl
    
    path = os.getcwd()+"\\output\\"
    os.makedirs(path, exist_ok=True)
    
    img_raw_file = path + 'img_raw.pkl'
    img_feat_file = path + 'img_feat.pkl'
    img_y_file = path + 'img_y.pkl'
    
    save2pkl(img_raw_file,img_raw)
    save2pkl(img_feat_file,img_feat)
    save2pkl(img_y_file,img_y)
    
    traj_raw_file = path + 'traj_raw.pkl'
    traj_feat_file = path + 'traj_feat.pkl'
    traj_y_file = path + 'traj_y.pkl'
    
    save2pkl(traj_raw_file,traj_raw)
    save2pkl(traj_feat_file,traj_feat)
    save2pkl(traj_y_file,traj_y)
    
    # plot and save performance metrics 


