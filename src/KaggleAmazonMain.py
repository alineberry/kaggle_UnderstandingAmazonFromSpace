import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from skimage.io import imread, imshow
from skimage import transform, img_as_float, filters
from skimage.color import rgb2gray
import glob
import math
import scipy


def load_training_data(ftype='jpg'):
    """
    Returns (train_imgs, labels, im_names, tagged_df)
    
    - train_imgs is a numpy array size (N x 256 x 256 x 3)
    - labels is a list of pandas series, containing label dummy vectors for each image in train_images
      labels could be smaller than tagged_df if only a sample of images is loaded
    - im_names is a list of strings containing the filenames with extension removed
    - tagged_df is a dictionary of all image names and their tags. it is returned as a pandas dataframe
    """
    
    cwd = os.getcwd()
    print("cwd",cwd)

    #open data from current directory. Should work with any direcotry path
    with open(os.path.join(cwd, "..", "data", "train.csv")) as file:
        tagged_df = pd.read_csv(file)

    #split the tags into new rows
    tagged_df = pd.DataFrame(tagged_df.tags.str.split(' ').tolist(), index=tagged_df.image_name).stack()
    tagged_df = tagged_df.reset_index()[[0, 'image_name']] # dataframe with two columns
    tagged_df.columns = ['tags', 'image_name'] # rename columns
    tagged_df.set_index('image_name', inplace=True) # rest index to image_name again

    #create dummy variables for each tag
    tagged_df = pd.get_dummies(tagged_df['tags']) # creates dummy rows
    tagged_df = tagged_df.groupby(tagged_df.index).sum() # adds dummy rows together by image_name index

    train_imgs = []
    labels = []
    im_names = []
    print('Loading {} image dataset'.format(ftype))
    path = os.path.join(cwd, '..', 'data','train-{}-sample'.format(ftype),'*.'+ftype)
    files = glob.glob(path)
    for fs in files:
        img = imread(fs)
        # img = transform.resize(img, output_shape=(h,w,d), preserve_range=True)  if needed
        train_imgs.append(img)
        
        imname = os.path.basename(fs).split('.')[0]
        im_names.append(imname)
        
        labels_temp = tagged_df.loc[imname]
        labels.append(labels_temp)
    train_imgs = img_as_float(np.asarray(train_imgs))
    return train_imgs, labels, im_names, tagged_df

    
def get_labels(fname, tagged_df):
    """return list of labels for a given filename"""
    return ", ".join(tagged_df.loc[fname][tagged_df.loc[fname]==1].index.tolist())    


def plot_samples(X_train, names_train, tagged_df, nrow, ncol):
    """Plots random sample images with their titles and tag names"""
    sampling = np.random.randint(low=0, high=X_train.shape[0]-1, size = nrow*ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(15, 12))
    for i in range(0,len(sampling)):
        name = names_train[sampling[i]]
        tags = get_labels(name, tagged_df)

        row = math.floor(i/ncol)
        col = i - math.floor(i/ncol)*ncol
        if (nrow == 1 or ncol == 1):
            ind = (max(row,col))
        else:
            ind = (row,col)
        axes[ind].imshow(X_train[sampling[i]])
        axes[ind].set_title(name+'\n'+tags)
        axes[ind].tick_params(left=False, right=False)
        axes[ind].set_yticklabels([])
        axes[ind].set_xticklabels([])
    plt.tight_layout()


def get_rgb_vectors(X_train):
    """Returns RBG vectors for each image in the training set (return as tuple)"""
    print("There are {} unique features per image".format((X_train.shape[1]*X_train.shape[2])*X_train.shape[3]))
    red = X_train[:,:,:,0].reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[2]) # row of red pixel features for all images
    green = X_train[:,:,:,1].reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[2]) # row of blue pixel features for all images
    blue = X_train[:,:,:,2].reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[2]) #row of green pixel features for all images
    return red, green, blue
    

def plot_rgb_dist(img, title):
    """Plots the RGB kde for a given image. Image title passed in as argument (image labels perhaps)"""
    plt.subplots_adjust(wspace=0, hspace=0)
    fig = plt.figure(figsize=(18, 3))
    ax = fig.add_subplot(1,2,1)
    sns.kdeplot(img[:,:,0].flatten(), ax=ax, shade=True, color='red')
    sns.kdeplot(img[:,:,1].flatten(), ax=ax, shade=True, color='blue')
    sns.kdeplot(img[:,:,2].flatten(), ax=ax, shade=True, color='green')
    ax = fig.add_subplot(1,2,2)
    ax.imshow(img)
    plt.suptitle(title, fontsize=20)
    plt.axis('off')

def xform_to_gray(imgs):
    return rgb2gray(imgs)

def xform_to_sobel(imgs):
    imgs = xform_to_gray(imgs)
    sobels = []
    if imgs.ndim == 2:
        sobels.append(filters.sobel(imgs))
    else:
        for i in range(imgs.shape[0]):
            sobels.append(filters.sobel(imgs[i]))
    return np.asarray(sobels)

def get_features(imgs):
    """Input is a Nx256x256x3 numpy array of images, where N is number of images"""
        
    r_mean = []
    g_mean = []
    b_mean = []

    r_std = []
    g_std = []
    b_std = []

    r_max = []
    g_max = []
    b_max = []

    r_min = []
    g_min = []
    b_min = []

    r_kurtosis = []
    g_kurtosis = []
    b_kurtosis = []
    
    r_skew = []
    g_skew = []
    b_skew = []
    
    sobel_mean = []
    sobel_std = []
    sobel_max = []
    sobel_min = []
    sobel_kurtosis = []
    sobel_skew = []
    
    sobel_rowmean_std = []
    sobel_colmean_std = []
    
    r_bimodal = []
    g_bimodal = []
    b_bimodal = []
    
    # METRIC FOR BIMODALITY
    # bin each color intensity (histogram)
    # find 2 most populated bins
    # subtract and abs() to quantify bimodality

    for i in range(imgs.shape[0]):
        
        r = imgs[:,:,0].ravel()
        g = imgs[:,:,1].ravel()
        b = imgs[:,:,2].ravel()
                
        s = xform_to_sobel(imgs[i])
        
        r_mean.append(np.mean(r))
        g_mean.append(np.mean(g))
        b_mean.append(np.mean(b))
        
        r_std.append(np.std(r))
        g_std.append(np.std(g))
        b_std.append(np.std(b))
        
        r_max.append(np.max(r))
        b_max.append(np.max(b))
        g_max.append(np.max(g))
        
        r_min.append(np.min(r))
        b_min.append(np.min(b))
        g_min.append(np.min(g))
        
        r_kurtosis.append(scipy.stats.kurtosis(r))
        b_kurtosis.append(scipy.stats.kurtosis(b))
        g_kurtosis.append(scipy.stats.kurtosis(g))
        
        r_skew.append(scipy.stats.skew(r))
        b_skew.append(scipy.stats.skew(b))
        g_skew.append(scipy.stats.skew(g))
        
        sobel_mean.append(np.mean(s.ravel()))
        sobel_std.append(np.std(s.ravel()))
        sobel_max.append(np.max(s.ravel()))
        sobel_min.append(np.min(s.ravel()))
        sobel_kurtosis.append(scipy.stats.kurtosis(s.ravel()))
        sobel_skew.append(scipy.stats.skew(s.ravel()))
        
        sobel_rowmean_std.append(np.std(np.mean(s,axis=1)))
        sobel_colmean_std.append(np.std(np.mean(s,axis=0)))
        
        rb, gb, bb = binned_mode_features(imgs[i])
        r_bimodal.append(rb)
        g_bimodal.append(gb)
        b_bimodal.append(bb)
        
                      
    return pd.DataFrame(
        {'r_mean':r_mean, 'g_mean':g_mean, 'b_mean':b_mean,
         'r_std':r_std, 'g_std':g_std, 'b_std':b_std,
         'r_max':r_max, 'g_max':g_max, 'b_max':b_max,
         'r_min':r_min, 'g_min':g_min, 'b_min':b_min,
         'r_kurtosis':r_kurtosis, 'g_kurtosis':g_kurtosis, 'b_kurtosis':b_kurtosis,
         'r_skew':r_skew, 'g_skew':g_skew, 'b_skew':b_skew,
         'sobel_mean':sobel_mean, 'sobel_std':sobel_std, 
         'sobel_max':sobel_max, 'sobel_min':sobel_min,
         'sobel_kurtosis':sobel_kurtosis, 'sobel_skew':sobel_skew,
         'sobel_rowmean_std':sobel_rowmean_std, 'sobel_colmean_std':sobel_colmean_std,
         'r_bimodal':r_bimodal, 'g_bimodal':g_bimodal, 'b_bimodal':b_bimodal})


def binned_mode_features(img, nbins=10):
                                          
    steps=np.arange(start=0,stop=1, step=1/nbins)
                                                                            
    ## red ##
    #split on mean
    m=img[:,:,0].flatten().mean()
    left = img[:,:,0].flatten()[img[:,:,0].flatten()<m]
    right = img[:,:,0].flatten()[img[:,:,0].flatten()>=m]
    #find mode in left and right
    max_ind_left = np.histogram(left, bins=steps, density=False)[0].argsort()[-1:]
    max_ind_right = np.histogram(right, bins=steps, density=False)[0].argsort()[-1:]
    #calc bimodal metric
    mo1 = np.histogram(right, bins=steps, density=False)[1][max_ind_right]
    mo2 = np.histogram(left, bins=steps, density=False)[1][max_ind_left]
    mods_diff_r=abs(mo1-mo2)

    ## green ##
    m=img[:,:,1].flatten().mean()
    left = img[:,:,1].flatten()[img[:,:,1].flatten()<m]
    right = img[:,:,1].flatten()[img[:,:,1].flatten()>=m]
    max_ind_left = np.histogram(left, bins=steps, density=False)[0].argsort()[-1:]
    max_ind_right = np.histogram(right, bins=steps, density=False)[0].argsort()[-1:]
    mo1 = np.histogram(right, bins=steps, density=False)[1][max_ind_right]
    mo2 = np.histogram(left, bins=steps, density=False)[1][max_ind_left]
    mods_diff_g=abs(mo1-mo2)

    ## blue ##
    m=img[:,:,2].flatten().mean()
    left = img[:,:,2].flatten()[img[:,:,2].flatten()<m]
    right = img[:,:,2].flatten()[img[:,:,2].flatten()>=m]
    max_ind_left = np.histogram(left, bins=steps, density=False)[0].argsort()[-1:]
    max_ind_right = np.histogram(right, bins=steps, density=False)[0].argsort()[-1:]
    mo1 = np.histogram(right, bins=steps, density=False)[1][max_ind_right]
    mo2 = np.histogram(left, bins=steps, density=False)[1][max_ind_left]
    mods_diff_b=abs(mo1-mo2)

    return mods_diff_r[0], mods_diff_g[0], mods_diff_b[0]
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
