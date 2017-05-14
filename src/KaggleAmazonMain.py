import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from skimage.io import imread, imshow
from skimage import transform, img_as_float
import glob
import math


def load_training_data(ftype='jpg'):
    """Returns (train_imgs, labels, im_names, tagged_df)"""
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
    train_imgs = np.asarray(train_imgs)
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



