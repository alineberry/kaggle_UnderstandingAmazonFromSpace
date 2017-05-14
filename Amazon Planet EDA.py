
# coding: utf-8

# # Exploratory Data Analysis

# In[15]:

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


# In[16]:

#get and print current directory
cwd = os.getcwd()
cwd


# In[17]:

#open data from current directory. Should work with any direcotry path
with open(os.path.join(cwd, "data", "train.csv")) as file:
    tagged_df = pd.read_csv(file)
tagged_df.head()


# In[18]:

#split the tags into new rows
tagged_df = pd.DataFrame(tagged_df.tags.str.split(' ').tolist(), index=tagged_df.image_name).stack()
tagged_df = tagged_df.reset_index()[[0, 'image_name']] # dataframe with two columns
tagged_df.columns = ['tags', 'image_name'] # rename columns
tagged_df.set_index('image_name', inplace=True) # rest index to image_name again

#create dummy variables for each tag
tagged_df = pd.get_dummies(tagged_df['tags']) # creates dummy rows
tagged_df = tagged_df.groupby(tagged_df.index).sum() # adds dummy rows together by image_name index
tagged_df.head()


# In[19]:

#Barplot of tag counts
get_ipython().magic('matplotlib inline')
plt.rcParams['figure.figsize'] = (12, 5)
print('There are {} unique tags in this data'.format(len(tagged_df.columns)))
colors = cm.rainbow(np.linspace(0, 1, len(tagged_df.columns)))
tagged_df.sum().sort_values(ascending=False).plot(title="Counts of Tags", color=colors, kind='bar')
plt.show()
tagged_df.sum().sort_values(ascending=False)


# # Load Image Data

# In[20]:

def load_training_data(ftype='jpg'):
    train_imgs = []
    labels = []
    im_names = []
    print('Loading {} image dataset'.format(ftype))
    path = os.path.join('data','train-{}-sample'.format(ftype),'*.'+ftype)
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
    return train_imgs, labels, im_names


# In[21]:

X_train, y_train, names_train = load_training_data()


# In[22]:

# 100 files, images are 256x256 pixels, with a channel dimension size 3 = RGB
print('X_train is a {} object'.format(type(X_train)))
print('it has shape {}'.format(X_train.shape))


# In[23]:

print('y_train is a {} object'.format(type(y_train)))
print('it has {} elements'.format(len(y_train)))
print('each element is of type {}'.format(type(y_train[0])))
print('and the elements are of size {}'.format(y_train[0].shape))


# In[24]:

print('names_train is a {} object'.format(type(names_train)))
print('it has {} elements'.format(len(names_train)))
print('each element is of type {}'.format(type(names_train)))


# In[25]:

def get_labels(fname):
    return ", ".join(tagged_df.loc[fname][tagged_df.loc[fname]==1].index.tolist())    


# In[26]:

def plot_samples(nrow, ncol):
    sampling = np.random.randint(low=0, high=X_train.shape[0]-1, size = nrow*ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(15, 12))
    for i in range(0,len(sampling)):
        name = names_train[sampling[i]]
        tags = get_labels(name)

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


# In[27]:

plot_samples(4,4)


# In[28]:

X_train[0,:,:,2]


# In[29]:

'train_10'


# # Feature Engineering
# What type of features are we working with here?

# In[30]:

print("There are {} unique features per image".format((256*256)*3))


# In[31]:

red = X_train[:,:,:,0].reshape(100, 256*256) # row of red pixel features for all images
green = X_train[:,:,:,1].reshape(100, 256*256) # row of blue pixel features for all images
blue = X_train[:,:,:,2].reshape(100, 256*256) #row of green pixel features for all images


# In[32]:

red = pd.DataFrame(red, index=???)


# In[113]:

fig, axes = plt.subplots(1, 3, figsize=(15, 12))
axes[0].imshow(X_train[0,:,:,0], cmap='gray')
axes[1].imshow(X_train[0,:,:,1])
axes[2].imshow(X_train[0,:,:,2])
plt.axis('off')


# In[ ]:

fig=plt.figure()
data=np.arange(900).reshape((30,30))
for i in range(1,5):
    ax=fig.add_subplot(2,2,i)        
    ax.imshow(data)

plt.suptitle('Main title')
plt.show() 


# In[98]:

def plot_rgb_dist(img, title):
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


# In[99]:

pics = ['train_10039', 'train_10059', 'train_10034']


# In[141]:

plt.subplots_adjust(wspace=0, hspace=0)
for i in range(0,3):
    sample = np.random.randint(low=0, high=X_train.shape[0]-1, size = 1)
    ind = names_train[sample[0]]
    tags = get_labels(ind)
    plot_rgb_dist(X_train[sample[0],:,:,:],tags)


# In[86]:

sample = np.random.randint(low=0, high=X_train.shape[0]-1, size = 1)
sample[0]
ind = names_train[sample[0]]
ind

