
# coding: utf-8

# # Exploratory Data Analysis

# In[1]:

import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from skimage.io import imread, imshow
from skimage import transform, img_as_float
import glob
import math
from importlib import reload


# In[2]:

cwd = os.getcwd()
path = os.path.join(cwd, '..', 'src')
if not path in sys.path:
    sys.path.append(path)
del cwd, path


# In[3]:

import KaggleAmazonMain


# In[29]:

reload(KaggleAmazonMain)


# In[26]:

X_train, y_train, names_train, tagged_df = KaggleAmazonMain.load_training_data()


# In[15]:

X_train.shape


# In[16]:

tagged_df.head()


# In[17]:

#Barplot of tag counts
get_ipython().magic('matplotlib inline')
plt.rcParams['figure.figsize'] = (12, 5)
print('There are {} unique tags in this data'.format(len(tagged_df.columns)))
colors = cm.rainbow(np.linspace(0, 1, len(tagged_df.columns)))
tagged_df.sum().sort_values(ascending=False).plot(title="Counts of Tags", color=colors, kind='bar')
plt.show()
tagged_df.sum().sort_values(ascending=False)


# # Load Image Data

# In[18]:

len(y_train)


# In[19]:

# 100 files, images are 256x256 pixels, with a channel dimension size 3 = RGB
print('X_train is a {} object'.format(type(X_train)))
print('it has shape {}'.format(X_train.shape))


# In[20]:

print('y_train is a {} object'.format(type(y_train)))
print('it has {} elements'.format(len(y_train)))
print('each element is of type {}'.format(type(y_train[0])))
print('and the elements are of size {}'.format(y_train[0].shape))


# In[21]:

print('names_train is a {} object'.format(type(names_train)))
print('it has {} elements'.format(len(names_train)))
print('each element is of type {}'.format(type(names_train)))


# In[30]:

KaggleAmazonMain.plot_samples(X_train, names_train, tagged_df, 4,4)


# # Feature Engineering
# What type of features are we working with here?

# In[31]:

fig, axes = plt.subplots(1, 3, figsize=(15, 12))
axes[0].imshow(X_train[0,:,:,0], cmap='gray')
axes[1].imshow(X_train[0,:,:,1])
axes[2].imshow(X_train[0,:,:,2])
plt.axis('off')


# In[32]:

pics = ['train_10039', 'train_10059', 'train_10034']


# In[35]:

plt.subplots_adjust(wspace=0, hspace=0)
for i in range(0,3):
    sample = np.random.randint(low=0, high=X_train.shape[0]-1, size = 1)
    ind = names_train[sample[0]]
    tags = KaggleAmazonMain.get_labels(ind, tagged_df)
    KaggleAmazonMain.plot_rgb_dist(X_train[sample[0],:,:,:],tags)


# In[36]:

sample = np.random.randint(low=0, high=X_train.shape[0]-1, size = 1)
sample[0]
ind = names_train[sample[0]]
ind

