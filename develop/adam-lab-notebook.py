
# coding: utf-8

# # Exploratory Data Analysis

# In[27]:

import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from skimage.io import imread, imshow
from skimage import transform, img_as_float, filters
from skimage.color import rgb2gray
import glob
import math
from importlib import reload
import scipy


# In[2]:

cwd = os.getcwd()
path = os.path.join(cwd, '..', 'src')
if not path in sys.path:
    sys.path.append(path)
del cwd, path


# In[3]:

import KaggleAmazonMain


# In[66]:

reload(KaggleAmazonMain)


# In[5]:

X_train, y_train, names_train, tagged_df = KaggleAmazonMain.load_training_data()


# In[6]:

X_train.shape


# In[7]:

tagged_df.head()


# In[8]:

#Barplot of tag counts
get_ipython().magic('matplotlib inline')
plt.rcParams['figure.figsize'] = (12, 5)
print('There are {} unique tags in this data'.format(len(tagged_df.columns)))
colors = cm.rainbow(np.linspace(0, 1, len(tagged_df.columns)))
tagged_df.sum().sort_values(ascending=False).plot(title="Counts of Tags", color=colors, kind='bar')
plt.show()
tagged_df.sum().sort_values(ascending=False)


# # Load Image Data

# In[9]:

len(y_train)


# In[10]:

# 100 files, images are 256x256 pixels, with a channel dimension size 3 = RGB
print('X_train is a {} object'.format(type(X_train)))
print('it has shape {}'.format(X_train.shape))


# In[11]:

print('y_train is a {} object'.format(type(y_train)))
print('it has {} elements'.format(len(y_train)))
print('each element is of type {}'.format(type(y_train[0])))
print('and the elements are of size {}'.format(y_train[0].shape))


# In[12]:

print('names_train is a {} object'.format(type(names_train)))
print('it has {} elements'.format(len(names_train)))
print('each element is of type {}'.format(type(names_train)))


# In[13]:

KaggleAmazonMain.plot_samples(X_train, names_train, tagged_df, 4,4)


# # Feature Engineering
# What type of features are we working with here?

# In[14]:

fig, axes = plt.subplots(1, 3, figsize=(15, 12))
axes[0].imshow(X_train[0,:,:,0], cmap='gray')
axes[1].imshow(X_train[0,:,:,1])
axes[2].imshow(X_train[0,:,:,2])
plt.axis('off')


# In[15]:

pics = ['train_10039', 'train_10059', 'train_10034']


# In[16]:

plt.subplots_adjust(wspace=0, hspace=0)
for i in range(0,3):
    sample = np.random.randint(low=0, high=X_train.shape[0]-1, size = 1)
    ind = names_train[sample[0]]
    tags = KaggleAmazonMain.get_labels(ind, tagged_df)
    KaggleAmazonMain.plot_rgb_dist(X_train[sample[0],:,:,:],tags)


# In[17]:

imshow(filters.sobel(rgb2gray(X_train[0,:,:,:])), cmap='gray')


# In[18]:

X_train_g = rgb2gray(X_train)

X_train_sobel = []
for i in range(X_train_g.shape[0]):
    X_train_sobel.append(filters.sobel(X_train_g[i]))
X_train_sobel = np.asarray(X_train_sobel)


# In[19]:

KaggleAmazonMain.plot_samples(X_train_sobel, names_train, tagged_df, 4,4)


# In[20]:

X_train_sobel = KaggleAmazonMain.xform_to_sobel(X_train)


# In[73]:

len(y_train)


# In[78]:

pd.DataFrame(y_train)


# In[63]:

features = KaggleAmazonMain.get_features(X_train)


# In[64]:

X_train[0].shape


# In[65]:

features.head()


# In[30]:

features.describe()


# In[79]:

from sklearn.ensemble import RandomForestClassifier


# In[82]:

len(features)


# In[83]:

y_train_df = pd.DataFrame(y_train)


# In[85]:

len(y_train_df)


# In[ ]:

rf = RandomForestClassifier()

