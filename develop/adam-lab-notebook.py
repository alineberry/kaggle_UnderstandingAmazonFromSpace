
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
from skimage import transform, img_as_float, filters
from skimage.color import rgb2gray
import glob
import math
from importlib import reload
import scipy


# ### Import the custom KaggleAmazonMain module

# In[2]:

cwd = os.getcwd()
path = os.path.join(cwd, '..', 'src')
if not path in sys.path:
    sys.path.append(path)
del cwd, path

import KaggleAmazonMain


# In[3]:

reload(KaggleAmazonMain)


# # Load training image data

# Load from pickle unless something has changed:

# In[60]:

X_train = pd.read_pickle('X_train.pkl')
y_train = pd.read_pickle('y_train.pkl')


# In[62]:

type(y_train)


# In[64]:

y_train.describe()


# In[65]:

tagged_df.describe()


# In[68]:

tagged_df.ix[tagged_df['cultivation']==2]


# Below cell will recreate the feature matrix. Use with caution as this may take around 30 minutes to complete.

# In[4]:

# X_train, y_train, names_train, tagged_df = KaggleAmazonMain.load_training_data(sampleOnly=False)
# X_train.to_pickle('X_train.pkl')
# y_train.to_pickle('y_train.pkl')


# In[13]:

X_train.head()


# In[21]:

X_train.describe()


# In[16]:

y_train


# In[17]:

y_train.describe()


# ### See distribution of label counts. Note a significant imbalance.

# In[6]:

#Barplot of tag counts
get_ipython().magic('matplotlib inline')
plt.rcParams['figure.figsize'] = (12, 5)
print('There are {} unique tags in this data'.format(len(tagged_df.columns)))
colors = cm.rainbow(np.linspace(0, 1, len(tagged_df.columns)))
tagged_df.sum().sort_values(ascending=False).plot(title="Counts of Tags", color=colors, kind='bar')
plt.show()
tagged_df.sum().sort_values(ascending=False)


# ### Get a feel for the size and shape of the data

# In[7]:

# n files, images are 256x256 pixels, with a channel dimension size 3 = RGB
print('X_train is a {} object'.format(type(X_train)))
print('it has shape {}'.format(X_train.shape))


# In[10]:

print('y_train is a {} object'.format(type(y_train)))
print('it has {} elements'.format(len(y_train)))


# In[11]:

print('names_train is a {} object'.format(type(names_train)))
print('it has {} elements'.format(len(names_train)))
print('each element is of type {}'.format(type(names_train)))


# # Exploratory plotting

# ## Plot some random images with their labels

# In[118]:

KaggleAmazonMain.plot_samples(X_train, names_train, tagged_df, 4,4)


# ## Plot images with labels and their RGB intensity distributions

# In[16]:

plt.subplots_adjust(wspace=0, hspace=0)
for i in range(0,3):
    sample = np.random.randint(low=0, high=X_train.shape[0]-1, size = 1)
    ind = names_train[sample[0]]
    tags = KaggleAmazonMain.get_labels(ind, tagged_df)
    KaggleAmazonMain.plot_rgb_dist(X_train[sample[0],:,:,:],tags)


# ## Plot sobels of some images with labels

# In[18]:

X_train_g = rgb2gray(X_train)

X_train_sobel = []
for i in range(X_train_g.shape[0]):
    X_train_sobel.append(filters.sobel(X_train_g[i]))
X_train_sobel = np.asarray(X_train_sobel)


# In[19]:

KaggleAmazonMain.plot_samples(X_train_sobel, names_train, tagged_df, 4,4)


# # Develop predictive models

# In[37]:

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# In[76]:

rf = RandomForestClassifier(n_estimators = 100, 
                            max_features = 'sqrt',
                            bootstrap = True, 
                            oob_score = True,
                            n_jobs = -1,
                            random_state = 14113,
                            class_weight = 'balanced')


# In[77]:

y_train[y_train > 1] = 1


# In[78]:

X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.40, random_state=14113)


# In[79]:

rf.fit(X_train, y_train)


# In[80]:

predictions = rf.predict(X_validation)


# In[81]:

from sklearn.metrics import fbeta_score


# In[51]:

np.asarray(y_validation)


# ### there were 9 test observations which were given no predictions (labels)

# In[99]:

zeros = pd.DataFrame(np.sum(predictions,axis=1))
sum(np.asarray(zeros) == 0)


# In[82]:

fbeta_score(np.asarray(y_validation), predictions, beta=2, average='samples')

