
# coding: utf-8

# # Exploratory Data Analysis

# In[5]:

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

# In[6]:

cwd = os.getcwd()
path = os.path.join(cwd, '..', 'src')
if not path in sys.path:
    sys.path.append(path)
del cwd, path

import KaggleAmazonMain


# In[7]:

reload(KaggleAmazonMain)


# # Load training image data

# Load from pickle unless something has changed:

# In[76]:

X = pd.read_pickle('X_train.pkl')
y = pd.read_pickle('y_train.pkl')


# In[5]:

type(y_train)


# In[6]:

y_train.describe()


# Below cell will recreate the feature matrix. Use with caution as this may take around 30 minutes to complete.

# In[4]:

# X, y, names_train, tagged_df = KaggleAmazonMain.load_training_data(sampleOnly=False)
# X.to_pickle('X.pkl')
# y.to_pickle('y.pkl')


# In[8]:

X_train.head()


# In[9]:

X_train.describe()


# In[10]:

y_train


# In[11]:

y_train.describe()


# ### See distribution of label counts. Note a significant imbalance.

# In[12]:

#Barplot of tag counts
get_ipython().magic('matplotlib inline')
plt.rcParams['figure.figsize'] = (12, 5)
print('There are {} unique tags in this data'.format(len(tagged_df.columns)))
colors = cm.rainbow(np.linspace(0, 1, len(tagged_df.columns)))
tagged_df.sum().sort_values(ascending=False).plot(title="Counts of Tags", color=colors, kind='bar')
plt.show()
tagged_df.sum().sort_values(ascending=False)


# ### Get a feel for the size and shape of the data

# In[13]:

# n files, images are 256x256 pixels, with a channel dimension size 3 = RGB
print('X_train is a {} object'.format(type(X_train)))
print('it has shape {}'.format(X_train.shape))


# In[14]:

print('y_train is a {} object'.format(type(y_train)))
print('it has {} elements'.format(len(y_train)))


# In[15]:

print('names_train is a {} object'.format(type(names_train)))
print('it has {} elements'.format(len(names_train)))
print('each element is of type {}'.format(type(names_train)))


# # Exploratory plotting

# ## Plot some random images with their labels

# In[16]:

KaggleAmazonMain.plot_samples(X_train, names_train, tagged_df, 4,4)


# ## Plot images with labels and their RGB intensity distributions

# In[17]:

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

# In[22]:

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import GridSearchCV


# ### Create test-train split

# In[33]:

X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.40, random_state=14113)


# **need to fix this in the right place**. It is fixed in KaggleAmazonMain.py, but need to reload to update the pickle file

# In[62]:

y_train[y_train > 1] = 1
y_validation[y_validation > 1] = 1


# ## Search random forest hyperparameter space 
# - use GridSearchCV
# - change score to the f2 score

# In[38]:

f2_score = make_scorer(fbeta_score, beta=2, average='samples')

rf = RandomForestClassifier(bootstrap = True, 
                            oob_score = False,
                            n_jobs = -1,
                            random_state = 14113
                            )

parameters = {
    'n_estimators' : [100, 200, 300],
    'max_features' : ['sqrt', 'log2', 1, 2, 0.5, None],
    'class_weight' : ['balanced', 'balanced_subsample']
}

grid_search_obj = GridSearchCV(rf, parameters, scoring=f2_score, n_jobs=-1, cv=3)

grid_search_obj.fit(X_train, y_train)


# Pickle results for future use

# In[57]:

grid_search_results = pd.DataFrame(grid_search.cv_results_)
grid_search_results.to_pickle('grid_search_results_df.pkl')

import pickle
pickle.dump(grid_search, open('grid_search_object.pkl', "wb"))


# Load grid search results from pickle if needed in future

# In[56]:

grid_search_results = pd.read_pickle('grid_search_results_df.pkl')


# In[58]:

grid_search_obj = pickle.load(open('grid_search_object.pkl', "rb"))


# ### Print out grid search results

# In[43]:

grid_search.best_score_


# In[44]:

grid_search.best_params_


# In[45]:

grid_search.best_estimator_


# ### Predict on validation set using *best estimator* and calculate F2 score

# In[60]:

predictions_best = grid_search.best_estimator_.predict(X_validation)


# In[63]:

fbeta_score(np.asarray(y_validation), predictions_best, beta=2, average='samples')


# ### Tinker with threshold

# In[64]:

predict_probas_best = grid_search.best_estimator_.predict_proba(X_validation)


# In[71]:

predict_probas_best[0].shape


# In[129]:

def get_prediction_matrix(probs, threshold):
    """
    Input is a matrix of probabilities, and a classification threshold
    Output is a binary matrix of predictions
    """
    
    # need to work with an n x #outcomes matrix where elements are probabilities of class 1
    if type(probs) is list:
        probs = restructure_probs_matrix(probs)
    
    return (probs > threshold).astype(int)


# In[128]:

test=restructure_probs_matrix(predict_probas_best)
(test>0.3).astype(int)


# In[130]:

get_prediction_matrix(predict_probas_best, 0.3)


# In[116]:

def restructure_probs_matrix(probs):
    probs_r = probs[0][:,1]
    for arr in probs[1:]:
        probs_r = np.column_stack((probs_r,arr[:,1]))
    return probs_r


# In[ ]:

best_score = 0
best_thresh = 0
scores = []
thresh = np.arange(0, 0.6, 0.02)
for t in thresh:  # loop through possible threshold values
    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.30)
    y_train = np.asarray(y_train); y_validation = np.asarray(y_validation)
    y_train[y_train > 1] = 1; y_validation[y_validation > 1] = 1
    rf = RandomForestClassifier(bootstrap=True, class_weight='balanced',
                criterion='gini', max_depth=None, max_features='sqrt',
                max_leaf_nodes=None, min_impurity_split=1e-07,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=300, n_jobs=-1,
                oob_score=False, verbose=0,
                warm_start=False)
    rf.fit(X_train, y_train)
    probs = rf.predict_proba(X_validation)
    yhat = get_prediction_matrix(probs,t)
    #print(yhat.shape, y_validation.shape)
    score = fbeta_score(np.asarray(y_validation), yhat, beta=2, average='samples')
    if score > best_score:
        best_score = score
        best_thresh = t
    scores.append(score)
    print('threshold: {}\tscore: {}\tbest score: {}\n'.format(t,score,best_score))
print('\nbest score is: {} at a threshold of {}'.format(best_score, best_thresh))


# In[ ]:

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.plot(thresh, scores)


# In[103]:

def best_threshold(feature_index):
    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.30)
    y_train = np.asarray(y_train); y_validation = np.asarray(y_validation)
    y_train[y_train > 1] = 1; y_validation[y_validation > 1] = 1
    rf = RandomForestClassifier(bootstrap=True, class_weight='balanced',
                criterion='gini', max_depth=None, max_features='sqrt',
                max_leaf_nodes=None, min_impurity_split=1e-07,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=300, n_jobs=-1,
                oob_score=False, verbose=0,
                warm_start=False)
    rf.fit(X_train, y_train)
    probs = np.asarray(rf.predict_proba(X_validation))
    predict = np.asarray(rf.predict(X_validation))

    best_score = 0
    best_thresh = 0
    thresh = np.arange(0.1, 0.6, 0.05)
    for t in thresh:  # loop through possible threshold values
        print('t:',t)
        for j in range(len(probs[feature_index])): # loop through data predictions
            if probs[feature_index][j,1] > t and probs[feature_index][j,1] < 0.5:
                #print('found a {}'.format(probs[feature_index][j,1]))
                predict[j,feature_index] = 1
        score = fbeta_score(np.asarray(y_validation), predict, beta=2, average='samples')
        print('score:',score)
        print('best score:', best_score)
        if score > best_score:
            best_score = score
            best_thresh = t
    print('best score is: {} at a threshold of {}'.format(best_score, best_thresh))
    return best_thresh, best_score


# In[104]:

feature = 'blow_down'
feature_index = y_train.columns.get_loc(feature)
best_threshold(feature_index)


# In[105]:

thresholds = []
for i in range(len(y_train.columns)):
    thresholds.append(best_threshold(i))
thresholds


# In[86]:

best_threshold(0)


# In[89]:

best_threshold(1)


# In[88]:

y_train.sum()


# In[77]:

grid_search.best_estimator_


# In[ ]:




# In[ ]:




# In[74]:

len(predict_probas_best[0]), len(predict_probas_best)


# In[72]:

predictions_best.shape


# ## Fit a new random forest

# In[78]:

X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.40, random_state=14113)


# In[32]:

rf.fit(X_train, y_train)


# In[80]:

predictions = rf.predict(X_validation)


# In[81]:

from sklearn.metrics import fbeta_score


# In[51]:

np.asarray(y_validation)


# In[ ]:




# In[ ]:




# In[ ]:




# ### there were 9 test observations which were given no predictions (labels)

# In[99]:

zeros = pd.DataFrame(np.sum(predictions,axis=1))
sum(np.asarray(zeros) == 0)


# In[82]:

fbeta_score(np.asarray(y_validation), predictions, beta=2, average='samples')

