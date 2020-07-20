# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 13:34:49 2020

@author: Simon
"""

#%%
import os
import pandas as pd
import numpy as np
from matplotlib.pyplot import (figure, subplot, plot, xlabel, ylabel, title, 
yticks, show,legend,imshow, cm)
from scipy.io import loadmat
import scipy.linalg as linalg
from sklearn.neighbors import KNeighborsClassifier
from scipy import stats
from sklearn.model_selection import train_test_split


os.chdir('C:/Users/Simon/Desktop/Data Science/Kaggle/Digit Recognizer')

#%% Split train data into test and train

train = pd.read_csv('train.csv')

x_train, x_test = train_test_split(train, test_size = 0.2)


#%%
y_train = x_train['label']
x_train = x_train.drop(['label'], axis = 1)

y_test = x_test['label']
x_test = x_test.drop(['label'], axis = 1)


#%%    Check distribution of label

y_test.value_counts()


#%%     Check for missing data in X

x_train.isnull().any().describe()
x_test.isnull().any().describe()

#%%    Normalize the data

x_train = x_train / 255.0
x_test = x_test / 255.0

#%%    CHANGE TO NUMPY ARRAYS
x_train = x_train.values
x_test = x_test.values
y_train = y_train.values

#%%

# Number of principal components to use for classification,
# i.e. the reduced dimensionality

N,M = x_train.shape
Ntest = x_test.shape[0]

Y = x_train - np.ones((N,1))*x_train.mean(0)
Ytest = x_test - np.ones((Ntest,1))*x_train.mean(0)

#%%    PCA
# Obtain the PCA solution  by calculate the SVD of Y
U,S,V = linalg.svd(Y,full_matrices=False)
V = V.T



#%%   KNN Classification

# Repeat classification for different values of K
error_rates = []

K = [15,20,30,40,50,60,100,150]

for k in K:
    # Project data onto principal component space,
    Z = Y @ V[:,:k]
    Ztest = Ytest @ V[:,:k]

    # Classify data with knn classifier
    knn_classifier = KNeighborsClassifier(n_neighbors=3)
    knn_classifier.fit(Z,y_train.ravel())
    y_estimated = knn_classifier.predict(Ztest)

    # Compute classification error rates
    y_estimated = y_estimated.T
    er = (sum(y_test!=y_estimated)/float(len(y_test)))*100
    error_rates.append(er)
    print('K={0}: Error rate: {1:.1f}%'.format(k, er))

# Visualize error rates vs. number of principal components considered
figure()
plot(K,error_rates,'o-')
xlabel('Number of principal components K')
ylabel('Error rate [%]')
show()

# Notes: 3NN seemed slightly better than 1NN. Number of PC's (K) seem to be optimized around 40


#%%     Decision tree