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


os.chdir('C:/Users/Simon/Desktop/Data Science/Kaggle/Digit Recognizer')

#%%
train = pd.read_csv('train.csv')
x_test = pd.read_csv('test.csv')

y_train = train['label']
x_train = train.drop(['label'], axis = 1)

del train

#%%    Check distribution of label

y_train.value_counts()


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
# Obtain the PCA solution by calculating the SVD of Y
U,S,V = linalg.svd(Y,full_matrices=False)
V = V.T


#%% Classification

K = [15,20,30,40,50,60,100,150]

y_estimated = np.empty((len(K), Ntest))


for i in range(len(K)):
    
    # Project data onto principal component space,
    Z = Y @ V[:,:K[i]]
    Ztest = Ytest @ V[:,:K[i]]
    
    # Classify data with knn classifier
    knn_classifier = KNeighborsClassifier(n_neighbors=1)
    knn_classifier.fit(Z,y_train.ravel())
    y_estimated[i,:] = knn_classifier.predict(Ztest)


#%% Print results in desired DF

ImageId = np.arange(1,Ntest+1).reshape((Ntest,1))
y_final = y_estimated.T

results = pd.DataFrame(np.hstack((ImageId, y_final)))

results.to_csv('VicIsAChunk.csv')

