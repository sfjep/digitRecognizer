# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 11:37:11 2020

@author: Simon
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

# LOAD DATA
X_train = pd.read_csv('train.csv')
y_train = X_train['label'].values
X_train = X_train.drop(['label'], axis = 1)
#
## SPLIT INTO TEST AND TRAIN
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# CONVERT TO ARRAYS
X_train = np.asarray(X_train)
#X_test = np.asarray(X_test)
y_train = np.asarray(y_train)
#y_test = np.asarray(y_test)

# NORMALIZE TO MAKE ANN TRAIN FASTER
X_train = X_train / 255.0
#X_test = X_test / 255.0


# RESHAPE DATA FROM 2D TO 1D 
'''
-1: We need to reshape all the images
28*28 the size of the resultant 1D vector with 784 pixels
'''
X_train = X_train.reshape(-1, 28*28)
#X_test = X_test.reshape(-1, 28*28)


# CREATE MODEL OBJECT, THIS OBJECT WILL BE AN INSTANCE OF CLASS SEQUENTIAL

model = tf.keras.models.Sequential()


'''
Hyperparameters:
    Numer of neurons: 128 
    Activiation function ReLU
    input_shape: (784,)

Meaning: Output has 128 neurons in which we apply the ReLU activation function to break the 
lineaily and the input shape is (784,)

'''

model.add(tf.keras.layers.Dense(units = 128, activation = 'relu', input_shape = (784,)))

# ADD DROPOUT LAYER
'''
This is a regularization technique where we randomly set neurons in a layer to zero.
In this way, some percentage of neurons won't be updated, the whole training process is long, 
and we have less chance of overfitting
'''

model.add(tf.keras.layers.Dropout(0.4))

# ADD OUTPUT LAYER
'''
units = number of classes (10 in the Fashion MNIST)
activation = softmax(Returns a probability of the class)
'''

model.add(tf.keras.layers.Dense(units = 10, activation = 'softmax'))


# COMPILE THE MODEL
'''
It means that we have to connect the whole network to an optimizer and choose a loss.
An optimizer is a tool that will update the weights during the stochastic gradient descent, 
i.e. backpropagatig your loss into the Neural Network
'''

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['sparse_categorical_accuracy'])

# MODEL TRAIN
'''
We train the model using model.fit(), and we pass three arguments to inside the method
input: X_train to feed the network with the data
output: y_train, the correct data to classifify
no.of.epochs: The number of times you are going to train the network with the dataset.
'''

model.fit(X_train, y_train, epochs = 40)

# EVALUATE PERFORMANCE

test_loss, test_accuracy = model.evaluate(X_test, y_test)



X_test = pd.read_csv('test.csv')
X_test = np.asarray(X_test)
X_test = X_test / 255.0
X_test = X_test.reshape(-1, 28*28)

out = model.predict_classes(X_test)

out = pd.DataFrame(out)
out = out.reset_index()
out.columns = ['ImageId', 'Label']
out['ImageId'] = out['ImageId'] + 1 
out.to_csv('submission_ANN_v3.csv', index = False)

