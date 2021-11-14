# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 13:56:42 2021

@author: arian
"""

# %% Import packages/libraries
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import model_selection, tree
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy
from scipy.stats import kstest, norm, zscore
import statsmodels.api as sm
import pylab
from toolbox_02450 import feature_selector_lr, bmplot
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from toolbox_02450 import train_neural_net, draw_neural_net, visualize_decision_boundary
import torch

from matplotlib.pyplot import figure, plot, subplot, title, xlabel, ylabel, show, clim, hist, legend, ylim
import sklearn.linear_model as lm

from READ_DATA import classification_data, classification_target

X = classification_data.to_numpy()
y = classification_target.to_numpy()
y.shape = (y.shape[0], 1)
N, M = X.shape

# %%  Inspired by 8.2.2

K = 10
CV = model_selection.KFold(K,shuffle=True)

n_hidden_units = 20 
model = lambda: torch.nn.Sequential(
                    torch.nn.Linear(M, n_hidden_units),
                    torch.nn.Tanh(),                           
                    torch.nn.Linear(n_hidden_units, 1), 
                    torch.nn.Sigmoid() 
                    )

loss_fn = torch.nn.BCELoss()

max_iter = 100
print('Training model of type:\n{}\n'.format(str(model())))

errors = [] 

for k, (train_index, test_index) in enumerate(CV.split(X,y)): 
    print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K))    
   
    X_train = torch.Tensor(X[train_index,:] )
    y_train = torch.Tensor(y[train_index] )
    X_test = torch.Tensor(X[test_index,:] )
    y_test = torch.Tensor(y[test_index] )
    
    net, final_loss, learning_curve = train_neural_net(model,
                                                       loss_fn,
                                                       X=X_train,
                                                       y=y_train,
                                                       n_replicates=3,
                                                       max_iter=max_iter)
    
    print('\n\tBest loss: {}\n'.format(final_loss))
    
    y_sigmoid = net(X_test) 
    y_test_est = (y_sigmoid > .5).type(dtype=torch.uint8) 
    y_test = y_test.type(dtype=torch.uint8)

    e = (y_test_est != y_test)
    error_rate = (sum(e).type(torch.float)/len(y_test)).data.numpy()
    errors.append(error_rate) 
    
       
print('\nGeneralization error/average error rate: {0}%'.format(round(100*np.mean(errors),4)))
