#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 18:20:57 2021

@author: marc8165
"""

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
plt.rcParams.update({'font.size': 12})
classNames = ["0", "1"]

K = 10
CV = model_selection.KFold(K,shuffle=True)

#decision_boundaries = plt.figure(1, figsize=(10,10))
#subplot_size_1 = int(np.floor(np.sqrt(K))) 
#subplot_size_2 = int(np.ceil(K/subplot_size_1))
#plt.suptitle('Data and model decision boundaries', fontsize=20)
#plt.subplots_adjust(left=0, bottom=0, right=1, top=.9, wspace=.5, hspace=0.25)

#summaries, summaries_axes = plt.subplots(1, 2, figsize=(10,5))
#color_list = ['tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink','tab:gray', 'tab:olive', 'tab:cyan', 'tab:red', 'tab:blue']

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
    errors.append(error_rate) # store error rate for current CV fold 
    
       
    
    # Display the learning curve for the best net in the current fold
    #h, = summaries_axes[0].plot(learning_curve, color=color_list[k])
    #h.set_label('CV fold {0}'.format(k+1))
    #summaries_axes[0].set_xlabel('Iterations')
    #summaries_axes[0].set_xlim((0, max_iter))
    #summaries_axes[0].set_ylabel('Loss')
    #summaries_axes[0].set_title('Learning curves')
    
# Display the error rate across folds
#summaries_axes[1].bar(np.arange(1, K+1), np.squeeze(np.asarray(errors)), color=color_list)
#summaries_axes[1].set_xlabel('Fold');
#summaries_axes[1].set_xticks(np.arange(1, K+1))
#summaries_axes[1].set_ylabel('Error rate');
#summaries_axes[1].set_title('Test misclassification rates')
    
# Show the plots
# plt.show(decision_boundaries.number) # try these lines if the following code fails (depends on package versions)
# plt.show(summaries.number)
plt.show()

# Display a diagram of the best network in last fold
#print('Diagram of best neural net in last fold:')
#weights = [net[i].weight.data.numpy().T for i in [0,2]]
#biases = [net[i].bias.data.numpy() for i in [0,2]]
#tf =  [str(net[i]) for i in [1,3]]
#draw_neural_net(weights, biases, tf)

# Print the average classification error rate
print('\nGeneralization error/average error rate: {0}%'.format(round(100*np.mean(errors),4)))
