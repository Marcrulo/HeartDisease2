# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 14:27:28 2021

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

# %% Import data

# Read
#my_data = np.genfromtxt('heart.csv', names=header, delimiter=',')
df=pd.read_csv('heart.csv')


N, M = df.shape

X =(df.iloc[:,:13]).to_numpy()
y=(df.iloc[:,-1]).to_numpy()




# %% Inspired by 8.1.2


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.95, stratify=y)

mu = np.mean(X_train, 0)
sigma = np.std(X_train, 0)

X_train = (X_train - mu) / (sigma+0.000001)
X_test = (X_test - mu) / (sigma+0.000001)



# %%

lambda_interval = np.logspace(-8, 2, 50)
train_error_rate = np.zeros(len(lambda_interval))
test_error_rate = np.zeros(len(lambda_interval))

for k in range(0, len(lambda_interval)):
    mdl = LogisticRegression(penalty='l2', C=1/lambda_interval[k] )
    
    mdl.fit(X_train, y_train)

    y_train_est = mdl.predict(X_train).T
    y_test_est = mdl.predict(X_test).T
    
    train_error_rate[k] = np.sum(y_train_est != y_train) / len(y_train)
    test_error_rate[k] = np.sum(y_test_est != y_test) / len(y_test)


min_error = np.min(test_error_rate)
opt_lambda_idx = np.argmin(test_error_rate)
opt_lambda = lambda_interval[opt_lambda_idx]

print(f"Minimum error {round(min_error,5)}\nOptimal lambda: {round(opt_lambda,5)}")

