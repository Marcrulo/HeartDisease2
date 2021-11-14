import numpy as np
import pandas as pd
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

from compute_pca import apply_pca
from READ_DATA import classification_data, classification_target
from inner_fold import inner_fold

# %% Import data
X = classification_data.to_numpy()
y = classification_target.to_numpy()
#y.shape = (y.shape[0], 1)
N, M = X.shape

# %% Cross validation
K = 10
CV = model_selection.KFold(K,shuffle=True)

# %% Feature transformation

# PCA or normal standardization???
X = apply_pca(X)
#X = zscore(X,ddof=1)


# %% 
lambda_interval = np.logspace(-8, 2, 50)

logistic_error = {} # k: (error, lambda)

ann_test_error_rate= [] 
ann_optimal_h=[]

for k, (train_index, test_index) in enumerate(CV.split(X,y)): 
    
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]    
    
    # Logistic regression
    test_error_rate = np.zeros(len(lambda_interval))
    
    for lambda_i in range(0, len(lambda_interval)):
        mdl = LogisticRegression(penalty='l2', C=1/lambda_interval[lambda_i] )
        
        mdl.fit(X_train, y_train)

        y_test_est = mdl.predict(X_test).T        
        
        test_error_rate[lambda_i] = np.sum(y_test_est != y_test) / len(y_test)

    min_error = np.min(test_error_rate)
    opt_lambda_idx = np.argmin(test_error_rate)
    
    logistic_error[k] = (min_error, lambda_interval[opt_lambda_idx])

    
    # ANN
        

    

print(logistic_error)

