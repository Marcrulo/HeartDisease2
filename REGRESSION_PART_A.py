# exercise 8.1.1
from matplotlib import pyplot as plt
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid)
import numpy as np
from scipy.io import loadmat
import sklearn.linear_model as lm
from sklearn import model_selection
from toolbox_02450 import rlr_validate

from READ_DATA import regression_data, regression_target, regression_names
X = regression_data.to_numpy()
y = regression_target.to_numpy()
attributeNames = regression_names.to_numpy().tolist()
N, M = X.shape

#mat_data = loadmat('Data/body.mat')
#X = mat_data['X']
#y = mat_data['y'].squeeze()
#attributeNames = [name[0] for name in mat_data['attributeNames'][0]]
#N, M = X.shape

X = np.concatenate((np.ones((N,1)),X),1)
M=M+1
attributeNames = [u'offset']+attributeNames


## Crossvalidation
K = 10 # outer fold
CV = model_selection.KFold(K, shuffle=True)

# Values of lambda
lambdas = np.power(10.,range(-3,9))




# OUTER CROSS VALIDATION
gen_errors = []
for lambd in lambdas:
    
    k=0
    # Initialize variables
    Error_train = np.empty((K,1))
    Error_test = np.empty((K,1))
    Error_train_rlr = np.empty((K,1))
    Error_test_rlr = np.empty((K,1))
    Error_train_nofeatures = np.empty((K,1))
    Error_test_nofeatures = np.empty((K,1))
    
    w_rlr = np.empty((M,K))
    mu = np.empty((K, M-1))
    sigma = np.empty((K, M-1))
    w_noreg = np.empty((M,K))
    
    for train_index, test_index in CV.split(X,y):
        
        
        # EXTRACT TRAINING AND TEST SET FOR CURRENT CV FOLD
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]
            
        opt_lambda = lambd
        # INNER CROSS VALIDATION (VALIDATION ERRORS)
        #opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = \
        # rlr_validate(X_train, y_train, lambdas, internal_cross_validation)
        
        
        # STANDARDIZE OUTER FOLD TRAINING DATA
        mu[k, :] = np.mean(X_train[:, 1:], 0)
        sigma[k, :] = np.std(X_train[:, 1:], 0)
        X_train[:, 1:] = (X_train[:, 1:] - mu[k, :] ) / sigma[k, :] 
        X_test[:, 1:] = (X_test[:, 1:] - mu[k, :] ) / sigma[k, :] 
        Xty = X_train.T @ y_train
        XtX = X_train.T @ X_train
        
        
        # ESTIMATE WEIGHTS
        # optimal value of lambda, on entire training set
        lambdaI = opt_lambda * np.eye(M)
        lambdaI[0,0] = 0 # Do no regularize the bias term
        w_rlr[:,k] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
    
        # unregularized linear regression, on entire training set
        w_noreg[:,k] = np.linalg.solve(XtX,Xty).squeeze()
        
        
        # COMPUTE SQUARED ERROR (TEST ERRORS)
        # without using the input data at all
        Error_train_nofeatures[k] = np.square(y_train-y_train.mean()).sum(axis=0)/y_train.shape[0]
        Error_test_nofeatures[k] = np.square(y_test-y_test.mean()).sum(axis=0)/y_test.shape[0]  
        
        # without regularization
        Error_train[k] = np.square(y_train-X_train @ w_noreg[:,k]).sum(axis=0)/y_train.shape[0]
        Error_test[k] = np.square(y_test-X_test @ w_noreg[:,k]).sum(axis=0)/y_test.shape[0]
        
        # with regularization with optimal lambda
        Error_train_rlr[k] = np.square(y_train-X_train @ w_rlr[:,k]).sum(axis=0)/y_train.shape[0]
        Error_test_rlr[k] = np.square(y_test-X_test @ w_rlr[:,k]).sum(axis=0)/y_test.shape[0]
         
        k+=1
    
    print('\n Weights: lambda='+str(int(np.log10(opt_lambda))))
    for m in range(M):
        print('{:>15} {:>15}'.format(attributeNames[m], np.round(w_rlr[m,-1],10)))
    gen_errors.append(Error_test_rlr.mean())



plt.plot(np.log10(lambdas),gen_errors) # x-axis: 10^x
show()


    

