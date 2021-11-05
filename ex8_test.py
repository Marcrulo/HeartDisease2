# exercise 8.1.1

from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid)
import numpy as np
from scipy.io import loadmat
import sklearn.linear_model as lm
from sklearn import model_selection
from toolbox_02450 import rlr_validate

# %%
from READ_DATA import regression_data, regression_target, regression_names
X = regression_data.to_numpy()
y = regression_target.to_numpy()
N, M = X.shape
X = np.concatenate((np.ones((N,1)),X),1)
M=M+1
attributeNames = regression_names.to_numpy().tolist()
attributeNames = [u'offset']+attributeNames

# %%

## Crossvalidation
K = 10 # outer fold
CV = model_selection.KFold(K, shuffle=True)

# Values of lambda
lambdas = np.power(10.,range(-1,7))

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


# OUTER CROSS VALIDATION
k=0
for train_index, test_index in CV.split(X,y):
    # EXTRACT TRAINING AND TEST SET FOR CURRENT CV FOLD
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    internal_cross_validation = 10    
    
    # INNER CROSS VALIDATION (VALIDATION ERRORS)
    opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = \
     rlr_validate(X_train, y_train, lambdas, internal_cross_validation)
    
    
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

    
    
    
    # TABLE CONTENT
    print("Outer fold k={}, lambda*={},rlr_error={},base_error={}".format(\
          k+1,opt_lambda,Error_test_rlr[k],Error_test_nofeatures[k]))
    
    

    # Display the results for the last cross-validation fold
    if k == K-1:
        
        figure(k, figsize=(12,8))
        subplot(1,2,1)
        semilogx(lambdas,mean_w_vs_lambda.T[:,1:],'.-') # Don't plot the bias term
        xlabel('Regularization factor')
        ylabel('Mean Coefficient Values')
        grid()
        #legend(attributeNames[1:], loc='best')
        
        subplot(1,2,2)
        title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
        loglog(lambdas,train_err_vs_lambda.T,'b.-',lambdas,test_err_vs_lambda.T,'r.-')
        xlabel('Regularization factor')
        ylabel('Squared error (crossvalidation)')
        legend(['Train error','Validation error'])
        grid()
    


    k+=1

show()

# Display results
def results():
    print('Linear regression without feature selection:')
    print('- Training error: {0}'.format(Error_train.mean()))
    print('- Test error:     {0}'.format(Error_test.mean()))
    print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train.sum())/Error_train_nofeatures.sum()))
    print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test.sum())/Error_test_nofeatures.sum()))
    
    print('Regularized linear regression:')
    print('- Training error: {0}'.format(Error_train_rlr.mean()))
    print('- Test error:     {0}'.format(Error_test_rlr.mean()))
    print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train_rlr.sum())/Error_train_nofeatures.sum()))
    print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test_rlr.sum())/Error_test_nofeatures.sum()))
    
    print('Weights in last fold:')
    for m in range(M):
        print('{:>15} {:>15}'.format(attributeNames[m], np.round(w_rlr[m,-1],2)))
        
results()
