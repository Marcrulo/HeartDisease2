# exercise 8.1.1

from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid)
import numpy as np
from sklearn import model_selection
from toolbox_02450 import rlr_validate, train_neural_net
import torch
from compute_pca import apply_pca
import pingouin

# %%
from READ_DATA import regression_data, regression_target, regression_names
X = regression_data.to_numpy()
y = regression_target.to_numpy()
attributeNames = regression_names.to_numpy().tolist()
N, M = X.shape

X = apply_pca(X)

X = np.concatenate((np.ones((N,1)),X),1)
M=M+1
attributeNames = [u'offset']+attributeNames




# %% Prepare data

K = 10 
CV = model_selection.KFold(K, shuffle=True)


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



# %% RLR
internal_cross_validation = 10  
lambdas = np.power(10.,range(-1,7))

lambdas_at_k = {}

# %%
loss_at_k = {}
z1 = np.empty((K,1))

def inner_fold(X_train,y_train,X_test,y_test,k):
    
    
    #max_h = 152
    h_list = [1,2,4,8,16,32,64,128,256,512,1024]
    min_loss = float('inf')
    opt_h = None
    
    for h in h_list:
        # Parameters for neural network classifier
        n_hidden_units = h      # number of hidden units
        n_replicates = 3        # number of networks trained in each k-fold
        max_iter = 10000
        
        model = lambda: torch.nn.Sequential(
                            torch.nn.Linear(M, n_hidden_units), 
                            torch.nn.Tanh(),   
                            torch.nn.Linear(n_hidden_units, 1)
                            )
        loss_fn = torch.nn.MSELoss()
        
        print(f'h={h}, k={k}')
        net, final_loss, learning_curve = train_neural_net(model,
                                                           loss_fn,
                                                           X=X_train,
                                                           y=y_train,
                                                           n_replicates=n_replicates,
                                                           max_iter=max_iter)
        
        y_test_est = net(X_test)
        se = (y_test_est.float()-y_test.float())**2 # squared error
        mse = (sum(se).type(torch.float)/len(y_test)).data.numpy() #mean
        if mse < min_loss:
            min_loss = mse
            opt_h = h
            
    
    return min_loss, opt_h

# %%



# OUTER CROSS VALIDATION
k=0
for train_index, test_index in CV.split(X,y):
    # RLR DATA
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
      
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
        
    # with regularization with optimal lambda
    Error_train_rlr[k] = np.square(y_train-X_train @ w_rlr[:,k]).sum(axis=0)/y_train.shape[0]
    Error_test_rlr[k] = np.square(y_test-X_test @ w_rlr[:,k]).sum(axis=0)/y_test.shape[0]

    lambdas_at_k[k]=opt_lambda
    
    ###########################################################################
    # ANN DATA
    X_train = torch.Tensor(X[train_index,:])
    y_train = torch.Tensor(y[train_index]).float().unsqueeze(1)
    X_test = torch.Tensor(X[test_index,:])
    y_test = torch.Tensor(y[test_index]).float().unsqueeze(1)
    
    
    opt_err, opt_h = inner_fold(X_train,y_train,X_test,y_test,k)
    loss_at_k[(k,opt_h)] = opt_err
    z1[k]=opt_err
    
    
    
    

    k+=1

show()
# %%
p12 = pingouin.ttest(z1.squeeze(),Error_test_rlr.squeeze(),paired=True)                     # ANN vs Lin
p23 = pingouin.ttest(Error_test_rlr.squeeze(),Error_test_nofeatures.squeeze(),paired=True)  # Lin vs base
p31 = pingouin.ttest(z1.squeeze(),Error_test_nofeatures.squeeze(),paired=True)              # ANN vs base

print("p-values:")
print(p12['p-val']['T-test'])
print(p23['p-val']['T-test'])
print(p31['p-val']['T-test'])       

print('\nconfidence intervals:')
print(p12['CI95%']['T-test'])
print(p23['CI95%']['T-test'])
print(p31['CI95%']['T-test'])
 