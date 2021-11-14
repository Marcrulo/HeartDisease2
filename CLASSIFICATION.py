import numpy as np
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from toolbox_02450 import train_neural_net
import torch
import pingouin

from compute_pca import apply_pca
from READ_DATA import classification_data, classification_target

# %% Import data
X = classification_data.to_numpy()
y = classification_target.to_numpy()
#y.shape = (y.shape[0], 1)
N, M = X.shape

# %% Cross validation
K = 10
#CV = model_selection.RepeatedKFold(n_splits=K,n_repeats=3)
CV = model_selection.KFold(K,shuffle=True)


# %% Feature transformation

# PCA 
X = apply_pca(X)
#X = zscore(X,ddof=1)


# %% Inner fold
def inner_fold(X_train,y_train,X_test,y_test,k):
    
    h_list = [1,2,4,8,16,32,64,128,256,512,1024]
    min_loss = float('inf')
    opt_h = None
    
    for h in h_list:
        # Parameters for neural network classifier
        n_hidden_units = h      # number of hidden units
        n_replicates = 3        # number of networks trained in each k-fold
        max_iter = 10
        
        model = lambda: torch.nn.Sequential(
                    torch.nn.Linear(M, n_hidden_units),
                    torch.nn.Tanh(),                           
                    torch.nn.Linear(n_hidden_units, 1), 
                    torch.nn.Sigmoid() 
                    )
        loss_fn = torch.nn.BCELoss()
        
        print(f'h={h}, k={k}')
        net, final_loss, learning_curve = train_neural_net(model,
                                                           loss_fn,
                                                           X=X_train,
                                                           y=y_train,
                                                           n_replicates=n_replicates,
                                                           max_iter=max_iter)
        y_sigmoid = net(X_test) 
        y_test_est = (y_sigmoid > .5).type(dtype=torch.uint8) 
        y_test = y_test.type(dtype=torch.uint8) 
        e = (y_test_est != y_test)
        error_rate = (sum(e).type(torch.float)/len(y_test)).data.numpy()
        
        if error_rate < min_loss:
            min_loss = error_rate
            opt_h = h
            
    
    return min_loss, opt_h


# %% 
lambda_interval = np.logspace(-8, 2, 50)
logistic_error = {} # k: (error, lambda)
ann_error = {} # k: (error, h)
baseline_error = {} # k: error

z1 = []
z2 = []
z3 = []

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
    z1.append(min_error)
    
    # Baseline
    base = np.ones(y_test.shape)
    baseline_error[k] = np.mean(base.squeeze() != y_test.squeeze())
    z2.append(baseline_error[k])
    
    # ANN
    X_train = torch.Tensor(X[train_index,:])
    y_train = torch.Tensor(y[train_index]).float().unsqueeze(1)
    X_test = torch.Tensor(X[test_index,:])
    y_test = torch.Tensor(y[test_index]).float().unsqueeze(1)
    
    
    opt_err, opt_h = inner_fold(X_train,y_train,X_test,y_test,k)
    ann_error[k] = (opt_err[0],opt_h)
    z3.append(opt_err[0])
    
    
# %%
p12 = pingouin.ttest(z3,z1,paired=True)  # ANN vs Logistic
p23 = pingouin.ttest(z1,z2,paired=True)  # Logistic vs base
p31 = pingouin.ttest(z3,z2,paired=True)  # ANN vs base

print("p-values:")
print(p12['p-val']['T-test'])
print(p23['p-val']['T-test'])
print(p31['p-val']['T-test'])       

print('\nconfidence intervals:')
print(p12['CI95%']['T-test'])
print(p23['CI95%']['T-test'])
print(p31['CI95%']['T-test'])    
    

#print(logistic_error)
#print()
#print(ann_error)
#print()
#print(baseline_error)
