from sklearn import model_selection
from MAIN_1 import X,y
import numpy as np

# k-fold CV
K=10
CV = model_selection.KFold(K, shuffle=True)

# lambda
lambdas = np.power(10.,range(-5,9))


for train_index, test_index in CV.split(X,y):
    
    # D_k^train and D_k^test
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    
    # Standardize
    mu = np.mean(X_train, 0)
    sigma = np.std(X_train, 0)
    
    X_train = (X_train - mu) / sigma
    X_test = (X_test - mu) / sigma

    # 