# %% Import data
from READ_DATA import regression_data, regression_target
from sklearn.model_selection import train_test_split
import numpy as np

# %% Prepare data
names = regression_data.columns
X = regression_data.to_numpy()
y = regression_target.to_numpy()
N, M = X.shape


# Feature transformation

#mu = np.mean(X_train, 0)
#sigma = np.std(X_train, 0)
#X_train = (X_train - mu) / sigma
#X_test = (X_test - mu) / sigma

# lambda
#lambdas = np.power(10.,range(-5,9))
