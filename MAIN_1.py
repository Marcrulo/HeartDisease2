# %% Import data
from READ_DATA import regression_data, regression_target
from sklearn.model_selection import train_test_split
import numpy as np

# %% Prepare data
X = regression_data
y = regression_target

# Split train/test data
test_size = .5
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

# Feature transformation
mu = np.mean(X_train, 0)
sigma = np.std(X_train, 0)

X_train = (X_train - mu) / sigma
X_test = (X_test - mu) / sigma