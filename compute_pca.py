from scipy import stats
import numpy as np

def apply_pca(X):
    # Normalize data
    X = stats.zscore(X)
                    
    # Normalize and compute PCA (change to True to experiment with PCA preprocessing)
    do_pca_preprocessing = False
    if do_pca_preprocessing:
        Y = stats.zscore(X,0)
        U,S,V = np.linalg.svd(Y,full_matrices=False)
        V = V.T
        #Components to be included as features
        k_pca = 3
        X = X @ V[:,:k_pca]
        N, M = X.shape
        
    return X