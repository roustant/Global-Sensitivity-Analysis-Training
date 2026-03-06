import numpy as np

# A toy model

d = 4

def model(X):
        
    k_indices = np.arange(1, d + 1)
    a_k = ((-1)**k_indices) / (k_indices + 1)
    
    terms = 1 / (1 + (X - a_k)**2)
    
    return np.prod(terms, axis=1)

eps = 1e-6

def model_gradient(X):
    nSample = X.shape[0]
    nVar = X.shape[1]
    gradient = np.zeros((nSample,nVar))
    
    for i in range(nVar):
        X_plus = X.copy()
        X_minus = X.copy()

        X_plus[:,i] += eps
        X_minus[:,i] -= eps
        gradient[:,i] = (model(X_plus) - model(X_minus)) / (2 * eps)

    return gradient

def getSample(size):
    
    X = np.zeros((size, d))

    for i in range(d):
        X[:, i] = np.random.uniform(-1, 1, size=size)

    return X