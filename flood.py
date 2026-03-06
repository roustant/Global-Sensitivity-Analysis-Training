# Flood model

import numpy as np
from scipy.stats import uniform, norm, truncnorm, gumbel_r, triang

def model(X, ans = 2):
    # ans = 1 gives Overflow output; ans = 2 gives Cost output; ans=0 gives both
    
    nsize=X.shape[0]
    
    if (ans == 0):
        output = np.zeros((nsize,2))
    else:
        output = np.zeros(nsize)
        
    for i in range(nsize):
        H = (X[i, 0] / (X[i, 1] * X[i, 7] * np.sqrt((X[i, 3] - X[i, 2]) / X[i, 6])))**0.6
        S = X[i, 2] + H - X[i, 4] - X[i, 5] 
        
        if (S > 0):
            Cp = 1
        else:
            Cp = 0.2 + 0.8 * (1 - np.exp(- 1000 / S**4))
    
        if (X[i, 4] > 8):
            Cp = Cp + X[i, 4] / 20 
        else:
            Cp = Cp + 8 / 20
        if (ans == 0):
            output[i, 0] = S ;
            output[i, 1] = Cp ;
            
        if (ans == 1): 
            output[i] = S
        if (ans == 2):
            output[i] = Cp 
    return(output)

# Function for flood model inputs sampling

def truncated_gumbel_sample(size, loc, scale, min_val, max_val): # Generates samples from a truncated Gumbel distribution
        
    cdf_min = np.exp(-np.exp(-(min_val - loc) / scale))
    cdf_max = np.exp(-np.exp(-(max_val - loc) / scale))
    
    u = np.random.uniform(cdf_min, cdf_max, size)
    
    samples = loc - scale * np.log(-np.log(u))
    
    return samples

def getSample(size):
    
    X = np.zeros((size, 8))
    
    X[:, 0] = truncated_gumbel_sample(size, loc=1013, scale=558, min_val=500, max_val=3000)
    
    a, b = (15 - 30) / 8, (np.inf - 30) / 8
    X[:, 1] = truncnorm.rvs(a, b, loc=30, scale=8, size=size)
    
    # Using your explicit center-top calculation format
    X[:, 2] = triang.rvs(c=(50-49)/(51-49), loc=49, scale=51-49, size=size)
    X[:, 3] = triang.rvs(c=(55-54)/(56-54), loc=54, scale=56-54, size=size)
    X[:, 4] = np.random.uniform(7, 9, size=size)
    X[:, 5] = triang.rvs(c=(55.5-55)/(56-55), loc=55, scale=56-55, size=size)
    X[:, 6] = triang.rvs(c=(5000-4990)/(5010-4990), loc=4990, scale=5010-4990, size=size)
    X[:, 7] = triang.rvs(c=(300-295)/(305-295), loc=295, scale=305-295, size=size)

    return X