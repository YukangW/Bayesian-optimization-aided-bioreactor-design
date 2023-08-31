#!/usr/bin/python3

import jax.numpy as jnp


class BoundScaler:
    """
    Scale input data into N_DIM-dimensional unit hypercube
    """
    def __init__(self, bounds):
        try:
            self.lb = bounds[0, :].reshape(1, -1)  # row vector
            self.ub = bounds[1, :].reshape(1, -1)  # row vector
        except IndexError:
            print("BOUND must be an array with 2 rows")

    def transform(self, X):
        #print(f"X shape:{X.shape}")
        if X.ndim == 1:
            X = X.reshape(-1, 1)  # convert 1D array to 2D vector
        if X.shape[1]==self.lb.shape[1]:
            return (X - self.lb) / (self.ub - self.lb)
        elif X.shape[0] == self.lb.shape[1] and X.shape[1] == 1:
            X = X.T
            return (X - self.lb) / (self.ub - self.lb)
    
    def inverse_transform(self, X):
        assert(X.shape[1]==self.lb.shape[1]), "The length of input X cannot align with BOUND"
        return X * (self.ub - self.lb) + self.lb 
    

class ZNormScaler:
    """
    z-normalization
    """
    def __init__(self):
        self.mean = None
        self.std = None

    def fit_transform(self, y):
        self.mean = jnp.mean(y, axis=0)
        self.std = jnp.std(y, axis=0)
        return self.transform(y)
    
    def transform(self, y):
        if (self.mean is None or self.std is None):
            print("warning: ZNormScaler instance hasn't been fitted yet")
        else:
            return (y - self.mean) / self.std
    
    def inverse_transform(self, y):
        if (self.mean is None or self.std is None):
            print("warning: ZNormScaler instance hasn't been fitted yet")
        else:
#!/usr/bin/python3

import jax.numpy as jnp


class BoundScaler:
    """
    Scale input data into N_DIM-dimensional unit hypercube
    """
    def __init__(self, bounds):
        try:
            self.lb = bounds[0, :].reshape(1, -1)  # row vector
            self.ub = bounds[1, :].reshape(1, -1)  # row vector
        except IndexError:
            print("BOUND must be an array with 2 rows")

    def transform(self, X):
        #print(f"X shape:{X.shape}")
        if X.ndim == 1:
            X = X.reshape(-1, 1)  # convert 1D array to 2D vector
        if X.shape[1]==self.lb.shape[1]:
            return (X - self.lb) / (self.ub - self.lb)
        elif X.shape[0] == self.lb.shape[1] and X.shape[1] == 1:
            X = X.T
            return (X - self.lb) / (self.ub - self.lb)
    
    def inverse_transform(self, X):
        assert(X.shape[1]==self.lb.shape[1]), "The length of input X cannot align with BOUND"
        return X * (self.ub - self.lb) + self.lb 
    

class ZNormScaler:
    """
    z-normalization
    """
    def __init__(self):
        self.mean = None
        self.std = None

    def fit_transform(self, y):
        self.mean = jnp.mean(y, axis=0)
        self.std = jnp.std(y, axis=0)
        return self.transform(y)
    
    def transform(self, y):
        if (self.mean is None or self.std is None):
            print("warning: ZNormScaler instance hasn't been fitted yet")
        else:
            return (y - self.mean) / self.std
    
    def inverse_transform(self, y):
        if (self.mean is None or self.std is None):
            print("warning: ZNormScaler instance hasn't been fitted yet")
        else:
            return y * self.std + self.mean