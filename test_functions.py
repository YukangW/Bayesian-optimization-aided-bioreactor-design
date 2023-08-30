import jax.numpy as jnp
import torch


def mf_simple_1d_helper(x, z):
    """
    One Dimensional Simple Target function
    bounds: [-2., 10.]
    inv_mass_matrix: [0.03, 0.3, 0.1]
    num_hmc_samples: 300000
    """
    return (1-z) * jnp.exp(-3 * (x - 2)**2) + (1 - 0.2*z) * jnp.exp(-(x - 6)**2 / 10) + (1-0.4*z) / (x**2+1) + z * jnp.exp(-(x - 2.5)**2 / (z + jnp.array(1e-3))) + z * 0.3 * jnp.cos(1.4 * jnp.pi * x) / (0.3 * x**2 + 1)

def mf_simple_1d(x):
    if x.ndim == 1:
        x = x.reshape(1, -1)
    z = x[:, -1]
    x = x[:, 0]
    return mf_simple_1d_helper(x, z).reshape(-1, 1)

def simple_1d(x):
    return (mf_simple_1d_helper(x, jnp.array(1.))).reshape(-1, 1)

def func_wrapper(func):
    def wrapper_helper(X):
        X = torch.tensor(X.tolist())  # covert X to a tensor
        y = func(X)
        y = jnp.array(y.numpy()).reshape(-1, 1)
        return -y
    return wrapper_helper

"""
Branin: [1e-4, 1e-4, 1e-4, 0.1]

AugmentedBranin: 
"""

class SyntheticFunction:
    """
    Synthetic function for test
    """
    def __init__(self, func, bounds):
        self.f = func
        self.bounds = bounds
        self.ndim = self.bounds.shape[1]

