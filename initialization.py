import jax.numpy as jnp
import jax.random as jr
from scipy.stats.qmc import LatinHypercube

from test_functions import SyntheticFunction

def generate_initial_data_1D(num_points, benchmark, subkey, noise=0.0):
    """
    Generate initial data for training (1D)
    """
    assert(isinstance(benchmark, SyntheticFunction))
    bounds = benchmark.bounds
    x_train = jnp.linspace(bounds[0, 0], bounds[1, 0], num_points).reshape(-1, 1)
    #x_train = jr.uniform(key=key, shape=(n, 1)) * 12 - 2
    y_train = benchmark.f(x_train) + jr.normal(key=subkey, shape=x_train.shape) * noise
    return x_train, y_train

def generate_initial_data(num_points, benchmark):
    """
    Generate initial data for training
    """
    assert(isinstance(benchmark, SyntheticFunction))
    dim = benchmark.ndim
    bounds = benchmark.bounds
    sampler = LatinHypercube(d=dim)
    samples = jnp.array(sampler.random(n=num_points))  # NUM_POINTS x DIM, in [0, 1)
    X_train = (bounds[1, :] - bounds[0, :]) * samples + bounds[0, :]
    #X_train = torch.tensor(X_train.tolist())  # convert X_train to a tensor
    y_train = benchmark.f(X_train)

    # covert X_train and y_train to tensors
    #X_train = jnp.array(X_train.numpy())
    #y_train = jnp.array(y_train.numpy()).reshape(-1, 1)
    return X_train, y_train