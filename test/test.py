#!/usr/bin/python3

import sys
sys.path.append('..')

# Enable Float64 for more stable matrix inversions.
from jax.config import config

config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
import jaxkern
import jax
import gpjax as gpx
from script.model import GPModel
from script.GP_optimizer import BayOptimizer
import matplotlib
from matplotlib import pyplot as plt

key = jr.PRNGKey(123)
key, subkey = jr.split(key)
cols = matplotlib.rcParams["axes.prop_cycle"].by_key()["color"]

## Dataset
def f(x):
    """
    Target function
    """
    return jnp.exp(-(x-2)**2) + jnp.exp(-(x-6)**2 / 10) + 1 / (x**2+1)

def generate_initial_data(n=16, noise=0.0):
    """
    Generate initial data for training
    """
    x_train = jnp.linspace(-2., 10., n).reshape(-1, 1)
    #x_train = jr.uniform(key=key, shape=(n, 1)) * 12 - 2
    y_train = f(x_train) + jr.normal(key=subkey, shape=x_train.shape) * noise
    return x_train, y_train

def plot_result(model, x_train, y_train, x_new, y_new, bounds):
    """
    visualization
    """
    assert(isinstance(model, BayOptimizer)), "`model` must be an instance of BayOptimizer"
    # data for visualization
    x_test = jnp.linspace(*bounds, 1000)
    y_test = f(x_test)

    ## Prediction (just for visualization purpose)
    pred_mean, pred_std = model.SMs[0].inference(x_test)
    #acqs_test = pred_mean + model.j * pred_std
    acqs_test = jnp.array([model.acqs_avg(x, model.j) for x in x_test])

    # visualization
    plt.figure(dpi=300)
    fig, ax = plt.subplots()
    ax.scatter(x_train, y_train, marker='o', label="Observations", color=cols[0], alpha=0.5)
    ax.plot(x_test, y_test, label="Latent function", color=cols[0], linewidth=2)
    ax.plot(x_test, pred_mean, label="Predictive mean", color=cols[1])
    ax.fill_between(x_test.squeeze(), pred_mean-2*pred_std, pred_mean+2*pred_std, label="$2\sigma$", alpha=0.5, color=cols[1])
    ax.scatter(x_new, y_new, marker='X', label="New point", color='r')
    ax.plot(x_test, -acqs_test, label="Acquisition function", color=cols[2])
    ax.legend()
    #plt.show()
    fig.savefig("../results/Bayesian/Figure_7.png")

# generate initial dataset
x_train, y_train = generate_initial_data(n=6)
print(f"Initialization with {x_train.shape[0]} points")

## Gaussian process config
bounds = jnp.array([[-2.], [10.]])
kernel_func = jaxkern.RBF()
mean_func = gpx.mean_functions.Zero()
model_options = {'seed': 42, 'verbose': 2, 'num_restarts': 8, 'n_iter': 5000, 
                 'inv_mass_matrix': jnp.array([1e-2, 0.3, 1e-2]), 'step_size': 1e-3, 'num_integration': 100, 'num_hmc_samples': 300000}
acqs_options = {'s': 1, 'p': 0.5, 'solver_options': {'disp': False, 'maxiter': 100000}}
optimization_options = {'HyperEst': 'bayesian', 'batch_size': 1}
options = {'model_options': model_options, 'acqs_options': acqs_options, 'optimization_options': optimization_options}

## Bayesian optimization iteration
N_ITER = 1
for i in range(N_ITER):
    print(f"Starting {i+1}th iteration")
    x_news= [0.0] * optimization_options['batch_size']
    y_news= [0.0] * optimization_options['batch_size']
    bay_opt = BayOptimizer(x_train, y_train, bounds, options)
    for m in range(optimization_options['batch_size']):
        x_new = bay_opt.one_iter()
        y_new = f(x_new)
        print(f"New point {m+1}: {x_new.item():.4f}\nNew evaluation {m+1}: {y_new.item():.3f}")
        x_news[m] = x_new.item()
        y_news[m] = y_new.item()

    plot_result(bay_opt, x_train, y_train, x_news, y_news, bounds)
    ## add new point to the dataset
    x_train = jnp.concatenate([x_train, jnp.array(x_news).reshape(-1, 1)], axis=0)
    y_train = jnp.concatenate([y_train, jnp.array(y_news).reshape(-1, 1)], axis=0)