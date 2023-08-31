#!/usr/bin/python3
import sys
sys.path.append('..')

# Enable Float64 for more stable matrix inversions.
from jax.config import config

config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
import jax

from script.model import GPModel
from script.GP_optimizer import BayOptimizer
from test_functions import simple_1d, func_wrapper, SyntheticFunction
from plotting_results import plot_GPs, plot_results_1D, plot_results_2D, plot_acqs_funcs
from initialization import generate_initial_data_1D, generate_initial_data
from config import Config

from botorch.test_functions import Branin
from scipy.optimize import minimize
import json

import time
# Record the start time
start_time = time.process_time()
time_list = []

key = jr.PRNGKey(123)
key, subkey = jr.split(key)

## Target function
#benchmark = SyntheticFunction(func=func_wrapper(Branin()), bounds=jnp.array([[-5., 0.], [10., 15.]]))
benchmark = SyntheticFunction(func=simple_1d, bounds=jnp.array([[-2.], [10.]]))
optimal_value = -1 * jnp.array(Branin._optimal_value)

## Generate initial dataset
N_INIT = 4
#X_train, y_train = generate_initial_data(N_INIT, benchmark)
X_train, y_train = generate_initial_data_1D(N_INIT, benchmark, subkey)

# save data in JSON file
#D = jnp.concatenate([X_train, y_train], axis=1)
#with open("../data/AugmentedBranin_8.json", 'w') as f:
#    json.dump(D.tolist(), f)
#print(D)

# load data from JSON file
#with open("../data/AugmentedBranin_8.json", 'r') as f:
#    D = jnp.array(json.load(f))
#X_train, y_train = D[:, :-1], D[:, -1:]
print(f"Initialization with {X_train.shape[0]} points")

## Gaussian process config
BATCH_SIZE = 2
opt_config = Config(batch_size=BATCH_SIZE)

## Bayesian optimization iteration (Maximization)
N_ITER = 10  # BUDGET = 2 * 10 = 100
max_list = [y_train.max()]
max_so_far = y_train.max()

for i in range(N_ITER):
    iter_start_time = time.process_time()

    print(f"Starting {i+1}th iteration")
    X_news= [0.0] * opt_config.optimization_options['batch_size']
    y_news = [0.0] * opt_config.optimization_options['batch_size']
    bay_opt = BayOptimizer(X_train, y_train, benchmark.bounds, opt_config.options)

    sampling_end_time = time.process_time()
    time_list.append(sampling_end_time - iter_start_time)
    acqs_tests = []
    for m in range(opt_config.optimization_options['batch_size']):
        X_new = bay_opt.one_iter()[0]
        y_new = benchmark.f(X_new)[0]
        X_news[m] = X_new.tolist()
        y_news[m] = y_new.tolist()
        # plot GPs and return acquistion function
        acqs_test = plot_GPs(bay_opt, X_train, y_train, benchmark, iteration=i, batch=m, batch_size=BATCH_SIZE)
        acqs_tests.append(acqs_test)
    print(f"New points: {X_news}")
    print(f"New evaluations: {y_news}")

    ## add new points to the dataset
    X_news = jnp.array(X_news)
    y_news = jnp.array(y_news)
    
    # plot acquisition functions
    plot_acqs_funcs(bay_opt, acqs_tests, X_train, y_train, X_news, y_news, benchmark, iteration=i, batch_size=BATCH_SIZE)
    #plot_results_2D(X_train, X_news, benchmark, N_INIT, i)
    X_train = jnp.concatenate([X_train, X_news], axis=0)
    y_train = jnp.concatenate([y_train, y_news], axis=0)

    # update maximum so far
    if y_news.max() >= max_so_far:
        max_so_far = y_news.max()
    max_list.append(max_so_far)
    print(f"max so far:{max_so_far}\n")

    iter_end_time = time.process_time()
    time_list.append(iter_end_time - sampling_end_time)
final_index = jnp.unravel_index(jnp.argmax(y_train), y_train.shape)  # tuple

print(f"X_final: {X_train[final_index]}")
print(f"y_final: {y_train[final_index]}")

# Record the end time
print(time_list)

end_time = time.process_time()
cpu_time = end_time - start_time
print(f"CPU Time: {cpu_time:.3f} seconds")

# save results in JSON file
results = jnp.concatenate([X_train, y_train], axis=1)
print(f"y_train: {results[:, -1]}")
with open("../data/j-ATS-UCB/1D/0/simple_4_results_0.json", 'w') as f:
    json.dump(results.tolist(), f)

with open("../results/j-ATS-UCB/1D/0/simple_4_results_0.txt", 'w') as g:
    g.write(f"{cpu_time}")

with open("../data/j-ATS-UCB/1D/0/simple_4_CPU_time.json", 'w') as h:
    json.dump(time_list, h)