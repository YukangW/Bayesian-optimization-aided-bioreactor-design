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
from test_functions import mf_simple_1d, func_wrapper, SyntheticFunction
from plotting_results import plot_GPs, plot_results_1D, plot_mf_results_2D, plot_mf_final_2D, plot_cost_evaluation
from initialization import generate_initial_data_1D, generate_initial_data
from config import Config, cost_func

from botorch.test_functions import AugmentedBranin
from scipy.optimize import minimize
from matplotlib import pyplot as plt
import json
import sobol_seq

import time
# Record the start time
start_time = time.process_time()
time_list = []

key = jr.PRNGKey(123)
key, subkey = jr.split(key)

## Target function
benchmark = SyntheticFunction(func=func_wrapper(AugmentedBranin()), bounds=jnp.array([[-5., 0., 0.], [10., 15., 1.]]))
#benchmark = SyntheticFunction(func=mf_simple_1d, bounds=jnp.array([[-2., 0.], [10., 1.]]))

optimal_value = -1 * jnp.array(AugmentedBranin._optimal_value)

## Generate initial dataset
N_INIT = 8
X_train, y_train = generate_initial_data(N_INIT, benchmark)
#X_train, y_train = generate_initial_data_1D(N_INIT, benchmark, subkey)

# save data in JSON file
#D = jnp.concatenate([X_train, y_train], axis=1)
#with open("../data/AugmentedBranin_8.json", 'w') as f:
#    json.dump(D.tolist(), f)
#print(D)

# load data from JSON file
#with open("../data/AugmentedBranin_8.json", 'r') as f:
#    D = jnp.array(json.load(f))
#X_train, y_train = D[:, :-1], D[:, -1:]

cost_list = [cost_func(X_train[i, :]).item() for i in range(X_train.shape[0])]
print(f"Initialization with {X_train.shape[0]} points")

## Gaussian process config
opt_config = Config(batch_size=1)

## Bayesian optimization iteration (Maximization)
budget = 100
budget = budget - sum(cost_list)
max_list = [y_train.max()]
max_so_far = y_train.max()
c_final = jnp.array(5.)  # 5.
i = 0
while budget > 0:
    iter_start_time = time.process_time()
    
    print(f"Starting {i+1}th iteration")
    print(f"remaining budget: {budget:.4f}")
    X_news= [0.0] * opt_config.optimization_options['batch_size']
    c_news = [0.0] * opt_config.optimization_options['batch_size']
    bay_opt = BayOptimizer(X_train, y_train, benchmark.bounds, opt_config.options)

    sampling_end_time = time.process_time()
    time_list.append(sampling_end_time - iter_start_time)

    for m in range(opt_config.optimization_options['batch_size']):
        X_new = bay_opt.one_iter()[0]
        c_new = cost_func(X_new)
        X_news[m] = X_new.tolist()
        c_news[m] = c_new.tolist()

    # stopping criteria
    c = sum(c_news)
    if (budget - c) > c_final:
        # Evaluate
        y_news = [benchmark.f(jnp.array(X)).tolist()[0] for X in X_news]
        cost_list.extend(c_news)
        print(f"cost: {c:.4f}")
        print(f"New points: {X_news}")
        print(f"New evaluations: {y_news}")
        #plot_GP(i, m, bay_opt, X_train, y_train, bounds)
        #plot_results_1D(X_train, y_train, X_news, y_news, benchmark, i)

        ## add new points to the dataset
        X_news = jnp.array(X_news)
        y_news = jnp.array(y_news)
        #plot_mf_results_2D(X_train, X_news, benchmark, N_INIT, i)
        X_train = jnp.concatenate([X_train, X_news], axis=0)
        y_train = jnp.concatenate([y_train, y_news], axis=0)
        
        # update maximum so far
        if y_news.max() >= max_so_far:
            max_so_far = y_news.max()
        max_list.append(max_so_far)
        print(f"max so far:{max_so_far}\n")
        i = i + 1
        budget = budget - c
    else:
        # final recommendation
        print(f"\nFinal recommendation")
        final_model = GPModel(X_train, -y_train, benchmark.bounds, opt_config.mean_func, opt_config.kernel_func, opt_config.model_options)
        final_model.learned_params = final_model.frequentist()
        pred_mean_func = lambda x: final_model.inference(jnp.concatenate([x, jnp.array([1.])], axis=0))[0]

        # optimize pred mean func
        n_dim = benchmark.bounds.shape[1] - 1
        lb = benchmark.bounds[0, :-1]  # lower bound
        ub = benchmark.bounds[1, :-1]  # upper bound

        # transform bounds to a tuple of lower bound upper bound pairs
        bounds_tuple = tuple(zip(lb, ub))

        # Initial guess: Sobol sequence
        num_restarts = 16
        options = {'disp': False, 'maxiter': 100000}
        multi_start_vec = jnp.array(sobol_seq.i4_sobol_generate(n_dim, num_restarts))  # NUM_RESTARTS x N_DIM, in N_DIM-dimensional unit hypercube
        localsol = [0.0] * num_restarts  # values for multistart
        localval = [0.0] * num_restarts  # variables for multistart

        # optimization loop
        for k in range(num_restarts):
            x_init = lb + (ub - lb) * multi_start_vec[k, :]

            # optimize acquisition function
            mean_grad = jax.grad(pred_mean_func)  # gradient
            # acqs_func_hessian = jax.hessian(self.acqs_avg, (0, ))  # Hessian; method SLSQP does not use Hessian information
            res = minimize(pred_mean_func, x_init, method='SLSQP', jac=mean_grad, options=options, bounds=bounds_tuple, tol=1e-8)
            localsol[k] = res.x
            if res.success:
                localval[k] = res.fun
            else:
                localval[k] = jnp.inf
        
        localval = jnp.array(localval)
        if jnp.min(localval) == jnp.inf:
            print("warning, no feasible solution found")
        min_index = jnp.argmin(localval)  # choosing the best solution of the optimization
        X_final = localsol[min_index].flatten()  # selecting the objective value of the best solution
        y_final = benchmark.f(jnp.concatenate([X_final, jnp.array([1.])], axis=0))
        budget = budget - c_final
        cost_list.append(c_final.item())
        break

    iter_end_time = time.process_time()
    time_list.append(iter_end_time - sampling_end_time)
print(f"X_final: {X_final}")
print(f"y_final: {y_final}\n")
print(f"Maximum: {max_so_far}")

# Record the end time
print(time_list)
end_time = time.process_time()
cpu_time = end_time - start_time
print(f"CPU Time: {cpu_time:.3f} seconds\n\n")

# save results in JSON file
X_final_with_fidelity = jnp.concatenate([X_final, jnp.array([1.])], axis=0).reshape(1, -1)
X = jnp.concatenate([X_train, X_final_with_fidelity], axis=0)
y = jnp.concatenate([y_train, y_final], axis=0)
cost_array = jnp.array(cost_list).reshape(-1, 1)
results = jnp.concatenate([X, y, cost_array], axis=1)
#result_final = jnp.concatenate([X_final, jnp.array([1.]), y_final.flatten()], axis=0).reshape(1, -1)
#plot_mf_final_2D(X_train, X_final, benchmark, N_INIT)
#plot_cost_evaluation(results, N_INIT)
#results = jnp.concatenate([results, result_final], axis=0)
#cost_array = jnp.array(cost_list).reshape(-1, 1)
#results = jnp.concatenate([results, cost_array], axis=1)
print(f"y_train: {results[:, -2]}")
print(f"fidelities: {results[:, -3]}")
with open("../data/MF-UCB/cj-UCB-4/AugmentedBranin/4/AugmentedBranin_8_results_0.json", 'w') as f:
    json.dump(results.tolist(), f)

with open("../results/MF-UCB/cj-UCB-4/AugmentedBranin/4/AugmentedBranin_8_results_0.txt", 'w') as g:
    g.write(f"{cpu_time}")

#with open("../data/q-MF-UCB/1D/0/simple_6_CPU_time.json", 'w') as h:
#    json.dump(time_list, h)