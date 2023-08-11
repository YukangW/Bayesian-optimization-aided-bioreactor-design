#!/usr/bin/python3

# Enable Float64 for more stable matrix inversions.
from jax.config import config
config.update("jax_enable_x64", True)

import tensorflow_probability as tfp
import jax.numpy as jnp
import jax.random as jr
import jax
import gpjax as gpx
from functools import partial
import jaxkern

import sobol_seq
from scipy.optimize import minimize
from script.model import GPModel
import copy


class BayOptimizer:
    """
    Batch Bayesian optimizer
    """
    def __init__(self, init_X, init_y, bounds, options=None):
        """
        Initialize a batch Bayesian optimization
        ----------------------------------------------------
        Args:
            init_X: initial input data X
            init_y: initial input data y
            bounds: bounds for input
            options: Dict; optional arguments
        """
        self.X = init_X
        self.y = init_y
        self.bounds = bounds
        if options is not None:
            self.options = copy.deepcopy(options)  # deep copy, as Dict is mutable
        else:
            self.options = {}

        self.model_options = self.options.get('model_options', {})
        self.acqs_options = self.options.get('acqs_options', {})
        self.optimization_options = self.options.get('optimization_options', {})
        self.hyper_est = self.optimization_options.get('HyperEst', 'frequentist')

        self.s = self.optimization_options.get('s', 1)  # the number of surrogate models

        ## surrogate models
        self.mean_func = gpx.mean_functions.Zero()
        self.kernel_func = jaxkern.RBF()
        self.SMs = [GPModel(self.X, self.y, self.bounds, self.mean_func, self.kernel_func, self.model_options) for _ in range(self.s)]

        if self.hyper_est == 'bayesian':
            self.batch_size = self.optimization_options.get('batch_size', 1)
            # hyperparameter sampler
            print("Start Hamiltonian Monte Carlo sampling")
            self.theta_posterior_samples = self.SMs[0].bayesian(num_samples=10000)
            print("Sampling finished")


    def train(self):
        """
        Training Gaussian process surrogate model
        """
        if self.hyper_est == 'frequentist':
            assert(self.s==1), "Frequentist training only supports one surrogate model"
            for model in self.SMs:
                assert(isinstance(model, GPModel)), "`model` must be an instance of GPModel"
                model.learned_params = model.frequentist()
        elif self.hyper_est == 'bayesian':
            for model in self.SMs:
                theta = self.theta_posterior_samples[jax.random.randint(jr.PRNGKey(self.model_options.get('seed', 0)), (), -10000, -1).item()]
                model.learned_params = model._params_wrapper(model.original_params, theta)

    @staticmethod
    def get_acqs(model):
        """
        Get the acquisition function of surrogate model MODEL
        """
        assert(isinstance(model, GPModel)), "`model` must be an instance of GPModel"
        
        @partial(jax.jit, static_argnames=['j', 'negative'])
        def ucb(X, j, negative=True):
            """
            Acquisition function: upper confidence bound
            """
            pred_mean, pred_std = model.inference(X)
            sign = jnp.array(-1.) if negative else jnp.array(1.)
            return sign*(pred_mean + j * pred_std).reshape()
        return ucb

    def acqs_avg(self, X, j):
        """
        average of all acquisition functions
        """
        return jnp.array([self.get_acqs(model)(X, j) for model in self.SMs]).mean()
    
    @staticmethod
    def j_sampler(p=0.5):
        """
        Generator; Sample the jitter according to j | C = 1 ~ Beta(1, 12)
        """
        while True:
            if tfp.distributions.Bernoulli(probs=p).sample():
                yield jnp.array(tfp.distributions.Beta(1, 12).sample().numpy())
            else:
                yield jnp.array(1.)

    def optimize_acqs(self):
        """
        One iteration of Bayesian optimization with Gaussian process surrogate model: 
        Optimize the acquisition function to get new candidates and their 
        corresponding acquisition values
        """
        n_dim = self.bounds.shape[1]
        lb = self.bounds[0, :]  # lower bound
        ub = self.bounds[1, :]  # upper bound

        # transform bounds to a tuple of lower bound upper bound pairs
        bounds_tuple = tuple(zip(lb, ub))

        # Initial guess: Sobol sequence
        num_restarts = self.acqs_options.get('num_restarts', 4)
        options = self.acqs_options['solver_options']
        multi_start_vec = jnp.array(sobol_seq.i4_sobol_generate(n_dim, num_restarts))  # NUM_RESTARTS x N_DIM, in N_DIM-dimensional unit hypercube
        localsol = [0.0] * num_restarts  # values for multistart
        localval = [0.0] * num_restarts  # variables for multistart

        # sampling j from prior distribution
        p = self.acqs_options['p']
        self.j = next(self.j_sampler(p)).item()
        print(f"j: {self.j}")
        
        # optimization loop
        for i in range(num_restarts):
            x_init = lb + (ub - lb) * multi_start_vec[i, :]

            # optimize acquisition function
            acqs_func_grad = jax.grad(self.acqs_avg, (0, ))  # gradient
            # acqs_func_hessian = jax.hessian(self.acqs_avg, (0, ))  # Hessian; method SLSQP does not use Hessian information
            res = minimize(self.acqs_avg, x_init, args=(self.j, ), method='SLSQP', jac=acqs_func_grad, options=options, bounds=bounds_tuple, tol=1e-12)
            localsol[i] = res.x
            if res.success:
                localval[i] = res.fun
            else:
                localval[i] = jnp.inf
        
        localval = jnp.array(localval)
        if jnp.min(localval) == jnp.inf:
            print("warning, no feasible solution found")
        min_index = jnp.argmin(localval)  # choosing the best solution of the optimization
        x_new = localsol[min_index].reshape(1, -1)  # selecting the objective value of the best solution
        acqs_new = self.acqs_avg(x_new, self.j)
        return x_new, acqs_new

    def one_iter(self):
        """
        One iteration of batch Bayesian optimization
        """
        # training surrogate models
        self.train()

        # optimize acquisition function
        x_new, _ = self.optimize_acqs()
        return x_new