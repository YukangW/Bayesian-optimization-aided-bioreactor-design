#!/usr/bin/python3

# Enable Float64 for more stable matrix inversions.
from jax.config import config

config.update("jax_enable_x64", True)


import jax
import jax.numpy as jnp
import jax.random as jr
import optax as ox
import jaxutils
import gpjax as gpx
import blackjax
import tensorflow_probability.substrates.jax as tfp
import tensorflow as tf

import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats.qmc import LatinHypercube
from scipy.optimize import minimize
import copy

from utils import BoundScaler, ZNormScaler


class GPModel:
    """
    Gaussian process regression
    """

    def __init__(self, X, y, bounds, mean_func, kernel_func, options=None):
        """
        Initialize a Gaussian process regression model
        -----------------------------------------
        Args:
            X: Input data
            y: output data
            bounds: bounds for input
            mean_func: prior mean function
            kernel_func: kernel function
            options: Dict, optional arguments
        """
        ## Instance variables
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        self.bounds = bounds
        self.mean_func = mean_func
        self.kernel = kernel_func
        if options is not None:
            self.options = copy.deepcopy(options)  # deep copy, as Dict is mutable
        else:
            self.options = {}
        self._seed = self.options.get('seed', 123)
        #key = jr.PRNGKey(self._seed)
        #self._key, self._subkey = jr.split(key)
        self._verbose = self.options.get('verbose', 0)

        ## Dataset
        # scale X into unit hypercube
        self._scaler_X = BoundScaler(self.bounds)
        X_norm = self._scaler_X.transform(X)

        # z-normalization
        self._scaler_y = ZNormScaler()
        y_norm = self._scaler_y.fit_transform(y)

        # jax dataset
        self.D = jaxutils.Dataset(X=X_norm, y=y_norm)

        ## Defining the prior
        self.prior = gpx.Prior(mean_function=mean_func, kernel=kernel_func)

        ## Constructing the posterior
        self.likelihood = gpx.Gaussian(num_datapoints=self.D.n)
        self.posterior = self.prior * self.likelihood

        ## initialize hyperparameters
        self.learned_params = None

    @staticmethod
    def _params_wrapper(original_params, theta):
        """
        Parameters wrapper: return a new Dictionary object with new hyperparams `theta`
        """
        assert(len(theta)>=3), "hyperparameters `theta` must be a numpy.ndarray with at least 3 elements"
        theta = jnp.array(theta)
        new_params = copy.deepcopy(original_params)
        new_params['kernel']['lengthscale'] = theta[0:-2]  # 1D array
        new_params['kernel']['variance'] = theta[-2:-1]  # 1D array
        new_params['likelihood']['obs_noise'] = theta[-1:]  # 1D array
        return new_params

    def _initialise_params(self, num_restarts):
        """
        Generator; initialise parameters state of posterior
        """
        # Latin hypercube sampling
        sampler = LatinHypercube(d=self.n_features+2)  # lengthscale, variance and observation noise
        samples = jnp.array(sampler.random(n=num_restarts))  # NUM_RESTARTS x d, in [0, 1)
        
        i = 0
        while i < num_restarts:
            # edit original parameter state as new values
            lengthscale = 10** (3*samples[i, :-2]-2)  # 1D array, [1e-2, 1e1)
            variance = 10** (2*samples[i, -2]-1)   # 1D array, [1e-1, 1e1)
            obs_noise = 10** (5*samples[i, -1]-8)  # 1D array, [1e-8, 1e-3)
            yield jnp.array([*lengthscale, variance, obs_noise])
            i += 1

    def frequentist(self):
        """
        Frequentist: Maximum Likelihood Estimation
        ------------------------------------------
        Args:
            n_iter: number of iteration for optimization
        """
        def mll_helper(theta):
            """
            Put hyperparameters `theta` into wrapper
            """
            new_params = self._params_wrapper(self.original_params, theta)  
            return mll(new_params)

        # marginal log likelihood
        mll = jax.jit(self.posterior.marginal_log_likelihood(self.D, negative=True))

        # initialize parameter state
        parameter_state = gpx.initialise(self.posterior, key=jr.PRNGKey(self._seed))
        self.original_params, trainables, bijectors = parameter_state.unpack()

        # initialization
        # Initial hyperparameters affect learning process as the negative marginal log likelihood is not convex.
        num_restarts = self.options.get('num_restarts', 8)
        best_learned_params = None
        localsol = [0.0] * num_restarts  # values for multistart
        localval = [0.0] * num_restarts  # variables for multistart
        params_initializer = self._initialise_params(num_restarts=num_restarts)  # tuple generator

        # bounds for hyperparameters theta
        bounds_list = [(1e-8, None) for _ in range(self.n_features)]  # bounds for `lengthscale`
        bounds_list.extend([(1e-8, None), (0., 1e-3)])  # bounds for `variance` and `obs_noise`
        bounds_tuple = tuple(bounds_list)
        
        mll_grad = jax.grad(mll_helper, argnums=(0, ))  # gradient

        n_iter = self.options.get('n_iter', 5000)
        for i, theta_init in enumerate(params_initializer):
            res = minimize(mll_helper, theta_init, method='SLSQP', jac=mll_grad, options={'disp': False, 'maxiter': n_iter}, bounds=bounds_tuple, tol=1e-12)
            localsol[i] = res.x
            if res.success:
                localval[i] = res.fun
            else:
                localval[i] = jnp.inf
        localval = jnp.array(localval)

        if jnp.min(localval) == jnp.inf:
            print("warning, no feasible solution found")
        min_index = jnp.argmin(localval)  # choosing the best solution of the optimization
        theta_best = localsol[min_index]  # selecting the hyperparameters
        best_learned_params = self._params_wrapper(self.original_params, theta_best)

        # visualization
        if self._verbose == 1:
            print(f"lengthscale: {best_learned_params['kernel']['lengthscale'].tolist()}")
            print(f"variance: {best_learned_params['kernel']['variance'].item():.4f}")
            print(f"obs_noise: {best_learned_params['likelihood']['obs_noise'].item():.3e}")
        
        return best_learned_params
    
    def bayesian(self, num_samples):
        """
        Bayesian: sampling posterior distribution
        """
        def log_density_func(log_theta):
            """
            log probability density
            """
            #theta = jnp.concatenate([jnp.exp(log_theta[:-1]), log_theta[-1:]], axis=0)
            theta = jnp.exp(log_theta)
            lengthscale = theta[:-2]
            variance = theta[-2]
            obs_noise = theta[-1]

            # theta priors
            # p(theta) = Gamma(alpha=1, beta=6); priors are independent
            lengthscale_prior = tfp.distributions.Gamma(concentration=1, rate=6)
            variance_prior = tfp.distributions.Gamma(concentration=1, rate=6)
            obs_noise_prior = tfp.distributions.Gamma(concentration=1, rate=6)
            shift = tfp.bijectors.Shift(epsilon)  # shift epsilon unit
            shifted_lengthscale_prior = tfp.distributions.TransformedDistribution(distribution=lengthscale_prior, bijector=shift, name="ShiftedLengthscale")
            shifted_variance_prior = tfp.distributions.TransformedDistribution(distribution=variance_prior, bijector=shift, name="ShiftedVariance")
            shifted_obs_noise_prior = tfp.distributions.TransformedDistribution(distribution=obs_noise_prior, bijector=shift, name="ShiftedObsNoise")
            
            # Gamma distribution is defined on [0, inf). However, if the `log_prob` method takes an negative argument, it will return a positive
            # value that is bigger than the log probability density at zero
            log_prob_lengthscale = jnp.sum(jnp.where(lengthscale>=epsilon, shifted_lengthscale_prior.log_prob(lengthscale), jnp.array(jnp.log(epsilon))))
            log_prob_variance = jnp.where(variance>=epsilon, shifted_variance_prior.log_prob(variance), jnp.array(jnp.log(epsilon)))   
            log_prob_obs_noise = jnp.where(obs_noise>=epsilon, shifted_obs_noise_prior.log_prob(obs_noise), jnp.array(jnp.log(epsilon)))
            #log_prior = jnp.sum(shifted_lengthscale_prior.log_prob(lengthscale)) + shifted_variance_prior.log_prob(variance) + shifted_obs_noise_prior.log_prob(obs_noise)
            log_prior = log_prob_lengthscale + log_prob_variance + log_prob_obs_noise

            # marginal log likelihood
            mll = jax.jit(self.posterior.marginal_log_likelihood(self.D, negative=False))
            new_params = self._params_wrapper(self.original_params, theta)  # prepare parameters for `mll` function

            # log probability density of the posterior of theta
            log_posterior = log_prior + mll(new_params)

            return log_posterior

        # initialize parameter state
        parameter_state = gpx.initialise(self.posterior, key=jr.PRNGKey(self._seed))
        self.original_params, trainables, bijectors = parameter_state.unpack()

        # log probability gradient
        #log_posterior_grad = jax.grad(log_density_func, argnums=(0, ))

        # Hamiltonian Monte Carlo
        #inv_mass_matrix = jnp.ones(self.n_features+2)  # `n_features` lengthscale, one variance, one obs_noise
        inv_mass_matrix = self.options.get('inv_mass_matrix', jnp.ones(self.n_features+2))
        assert(inv_mass_matrix.shape[0] == self.n_features+2), "the shape of inv_mass_matrix must align with the number of features"
        num_integration_steps = self.options.get('num_integration', 100)
        step_size = self.options.get('step_size', 1e-3)
        num_hmc_samples = self.options.get('num_hmc_samples', 100000)
        #num_nuts_samples = self.options.get('num_hmc_samples', 100000)
        hmc = blackjax.hmc(log_density_func, step_size, inv_mass_matrix, num_integration_steps)
        #nuts = blackjax.nuts(log_density_func, step_size, inv_mass_matrix)

        # sample a theta from prior distribution
        epsilon = jnp.array(1e-14)  # to avoid numerical computation error
        initial_lengthscale = tfp.distributions.Gamma(jnp.array(1.), jnp.array(6.)).sample(sample_shape=(self.n_features), seed=jr.PRNGKey(self._seed)) + epsilon
        initial_variance = tfp.distributions.Gamma(jnp.array(1.), jnp.array(6.)).sample(sample_shape=(1, ), seed=jr.PRNGKey(self._seed)) + epsilon
        initial_obs_noise = tfp.distributions.Gamma(jnp.array(1.), jnp.array(6.)).sample(sample_shape=(1, ), seed=jr.PRNGKey(self._seed)) + epsilon
        initial_position = jnp.concatenate([initial_lengthscale, initial_variance, initial_obs_noise], axis=0)

        # build the kernel and inference loop
        initial_state = hmc.init(initial_position)
        hmc_kernel = jax.jit(hmc.step)
        #initial_state = nuts.init(initial_position)
        #nuts_kernel = jax.jit(nuts.step)

        def inference_loop(key, kernel, initial_state, num_hmc_samples):
            @jax.jit
            def one_step(state, key):
                state, _  = kernel(key, state)
                return state, state
            
            keys = jr.split(key, num_hmc_samples)
            _, states = jax.lax.scan(one_step, initial_state, keys)
            return states

        # inference
        rng_key = jr.PRNGKey(self._seed)

        states = inference_loop(rng_key, hmc_kernel, initial_state, num_hmc_samples)
        #states = inference_loop(rng_key, nuts_kernel, initial_state, num_nuts_samples)
        
        # multiple chains
        """
        def inference_loop_multiple_chains(key, kernel, initial_state, num_nuts_samples, num_chains):
            @jax.jit
            def one_step(states, key):
                keys = jr.split(key, num_chains)
                states, _  = jax.vmap(kernel)(keys, states)
                return states, states
            
            keys = jr.split(key, num_nuts_samples)
            _, states = jax.lax.scan(one_step, initial_state, keys)
            return states
        
        num_chains = multiprocessing.cpu_count()
        initial_positions = jnp.repeat(initial_position.reshape(1, -1), num_chains, axis=0)
        initial_states = jax.vmap(nuts.init, in_axes=(0))(initial_positions)

        states = inference_loop_multiple_chains(rng_key, nuts.step, initial_states, num_nuts_samples, num_chains)
        """

        theta_samples = jnp.exp(states.position.block_until_ready())

        # visualization
        if self._verbose == 2:
            cols = matplotlib.rcParams["axes.prop_cycle"].by_key()["color"]
            for k in range(self.n_features):
                lengthscale_samples = theta_samples[:, k]
                plt.figure(dpi=200)
                plt.plot(lengthscale_samples, color=cols[0])
                plt.xlabel("Samples")
                plt.ylabel("lengthscale")
                plt.yscale('log')
                plt.savefig(f"../results/HMC/HMC_GP_lengthscale_{k}.png")
                plt.close()

                
                plt.figure(dpi=200)
                sns.displot(lengthscale_samples[-num_samples:], color=cols[0])
                plt.xlabel("lengthscale")
                plt.ylabel("probability density")
                plt.savefig(f"../results/HMC/HMC_GP_lengthscale_{k}_dist.png")
                plt.close()
                
            variance_samples = theta_samples[:, -2]
            plt.figure(dpi=200)
            plt.plot(variance_samples, color=cols[1])
            plt.xlabel("Samples")
            plt.ylabel("variance")
            plt.yscale('log')
            plt.savefig("../results/HMC/HMC_GP_variance.png")
            plt.close()
            
            plt.figure(dpi=200)
            sns.displot(variance_samples[-num_samples:], color=cols[1])
            plt.xlabel("variance")
            plt.ylabel("probability density")
            plt.savefig("../results/HMC/HMC_GP_variance_dist.png")
            plt.close()

            obs_noise_samples = theta_samples[:, -1]
            plt.figure(dpi=200)
            plt.plot(obs_noise_samples, color=cols[2])
            plt.xlabel("Samples")
            plt.ylabel("obs_noise")
            plt.yscale('log')
            plt.savefig("../results/HMC/HMC_GP_obs_noise.png")
            plt.close()
            
            plt.figure(dpi=200)
            sns.displot(obs_noise_samples[-num_samples:], color=cols[2])
            plt.xlabel("obs_noise")
            plt.ylabel("probability density")
            plt.savefig("../results/HMC/HMC_GP_obs_noise_dist.png")
            plt.close()
            
        return theta_samples[-num_samples:, :]

    def inference(self, x):
        """
        distribution conditioning x
        """
        # scale x
        x = self._scaler_X.transform(x)
        
        # conditioning
        assert(self.learned_params is not None), "Gaussian Process model hasn't been trained yet."
        latent_distribution = self.posterior(self.learned_params, self.D)(x)
        pred_distribution = self.likelihood(self.learned_params, latent_distribution)
        pred_norm_mean = pred_distribution.mean().reshape(-1, 1)
        pred_mean = self._scaler_y.inverse_transform(pred_norm_mean).squeeze()  # inverse z-normalization
        pred_norm_std = pred_distribution.stddev()
        pred_std = pred_norm_std * self._scaler_y.std
        return pred_mean, pred_std