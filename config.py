import jax.numpy as jnp
import jaxkern
import gpjax as gpx

def cost_func(X):
    return jnp.array(4.) * X[-1] + jnp.array(1.)

class Config:
    def __init__(self, batch_size=1):
        self.kernel_func = jaxkern.RBF()
        self.mean_func = gpx.mean_functions.Zero()
        if batch_size == 1:
            self.model_options = {'seed': 42, 'verbose': 2, 'num_restarts': 8, 'n_iter': 5000}
            self.acqs_options = {'s': 1, 'p': 0.0, 'solver_options': {'disp': False, 'maxiter': 100000}, 'num_restarts': 8}
            self.optimization_options = {'HyperEst': 'frequentist', 'batch_size': batch_size}
            self.options = {'model_options': self.model_options, 'acqs_options': self.acqs_options, 'optimization_options': self.optimization_options}
        elif batch_size > 1:
            self.model_options = {'seed': 42, 'verbose': 2, 'num_restarts': 8, 'n_iter': 5000, 
                    'inv_mass_matrix': jnp.array([0.01, 0.01, 0.03, 0.1]), 'step_size': 1e-3, 'num_integration': 50, 'num_hmc_samples': 300000}
            self.acqs_options = {'s': 2, 'p': 0.5, 'solver_options': {'disp': False, 'maxiter': 100000}, 'num_restarts': 8}
            self.optimization_options = {'HyperEst': 'bayesian', 'batch_size': batch_size}
import jax.numpy as jnp
import jaxkern
import gpjax as gpx

def cost_func(X):
    return jnp.array(4.) * X[-1] + jnp.array(1.)

class Config:
    def __init__(self, batch_size=1):
        self.kernel_func = jaxkern.RBF()
        self.mean_func = gpx.mean_functions.Zero()
        if batch_size == 1:
            self.model_options = {'seed': 42, 'verbose': 2, 'num_restarts': 8, 'n_iter': 5000}
            self.acqs_options = {'s': 1, 'p': 0.0, 'solver_options': {'disp': False, 'maxiter': 100000}, 'num_restarts': 8}
            self.optimization_options = {'HyperEst': 'frequentist', 'batch_size': batch_size}
            self.options = {'model_options': self.model_options, 'acqs_options': self.acqs_options, 'optimization_options': self.optimization_options}
        elif batch_size > 1:
            self.model_options = {'seed': 42, 'verbose': 2, 'num_restarts': 8, 'n_iter': 5000, 
                    'inv_mass_matrix': jnp.array([0.01, 0.01, 0.03, 0.1]), 'step_size': 1e-3, 'num_integration': 50, 'num_hmc_samples': 300000}
            self.acqs_options = {'s': 2, 'p': 0.5, 'solver_options': {'disp': False, 'maxiter': 100000}, 'num_restarts': 8}
            self.optimization_options = {'HyperEst': 'bayesian', 'batch_size': batch_size}
            self.options = {'model_options': self.model_options, 'acqs_options': self.acqs_options, 'optimization_options': self.optimization_options}