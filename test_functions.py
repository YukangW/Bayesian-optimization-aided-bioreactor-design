import jax.numpy as jnp

def one_dimensional_simple(x):
    """
    One Dimensional Simple Target function
    """
    return jnp.exp(-(x-2)**2) + jnp.exp(-(x-6)**2 / 10) + 1 / (x**2+1)
