import jax
import jax.numpy as jnp
from tqdm import tqdm
from utils import calculate_necessary_values
from functools import partial

@jax.jit
def backward_process(x_t, t, eps_theta, beta, key):
    assert x_t.shape == eps_theta.shape

    alpha_, sqrt_alpha_, sqrt_1_alpha_ = calculate_necessary_values(beta)
    beta_t = jnp.take(beta, t)
    coef = beta_t / jnp.take(sqrt_1_alpha_, t) ** 0.5

    mean = 1 / (1 - beta_t) ** 0.5 * (x_t - coef * eps_theta)
    z = jax.random.normal(key=key, shape=(x_t.shape))

    return mean + beta_t ** 0.5 * z

@jax.jit
def apply_trained_model(trained_state, x_t, t):
    return trained_state.apply_fn({'params': trained_state.params}, x_t, jnp.array([t,]))

def execute_sample(trained_state, beta, new_dim, key, resize, data_dim):
    eps = jax.random.normal(key, (1, new_dim[0], new_dim[1], new_dim[2]))
    backward_img = [eps]

    time_steps = jnp.size(beta, axis=0)
    for t in tqdm(reversed(range(0, time_steps))):
        x_t = backward_img[-1]
        eps_theta = apply_trained_model(trained_state, x_t, t)
        dummy_key, key = jax.random.split(key)
        
        x_t_1 = backward_process(x_t, t, eps_theta, beta, key)
        backward_img.append(x_t_1)
    
    backward_img = list(map(jnp.squeeze, backward_img))
    backward_img = list(map(partial(jnp.clip, a_min=0, a_max=1), backward_img))
    if resize:
        backward_img = list(map(partial(jax.image.resize, shape=(data_dim[0], data_dim[1]), method='nearest'), backward_img))
    return backward_img
