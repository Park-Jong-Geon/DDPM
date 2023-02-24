import jax
import jax.numpy as jnp
from tqdm import tqdm
from utils import calculate_necessary_values
from functools import partial

@jax.jit
def backward_process(x_t, t, eps_theta, beta, z):
    assert x_t.shape == eps_theta.shape

    alpha_, sqrt_alpha_, sqrt_1_alpha_ = calculate_necessary_values(beta)
    beta_t = jnp.take(beta, t)
    alpha__t = jnp.take(alpha_, t)

    coef = beta_t / jnp.take(sqrt_1_alpha_, t) ** 0.5
    mean = jnp.expand_dims((1/(1-beta_t)**0.5), (1, 2, 3)) * (x_t - jnp.expand_dims(coef, (1, 2, 3)) * eps_theta)

    tilde_beta_t = beta_t * (1 - alpha__t/(1-beta_t)) / (1 - alpha__t)

    return mean + jnp.expand_dims(tilde_beta_t ** 0.5, (1, 2, 3)) * z

@jax.jit
def apply_trained_model(trained_state, x_t, t):
    return trained_state.apply_fn({'params': trained_state.params}, x_t, t)

@jax.jit
def img_rescale(image):
    return (jnp.clip(image, a_min=-1, a_max=1) + 1) / 2

'''
def execute_single_sample_at_all_steps(trained_state, beta, new_dim, key, resize, data_dim):
    eps = jax.random.normal(key, (1, *new_dim))
    backward_img = [eps]

    time_steps = jnp.size(beta, axis=0)
    for t in tqdm(reversed(range(0, time_steps))):
        x_t = backward_img[-1]
        vec_t = jnp.array([t,])
        eps_theta = apply_trained_model(trained_state, x_t, vec_t)
        dummy_key, key = jax.random.split(key)
        
        if t != 0:
            z = jax.random.normal(key=key, shape=x_t.shape)
        else:
            z = jnp.zeros(shape=x_t.shape)
        x_t_1 = backward_process(x_t, vec_t, eps_theta, beta, z)

        backward_img.append(x_t_1)
    
    backward_img = list(map(jnp.squeeze, backward_img))
    backward_img = list(map(img_rescale, backward_img))
    if resize:
        backward_img = list(map(partial(jax.image.resize, shape=data_dim, method='nearest'), backward_img))
    return backward_img
'''

def execute_sample(batch, trained_state, beta, new_dim, key, resize, data_dim):
    eps = jax.random.normal(key, (batch, *new_dim))
    x_t = eps
    time_steps = jnp.size(beta, axis=0)
    for t in (pbar := tqdm(reversed(range(0, time_steps)))):
        vec_t = jnp.ones(shape=(batch,), dtype=int) * t
        eps_theta = apply_trained_model(trained_state, x_t, vec_t)
        
        if t != 0:
            dummy_key, key = jax.random.split(key)
            z = jax.random.normal(key=key, shape=(batch, *new_dim))
        else:
            z = 0
        x_t_1 = backward_process(x_t, vec_t, eps_theta, beta, z)

        x_t = x_t_1

        pbar.set_description("Sampling")
    
    samples = jax.lax.map(img_rescale, x_t)
    if resize:
        samples = jax.lax.map(partial(jax.image.resize, shape=data_dim, method='nearest'), samples)
    return samples
