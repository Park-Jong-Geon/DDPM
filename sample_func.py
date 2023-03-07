import jax
import jax.numpy as jnp
import flax
from tqdm import tqdm
from utils import calculate_necessary_values
from functools import partial

@jax.jit
def backward_process(x_t, t, eps_theta, beta, eps):
    assert x_t.shape == eps_theta.shape

    alpha_, sqrt_alpha_, sqrt_1_alpha_ = calculate_necessary_values(beta)
    t = jnp.int16(t)
    beta_t = beta[t]
    # alpha__t = jnp.take(alpha_, t)

    coef = beta_t / jnp.take(sqrt_1_alpha_, t)
    mean = jnp.reshape((1 / jnp.sqrt(1.-beta_t)), (-1, 1, 1, 1)) * (x_t - jnp.reshape(coef, (-1, 1, 1, 1)) * eps_theta)

    # tilde_beta_t = beta_t * (1 - alpha__t/(1-beta_t)) / (1 - alpha__t)

    return mean + jnp.reshape(jnp.sqrt(beta_t), (-1, 1, 1, 1)) * eps

@jax.jit
def apply_trained_model(trained_state, x_t, t):
    return trained_state.apply_fn({'params': trained_state.params}, x_t, t)
'''
@jax.jit
def img_rescale(imgs):
    assert len(imgs.shape) == 4
    min = jnp.expand_dims(jnp.min(imgs, axis=(1, 2)), (1, 2))
    imgs = imgs - min
    max = jnp.expand_dims(jnp.max(imgs, axis=(1, 2)), (1, 2))
    imgs = jnp.uint8(imgs / max * 255)
    return imgs
'''

@jax.jit
def img_rescale(imgs):
    imgs = jnp.clip(imgs, a_min=-1, a_max=1)
    imgs = (imgs + 1) / 2 * 255
    imgs = jnp.uint8(imgs)
    return imgs

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

        backward_img.append(jnp.clip(x_t_1, a_min=-1, a_max=1))
    
    backward_img = list(map(jnp.squeeze, backward_img))
    backward_img = list(map(img_rescale, backward_img))
    if resize:
        backward_img = list(map(partial(jax.image.resize, shape=data_dim, method='nearest'), backward_img))
    return backward_img
'''
'''
def execute_sample(sample_num, trained_state, beta, new_dim, key, resize, data_dim):
    x_t = jax.random.normal(key, (sample_num, *new_dim))
    time_steps = jnp.size(beta, axis=0)
    for t in (pbar := tqdm(reversed(range(0, time_steps)))):
        vec_t = jnp.repeat(t, sample_num)
        eps_theta = apply_trained_model(trained_state, x_t, vec_t)
        
        if t > 0:
            another_key, key = jax.random.split(key)
            eps = jax.random.normal(key=key, shape=(sample_num, *new_dim))
        else:
            eps = 0
        
        x_t_1 = backward_process(x_t, vec_t, eps_theta, beta, eps)

        x_t = x_t_1
        # print(f"{t}\n{x_t}")

        pbar.set_description("Sampling")
    
    samples = img_rescale(x_t)
    if resize:
        samples = jax.image.resize(samples, shape=(sample_num, *data_dim), method='nearest')
    return samples
'''

def execute_sample(sample_num, trained_state, beta, new_dim, key, resize, data_dim):
    devices = jax.local_device_count()
    assert sample_num % devices == 0
    divided_sample_num = sample_num // devices
    
    trained_state = flax.jax_utils.replicate(trained_state)
    
    x_t = jax.random.normal(key, (devices, divided_sample_num, *new_dim))
    time_steps = jnp.size(beta, axis=0)
    for t in (pbar := tqdm(reversed(range(0, time_steps)))):
        vec_t = t * jnp.ones(shape=(devices, divided_sample_num))
        eps_theta = jax.pmap(apply_trained_model)(trained_state, x_t, vec_t)
        
        if t > 0:
            another_key, key = jax.random.split(key)
            eps = jax.random.normal(key=key, shape=(devices, divided_sample_num, *new_dim))
        else:
            eps = jnp.zeros(shape=(devices, divided_sample_num, *new_dim))
        
        x_t_1 = jax.pmap(lambda x_t, vec_t, eps_theta, eps: backward_process(x_t, vec_t, eps_theta, beta, eps))(x_t, vec_t, eps_theta, eps)
        
        x_t = x_t_1

        pbar.set_description("Sampling")
    
    x_t = jnp.reshape(x_t, (-1, x_t.shape[2], x_t.shape[3], x_t.shape[4]))
    
    samples = img_rescale(x_t)
    if resize:
        samples = jax.image.resize(samples, shape=(sample_num, *data_dim), method='nearest')
    return samples

'''
def execute_many_samples(device_memory_threshold, sample_num, trained_state, beta, new_dim, key, resize, data_dim):
    samples = []
    samples_left = sample_num
    while samples_left > 0:
        if samples_left >= device_memory_threshold:
            batch = execute_sample(device_memory_threshold, trained_state, beta, new_dim, key, resize, data_dim)
        else:
            batch = execute_sample(samples_left, trained_state, beta, new_dim, key, resize, data_dim)
        
        samples.append(batch)
        samples_left -= device_memory_threshold
        another_key, key = jax.random.split(key)
    
    return samples
'''

def execute_many_samples(device_memory_threshold, sample_num, trained_state, beta, new_dim, key, resize, data_dim):
    devices = jax.local_device_count()
    assert sample_num % devices == 0
    divided_sample_num = sample_num // devices
    
    samples = []
    if divided_sample_num <= device_memory_threshold:
        samples.append(execute_sample(sample_num, trained_state, beta, new_dim, key, resize, data_dim))
    else:
        raise NotImplementedError
    
    return samples
