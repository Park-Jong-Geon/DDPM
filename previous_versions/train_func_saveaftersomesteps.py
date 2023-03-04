import tensorflow as tf
import jax
import jax.numpy as jnp
import numpy as np
from flax.training import checkpoints
from sample_func import execute_many_samples
from utils import calculate_necessary_values, save_imgs, update_ema
from tqdm import tqdm
import os

@jax.jit
def forward_process(x_0, t, beta, eps):
    assert x_0.shape == eps.shape
    
    alpha_, sqrt_alpha_, sqrt_1_alpha_ = calculate_necessary_values(beta)
    x_t = jnp.reshape(jnp.take(sqrt_alpha_, t), (-1, 1, 1, 1)) * x_0 + jnp.reshape(jnp.take(sqrt_1_alpha_, t), (-1, 1, 1, 1)) * eps
    return x_t

@jax.jit
def train(state, x_t, t, eps):

    def loss_fn(params):
        eps_theta = state.apply_fn({'params': params}, x_t, t)
        loss = jnp.mean((eps - eps_theta) ** 2)
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return loss, state

def execute_train(epochs, ds, state, beta, key, ckpt, save_period, rand_flip, 
                  train_and_sample=False, train_and_sample_params=None,
                  use_ema=True, ema_decay=0.9999):
    if train_and_sample:
        key_ = key
        device_memory_threshold, sample_period, sample_dir, sample_num, ds_info, random_seed = train_and_sample_params
        data_dim, new_dim, resize = ds_info
        os.makedirs(sample_dir, exist_ok=True)
    
    if use_ema:
        params_ema = state.params
    
    time_steps = jnp.size(beta, axis=0)
    for epoch in range(1, epochs+1):
        loss_per_epoch = []
        
        for x_0 in (pbar := tqdm(ds)):
            if rand_flip:
                x_0 = jnp.array(tf.image.random_flip_left_right(x_0))
            
            another_key, key = jax.random.split(key)
            eps = jax.random.normal(key, x_0.shape)
            t = jax.random.randint(another_key, shape=(x_0.shape[0],), minval=0, maxval=time_steps)
            x_t = forward_process(x_0, t, beta, eps)
            loss, state = train(state, x_t, t, eps)
            
            if use_ema:
                params_ema = update_ema(params_ema, state.params, ema_decay)

            loss_per_epoch.append(loss)

            if state.step % save_period == 0:
                assert len(os.listdir(ckpt)) < 1e+8
                checkpoints.save_checkpoint(ckpt_dir=ckpt, target=state, step=state.step, keep=1e+8)
                print(f"Checkpoint saved after {state.step} steps at {ckpt}", flush=True)
                
                if use_ema:
                    another_state = state.replace(params=params_ema)
            
            if train_and_sample and (state.step % sample_period == 0):
                samples = execute_many_samples(device_memory_threshold, sample_num, state, beta, new_dim, key, resize, data_dim)
                
                save_imgs(samples, data_dim, sample_dir, state.step, random_seed, sample_num)
            
            pbar.set_description(f"Training at epoch {epoch}")
            pbar.set_postfix({'step' : state.step, 'loss' : loss})

        print(f"Loss after {epoch} epoch(s) or {state.step} steps: {np.mean(loss_per_epoch)}", flush=True)
    
    if use_ema:
        state = state.replace(params=params_ema)
    
    if state.step % save_period != 0:
        assert len(os.listdir(ckpt)) < 1e+8
        checkpoints.save_checkpoint(ckpt_dir=ckpt, target=state, step=state.step, keep=1e+8)
        print(f"Checkpoint saved after {state.step} steps at {ckpt}", flush=True)
    
    if train_and_sample and (state.step % sample_period != 0):
        samples = execute_many_samples(device_memory_threshold, sample_num, state, beta, new_dim, key, resize, data_dim)
        
        save_imgs(samples, data_dim, sample_dir, state.step, random_seed, sample_num)

    print(f"Finished training after {epoch} epochs or {state.step} steps", flush=True)

    return state