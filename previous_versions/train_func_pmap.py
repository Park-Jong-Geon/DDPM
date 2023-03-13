import tensorflow as tf
import jax
import jax.numpy as jnp
import numpy as np
import flax
from flax.training import checkpoints
from sample_func import execute_many_samples
from utils import calculate_necessary_values, save_imgs, update_ema
from tqdm import tqdm
import os
from functools import partial
'''
@jax.jit
def forward_process(x_0, t, beta, eps):    
    assert x_0.shape == eps.shape
    
    alpha_, sqrt_alpha_, sqrt_1_alpha_ = calculate_necessary_values(beta)
    x_t = jnp.reshape(sqrt_alpha_[t], (-1, 1, 1, 1)) * x_0 + jnp.reshape(sqrt_1_alpha_[t], (-1, 1, 1, 1)) * eps
    return x_t
'''

@jax.jit
def forward_process(x_0, t, beta, eps):    
    assert x_0.shape == eps.shape
    
    alpha_, sqrt_alpha_, sqrt_1_alpha_ = calculate_necessary_values(beta)
    x_t = jnp.reshape(sqrt_alpha_[t], (t.shape[0], -1, 1, 1, 1)) * x_0 + jnp.reshape(sqrt_1_alpha_[t], (t.shape[0], -1, 1, 1, 1)) * eps
    return x_t

def execute_train(epochs, ds, state, beta, key, ckpt, save_period, rand_flip, 
                  train_and_sample=False, train_and_sample_params=None,
                  use_ema=True, ema_decay=0.9999):
    devices = jax.local_device_count()
    state = flax.jax_utils.replicate(state)
    
    def save_ckpt():
        no_ema = state
        no_ema = flax.jax_utils.unreplicate(no_ema)
        
        assert (len(os.listdir(ckpt)) + 10) < 1e+8
        
        if use_ema:
            ema = state.replace(params=params_ema)
            ema = flax.jax_utils.unreplicate(ema)
            checkpoints.save_checkpoint(ckpt_dir=ckpt, target=ema, step=ema.step, keep=1e+8, overwrite=True)
            checkpoints.save_checkpoint(ckpt_dir=f"{ckpt}_no_ema", target=no_ema, step=no_ema.step, keep=1e+8, overwrite=True)
        else:
            checkpoints.save_checkpoint(ckpt_dir=ckpt, target=no_ema, step=no_ema.step, keep=1e+8, overwrite=True)
        
        print(f"Checkpoint saved at {ckpt} and {ckpt}_no_ema", flush=True)
        
        if train_and_sample:
            samples = execute_many_samples(device_memory_threshold, sample_num, no_ema, beta, new_dim, key_, resize, data_dim)    
            save_imgs(samples, data_dim, sample_dir, no_ema.step, random_seed, sample_num)
    
    def train(state, x_t, t):
        def loss_fn(params):
            eps_theta = state.apply_fn({'params': params}, x_t, t)
            loss = jnp.mean((eps - eps_theta) ** 2)
            return loss

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params)

        grads = jax.lax.pmean(grads, axis_name='batch')
        loss = jax.lax.pmean(loss, axis_name='batch')
        
        state = state.apply_gradients(grads=grads)

        return loss, state
    
    if train_and_sample:
        key_ = key
        device_memory_threshold, sample_period, sample_dir, sample_num, ds_info, random_seed = train_and_sample_params
        data_dim, new_dim, resize = ds_info
        os.makedirs(sample_dir, exist_ok=True)
    
    if use_ema:
        params_ema = state.params
        os.makedirs(f"{ckpt}_no_ema", exist_ok=True)
    
    time_steps = jnp.size(beta, axis=0)
    for epoch in range(1, epochs+1):
        loss_per_epoch = []
        
        for x_0 in (pbar := tqdm(ds)):
            if rand_flip:
                x_0 = jnp.array(tf.image.random_flip_left_right(x_0))
            
            another_key, key = jax.random.split(key)
            if x_0.shape[0] % devices == 0:
                div_shape = x_0.shape[0] // devices
                
                x_0 = jnp.reshape(x_0, newshape=(devices, div_shape, x_0.shape[1], x_0.shape[2], x_0.shape[3]))
                eps = jax.random.normal(key, shape=x_0.shape)
                t = jax.random.randint(another_key, shape=(devices, div_shape), minval=0, maxval=time_steps)
                
                x_t = forward_process(x_0, t, beta, eps)
                loss, state = jax.pmap(train, axis_name='batch')(state, x_t, t)
            else:
                raise NotImplementedError
            
            if use_ema:
                params_ema = update_ema(params_ema, state.params, ema_decay)
            
            loss_per_epoch.append(loss[0])
            
            pbar.set_description(f"Training at epoch {epoch}")
            pbar.set_postfix({'step' : state.step[0], 'loss' : loss[0]})
            
            if state.step[0] % save_period == 0:
                save_ckpt()
        '''
        if epoch == 1:
            lowest_epoch_loss = np.mean(loss_per_epoch)
        '''
        print(f"Loss after {epoch} epoch(s) or {state.step[0]} steps: {np.mean(loss_per_epoch)}", flush=True)   
        '''
        if lowest_epoch_loss > np.mean(loss_per_epoch):
            lowest_epoch_loss = np.mean(loss_per_epoch)
            save_ckpt()
        '''
    if state.step[0] % save_period != 0:
        save_ckpt()
            
    print(f"Finished training after {epoch} epochs or {state.step[0]} steps", flush=True)    