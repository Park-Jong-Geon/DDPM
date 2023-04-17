from tqdm import tqdm
from functools import partial
import numpy as np

import jax
import jax.numpy as jnp
import flax
from flax.training import train_state, checkpoints
import optax

from UNet import UNet
from utils import utils
from jax_fid.fid import FID

class DDPM(utils):
    def __init__(self):
        super().__init__()
        
        self.best_fid = 100
        
        # Random seeds
        self.key = jax.random.PRNGKey(self.random_seed)

        # Beta and etc
        self.beta = jnp.linspace(self.beta_0, self.beta_T, self.time_steps)
        self.alpha_ = jnp.cumprod(1.-self.beta)
        self.sqrt_alpha_ = jnp.sqrt(self.alpha_)
        self.sqrt_1_alpha_ = jnp.sqrt(1.-self.alpha_)

        # Initialize the model
        # model = UNet()
        model = UNet(self.ch, self.groups, self.scale, self.add_attn, self.dropout_rate, self.num_res_blocks)
        self.img_dim = self.data_dim if self.new_dim == None else self.new_dim
        params = model.init(self.key, jnp.ones(shape=(1, *self.img_dim)), jnp.ones(shape=(1, )))['params']
        '''
        scheduled_lr = optax.join_schedules(
                [optax.linear_schedule(init_value=0, end_value=self.lr, transition_steps=self.warmup_steps), optax.constant_schedule(self.lr)],
                [self.warmup_steps]
            )
        '''
        scheduled_lr = optax.warmup_exponential_decay_schedule(
                init_value=0.0,
                peak_value=self.lr,
                warmup_steps=self.warmup_steps,
                decay_rate=1,
                transition_steps=1
            )
        tx = optax.chain(optax.clip(self.grad_clip), optax.adam(scheduled_lr))
        
        self.state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
        self.ema_state = self.state
        
        self.fid = FID(self.random_seed)

        def update(state, ema_state, x_0, t, eps):
            
            def loss_fn(params, x_t, t, eps):
                eps_theta = state.apply_fn({'params': params}, x_t, t)
                loss = jnp.mean((eps - eps_theta) ** 2)
                return loss
            
            x_t = jnp.reshape(self.sqrt_alpha_[t], (-1, 1, 1, 1)) * x_0 + jnp.reshape(self.sqrt_1_alpha_[t], (-1, 1, 1, 1)) * eps
            
            grad_fn = jax.value_and_grad(loss_fn)
            loss, grads = grad_fn(state.params, x_t, t, eps)
            loss = jax.lax.pmean(loss, axis_name='batch')
            grads = jax.lax.pmean(grads, axis_name='batch')
            
            state = state.apply_gradients(grads=grads)
            updated_ema = jax.tree_map(lambda p_e, p: self.ema_decay * p_e + (1 - self.ema_decay) * p, 
                                       ema_state.params, state.params)
            ema_state = ema_state.replace(params=updated_ema, step=state.step)
            return loss, state, ema_state

        def apply_model(state, x, t):
            return state.apply_fn({'params': state.params}, x, t)

        def backward_process(x_t, t, eps_theta, eps):
            beta_t = self.beta[t]
            
            coef = beta_t / self.sqrt_1_alpha_[t]
            mean = jnp.reshape((1 / jnp.sqrt(1-beta_t)), (-1, 1, 1, 1)) * (x_t - jnp.reshape(coef, (-1, 1, 1, 1)) * eps_theta)

            return mean + jnp.reshape(jnp.sqrt(beta_t), (-1, 1, 1, 1)) * eps
        
        self.update = jax.pmap(update, axis_name='batch')
        self.apply_model = jax.pmap(apply_model, axis_name='batch')
        self.backward_process = jax.pmap(backward_process, axis_name='batch')
        
    def train(self, ds, ckpt=None, ckpt_ema=None):
        if ckpt != None:
            self.state = ckpt
            self.ema_state = ckpt_ema if ckpt_ema != None else ckpt
        
        self.state = flax.jax_utils.replicate(self.state)
        self.ema_state = flax.jax_utils.replicate(self.ema_state)
        
        key = self.key
        # cum_loss = jnp.array([])
        
        for x_0 in (pbar := tqdm(ds)):
            another_key, key = jax.random.split(key)
            
            t = jax.random.randint(key=another_key, shape=x_0.shape[0:2], minval=0, maxval=self.time_steps)
            eps = jax.random.normal(key=key, shape=x_0.shape)
            
            loss, self.state, self.ema_state = self.update(self.state, self.ema_state, x_0, t, eps)

            # cum_loss = jnp.append(cum_loss, loss[0])
            pbar.set_postfix({'step' : self.ema_state.step[0], 'loss' : loss[0]})
            '''
            if self.state.step[0] % self.calc_loss_period == 0:
                assert jnp.size(cum_loss) == self.calc_loss_period
                print(f"Loss after {self.state.step[0]} step(s): {jnp.mean(cum_loss)}", flush=True)
                cum_loss = jnp.array([])
            '''
            if (self.state.step[0] >= 0.75*self.step) and (self.state.step[0] % self.check_period == 0):
                state = flax.jax_utils.unreplicate(self.state)
                ema_state = flax.jax_utils.unreplicate(self.ema_state)
                self.sample(ema_state, 2000)
                
                current_fid = self.fid.calculate_fid(self.stats_dir, self.sample_dir)
                print(f"Current FID: {current_fid}", flush=True)
                
                if current_fid < self.best_fid:
                    self.save_ckpt(state, ema_state)

                    self.clear_dir(self.temp_dir)
                    self.best_fid = current_fid
                    print(f"Best FID achieved by far", flush=True)
                    
                    self.move_files(self.sample_dir, self.temp_dir)
                
                self.clear_dir(self.sample_dir)

            if self.state.step[0] >= self.step:
                break

    def sample(self, state, sample_num):
        def img_rescale(imgs):
            imgs = np.clip(imgs, a_min=-1, a_max=1)
            imgs = np.around((imgs + 1) / 2 * 255)
            return np.array(imgs).astype(np.uint8)
        
        if sample_num % self.devices != 0:
            raise NotImplementedError
        
        state = flax.jax_utils.replicate(state)
        key = self.key
        
        samples = None
        remaining_samples = sample_num
        while(remaining_samples > 0):
            num = min(remaining_samples, self.devices*self.device_memory_threshold)
            x = jax.random.normal(key, (self.devices, num//self.devices, *self.img_dim))
            
            for t in (pbar := tqdm(reversed(range(0, self.time_steps)))):
                pbar.set_description("Sampling")
                
                vec_t = jnp.int16(t * jnp.ones(shape=(self.devices, num//self.devices)))
                
                eps_theta = self.apply_model(state, x, vec_t)
                
                if t > 0:
                    another_key, key = jax.random.split(key)
                    eps = jax.random.normal(key=key, shape=x.shape)
                else:
                    eps = jnp.zeros(shape=x.shape)
                
                x_prev = self.backward_process(x, vec_t, eps_theta, eps)
                x = x_prev
            
            x = jnp.reshape(x, (-1, *x.shape[2:5]))   
        
            if self.new_dim != None:
                samples = jax.image.resize(samples, shape=(num, *self.new_dim), method='nearest')
                
            if samples == None:
                samples = x
            else:
                samples = jnp.concatenate([samples, x])
            
            remaining_samples -= num
        
        samples = img_rescale(samples)
        state = flax.jax_utils.unreplicate(state)  
        self.save_imgs(samples, state.step)
        