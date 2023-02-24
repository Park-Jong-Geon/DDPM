import jax
import jax.numpy as jnp
import numpy as np
from flax.training import checkpoints
from utils import calculate_necessary_values
from tqdm import tqdm
import os

@jax.jit
def forward_process(x_0, t, beta, key):
    eps = jax.random.normal(key, x_0.shape)
    alpha_, sqrt_alpha_, sqrt_1_alpha_ = calculate_necessary_values(beta)
    x_t = jnp.reshape(jnp.take(sqrt_alpha_, t), (-1, 1, 1, 1)) * x_0 + jnp.reshape(jnp.take(sqrt_1_alpha_, t), (-1, 1, 1, 1)) * eps
    
    assert x_t.shape == eps.shape
    return x_t, eps

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

def execute_train(epochs, ds, state, beta, key, ckpt, save_period):
    time_steps = jnp.size(beta, axis=0)
    for epoch in range(1, epochs+1):
        print(f"Epoch: {epoch}")

        loss_per_epoch = []
        
        for label, x_0 in enumerate((pbar := tqdm(ds))):
            another_key, key = jax.random.split(key)
            t = jax.random.randint(another_key, shape=(x_0.shape[0],), minval=0, maxval=time_steps)
            x_t, eps = forward_process(x_0, t, beta, key)
            loss, state = train(state, x_t, t, eps)

            loss_per_epoch.append(loss)

            pbar.set_postfix({'step' : state.step, 'loss' : loss})
            if state.step % save_period == 0:
                assert len(os.listdir(ckpt)) < 1e+8
                checkpoints.save_checkpoint(ckpt_dir=ckpt, target=state, step=state.step, keep=1e+8)
                print(f"Checkpoint saved after {state.step} steps at {ckpt}")

        print(f"Loss after {epoch} epoch(s) or {state.step} steps: {np.mean(loss_per_epoch)}", flush=True)
    
    if state.step % save_period != 0:
        assert len(os.listdir(ckpt)) < 1e+8
        checkpoints.save_checkpoint(ckpt_dir=ckpt, target=state, step=state.step, keep=1e+8)
        print(f"Checkpoint saved after {state.step} steps at {ckpt}", flush=True)

    print(f"Finished training after {epoch} epochs or {state.step} steps", flush=True)

    return state