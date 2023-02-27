import jax
import jax.numpy as jnp
import numpy as np
from flax.training import checkpoints
from utils import calculate_necessary_values, load_dataset, init_UNet
import os
from tqdm import tqdm
from time import ctime, time
from functools import partial
from train_func import forward_process, train
from sample_func import backward_process, apply_trained_model, img_rescale, execute_sample
from matplotlib import pyplot as plt

def debug(epochs, ds, state, beta, key, ckpt, save_period, sample_dir, sample_period, sample_num, new_dim, resize, data_dim):
    time_steps = jnp.size(beta, axis=0)
    for epoch in range(1, epochs+1):
        loss_per_epoch = []
        
        for x_0 in (pbar := tqdm(ds)):
            another_key, key = jax.random.split(key)
            eps = jax.random.normal(key, x_0.shape)
            t = jax.random.randint(another_key, shape=(x_0.shape[0],), minval=0, maxval=time_steps)
            x_t = forward_process(x_0, t, beta, eps)
            loss, state = train(state, x_t, t, eps)

            loss_per_epoch.append(loss)

            if state.step % save_period == 0:
                assert len(os.listdir(ckpt)) < 1e+8
                checkpoints.save_checkpoint(ckpt_dir=ckpt, target=state, step=state.step, keep=1e+8)
                print(f"Checkpoint saved after {state.step} steps at {ckpt}")
            
            if state.step % sample_period == 0:
                samples = execute_sample(sample_num, state, beta, new_dim, key, resize, data_dim)

                assert sample_num == jnp.size(samples, axis=0)
                for i in range(sample_num):
                    if data_dim[2] == 1:
                        plt.imshow(jnp.take(samples, i, axis=0), cmap='gray')
                        plt.savefig(f"{sample_dir}/seed{random_seed}_step{state.step}_img{i}.png")
                    else:
                        plt.imsave(f"{sample_dir}/seed{random_seed}_step{state.step}_img{i}.png", jnp.take(samples, i, axis=0))
        
            pbar.set_description(f"Training at epoch {epoch}")
            pbar.set_postfix({'step' : state.step, 'loss' : loss})

        print(f"Loss after {epoch} epoch(s) or {state.step} steps: {np.mean(loss_per_epoch)}", flush=True)
    
    if state.step % save_period != 0:
        assert len(os.listdir(ckpt)) < 1e+8
        checkpoints.save_checkpoint(ckpt_dir=ckpt, target=state, step=state.step, keep=1e+8)
        print(f"Checkpoint saved after {state.step} steps at {ckpt}", flush=True)

    print(f"Finished training after {epoch} epochs or {state.step} steps", flush=True)

    return state

sample_num = 4
epochs = 30
random_seed = 230227
checkpoint = 'save/mnist_debug'
sample_dir = 'sample/mnist_debug_rev'

mode = 'train'
dataset = 'mnist'
lr = 2e-4
batch = 128
save_period = 10000
sample_period = 1000

time_steps = 1000
beta_0 = 0.0001
beta_T = 0.02
ch = 32
groups = 8
# scale = [1, 2, 2, 2]
scale = [1, 2, 4, 8]
add_attn = [16]
dropout_rate = 0.1
# num_heads = 8
num_heads = 1
num_res_blocks = 2

print(f"python3 ddpm.py --mode {mode} --dataset {dataset} --batch {batch} --epochs {epochs} --random_seed {random_seed} --checkpoint {checkpoint}")

dataset_info = {
    'cifar10': [[32, 32, 3], [32, 32, 3], False],
    'mnist': [[28, 28, 1], [32, 32, 1], True]
}
data_dim, new_dim, resize = dataset_info[dataset]
assert data_dim[2] == 1 or data_dim[2] == 3

# Random seeds
key = jax.random.PRNGKey(random_seed)

# Define beta, which is required everywhere in DDPM
beta = jnp.linspace(beta_0, beta_T, time_steps)

# Initialize the model
model_args = [ch, groups, scale, add_attn, dropout_rate, num_heads, num_res_blocks]
state = init_UNet(new_dim, model_args, lr, key)

os.makedirs(checkpoint, exist_ok=True)
os.makedirs(sample_dir, exist_ok=True)

ds = load_dataset(dataset, batch, resize, new_dim)

debug(epochs, ds, state, beta, key, checkpoint, save_period, sample_dir, sample_period, sample_num, new_dim, resize, data_dim)