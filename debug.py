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

sample_num = 4
epochs = 200
random_seed = 230225
checkpoint = 'save/cifar10_debug'
sample_dir = 'sample/cifar10_debug'

mode = 'train'
dataset = 'cifar10'
lr = 2e-4
batch = 128
save_period = 10000
sample_period = 1000

time_steps = 1000
beta_0 = 0.0001
beta_T = 0.02
ch = 128
groups = 8
scale = [1, 2, 2, 2]
dropout_rate = 0.1
num_heads = 8

print(f"python3 ddpm.py --mode {mode} --dataset {dataset} --batch {batch} --epochs {epochs} --random_seed {random_seed} --checkpoint {checkpoint}")

dataset_info = {
    'cifar10': [[32, 32, 3], [32, 32, 3], False],
    'mnist': [[28, 28, 1], [32, 32, 1], True]
}
data_dim, new_dim, resize = dataset_info[args.dataset]
assert data_dim[2] == 1 or data_dim[2] == 3

# Random seeds
key = jax.random.PRNGKey(args.random_seed)

# Define beta, which is required everywhere in DDPM
beta = jnp.linspace(args.beta_0, args.beta_T, args.time_steps)

# Initialize the model
model_args = [ch, groups, scale, add_attn, dropout_rate, num_heads]
state = init_UNet(new_dim, model_args, args.lr, key)

os.makedirs(checkpoint, exist_ok=True)
os.makedirs(sample_dir, exist_ok=True)

def debug(epochs, ds, state, beta, key, ckpt, save_period, sample_dir, sample_period, sample_batch, new_dim, resize, data_dim):
    time_steps = jnp.size(beta, axis=0)
    for epoch in range(1, epochs+1):
        loss_per_epoch = []
        
        for label, x_0 in enumerate((pbar := tqdm(ds))):
            pbar.set_postfix({'step' : state.step, 'loss' : loss})

            another_key, key = jax.random.split(key)
            t = jax.random.randint(another_key, shape=(x_0.shape[0],), minval=0, maxval=time_steps)
            x_t, eps = forward_process(x_0, t, beta, key)
            loss, state = train(state, x_t, t, eps)

            loss_per_epoch.append(loss)

            if state.step % save_period == 0:
                assert len(os.listdir(ckpt)) < 1e+8
                checkpoints.save_checkpoint(ckpt_dir=ckpt, target=state, step=state.step, keep=1e+8)
                print(f"Checkpoint saved after {state.step} steps at {ckpt}")
            
            if state.step % sample_period == 0:
                samples = execute_sample(sample_batch, state, beta, new_dim, key, resize, data_dim)

                assert(args.sample_num == jnp.size(samples, axis=0))
                for i in range(args.sample_num):
                    if data_dim[2] == 1:
                        plt.imshow(jnp.take(samples, i, axis=0), cmap='gray')
                        plt.savefig(f"{sample_dir}/seed{args.random_seed}_img{i}.png")
                    else:
                        plt.imsave(f"{sample_dir}/seed{args.random_seed}_img{i}.png", jnp.take(samples, i, axis=0), cmap='gray')

        print(f"Loss after {epoch} epoch(s) or {state.step} steps: {np.mean(loss_per_epoch)}", flush=True)
    
    if state.step % save_period != 0:
        assert len(os.listdir(ckpt)) < 1e+8
        checkpoints.save_checkpoint(ckpt_dir=ckpt, target=state, step=state.step, keep=1e+8)
        print(f"Checkpoint saved after {state.step} steps at {ckpt}", flush=True)

    print(f"Finished training after {epoch} epochs or {state.step} steps", flush=True)

    return state

debug(epochs, ds, state, beta, key, ckpt, save_period, sample_dir, sample_period, sample_batch, new_dim, resize, data_dim)