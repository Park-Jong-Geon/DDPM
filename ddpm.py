import os
from time import ctime, time
import argparse
import jax
import jax.numpy as jnp
from utils import load_dataset
from utils import init_UNet
from train_func import execute_train
from sample_func import execute_sample
from flax.training import checkpoints
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description="Train or sample from DDPM")

parser.add_argument('--mode', type=str, choices=['train', 'sample'], required=True, help='Required')
parser.add_argument('--sample_num', type=int, default=256)
parser.add_argument('--dataset', type=str, required=True, help='Required')
parser.add_argument('--lr', type=float, required=True, help='Required')
parser.add_argument('--batch', type=int, help='Required in train mode')
parser.add_argument('--epochs', type=int, help='Required in train mode')
parser.add_argument('--random_seed', type=int, required=True, help='Required')
parser.add_argument('--checkpoint', type=str, required=True, help='Required')

parser.add_argument('--save_period', type=int, default=10000)
parser.add_argument('--train_further', action='store_true')
parser.add_argument('--old_checkpoint', type=str, default=None)

parser.add_argument('--sample_dir', type=str, default=None)

parser.add_argument('--time_steps', type=int, default=1000)
parser.add_argument('--beta_0', type=float, default=0.0001)
parser.add_argument('--beta_T', type=float, default=0.02)

parser.add_argument('--ch', type=int, default=128)
parser.add_argument('--groups', type=int, default=8)
parser.add_argument('--scale', type=int, default=[1, 2, 2, 2], nargs="+")
parser.add_argument('--add_attn', type=int, default=[16,], nargs="+")
parser.add_argument('--dropout_rate', type=float, default=0.1)
parser.add_argument('--num_heads', type=int, default=8)
parser.add_argument('--num_res_blocks', type=int, default=2)

args = parser.parse_args()

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
model_args = [args.ch, args.groups, tuple(args.scale), tuple(args.add_attn), args.dropout_rate, args.num_heads, args.num_res_blocks]
state = init_UNet(new_dim, model_args, args.lr, key)

# Print initial training settings
def print_settings(args):
    print(f"python3 ddpm.py --mode {args.mode} --dataset {args.dataset} --batch {args.batch} --epochs {args.epochs} --random_seed {args.random_seed} --checkpoint {args.checkpoint}", flush=True)
    print(ctime(time()), flush=True)
    print(f"dataset={args.dataset} lr={args.lr} batch_size={args.batch} epochs={args.epochs}", flush=True)
    print(f"checkpoint={args.checkpoint}", flush=True)
    print(f"train_further={args.train_further} old_checkpoint={args.old_checkpoint}", flush=True)
    print(f"Beta scheduling : time_steps={args.time_steps} beta_0={args.beta_0} beta_T={args.beta_T}", flush=True)
    print(f"U-Net Parameters : ch={args.ch} groups={args.groups} scale={tuple(args.scale)} add_attn={tuple(args.add_attn)} dropout_rate={args.dropout_rate} num_heads={args.num_heads} num_res_blocks={args.num_res_blocks}", flush=True)
    print(f"Random seed : {args.random_seed}", flush=True)
    print(f"Save path : {args.checkpoint}", flush=True)


# Train
if args.mode == 'train':
    os.makedirs(args.checkpoint, exist_ok=True)
    print_settings()

    # Load dataset
    ds = load_dataset(args.dataset, args.batch, resize, new_dim)

    # Train
    if args.train_further:
        state = checkpoints.restore_checkpoint(ckpt_dir=args.old_checkpoint, target=state)
        print(f"Checkpoint restored from {args.old_checkpoint}", flush=True)
    state = execute_train(args.epochs, ds, state, beta, key, args.checkpoint, args.save_period)

    print("")

# Sample
else:
    os.makedirs(args.sample_dir, exist_ok=True)

    restored_state = checkpoints.restore_checkpoint(ckpt_dir=args.checkpoint, target=state)
    print(f"Loaded trained model from {args.checkpoint}", flush=True)
    
    samples = execute_sample(args.sample_num, restored_state, beta, new_dim, key, resize, data_dim)

    assert(args.sample_num == jnp.size(samples, axis=0))
    for i in range(args.sample_num):
        if data_dim[2] == 1:
            plt.imshow(jnp.take(samples, i, axis=0), cmap='gray')
            plt.savefig(f"{args.sample_dir}/seed{args.random_seed}_img{i}.png")
        else:
            plt.imsave(f"{args.sample_dir}/seed{args.random_seed}_img{i}.png", jnp.take(samples, i, axis=0))

'''
    for t in range(args.time_steps):
        if t % 100 == 0:
            plt.imshow(backward_img[t], cmap='gray')
            plt.savefig(f"{args.sample_dir}/step_{t}.png")

    assert t == args.time_steps - 1
    plt.imshow(backward_img[t], cmap='gray')
    plt.savefig(f"{args.sample_dir}/step_{t}.png")
'''