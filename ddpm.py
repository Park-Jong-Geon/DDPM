import os
from glob import glob
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
parser.add_argument('--dataset', type=str, required=True, help='Required')
parser.add_argument('--lr', type=float, required=True, help='Required')
parser.add_argument('--batch', type=int, help='Required in train mode')
parser.add_argument('--epochs', type=int, help='Required in train mode')
parser.add_argument('--random_seed', type=int, required=True, help='Required')
parser.add_argument('--checkpoint', type=str, required=True, help='Required')

parser.add_argument('--save_period', type=int, default=100)
parser.add_argument('--train_further', action='store_true')
parser.add_argument('--old_checkpoint', type=str, default=None)

parser.add_argument('--sample_dir', type=str, default=None)

parser.add_argument('--time_steps', type=int, default=1000)
parser.add_argument('--beta_0', type=float, default=0.0001)
parser.add_argument('--beta_T', type=float, default=0.02)

parser.add_argument('--ch', type=int, default=128)
parser.add_argument('--groups', type=int, default=8)
parser.add_argument('--scale', type=int, default=[1, 2, 2, 2], nargs="+")
parser.add_argument('--add_attn', type=int, default=[2,], nargs="+")
parser.add_argument('--dropout_rate', type=float, default=0.1)

args = parser.parse_args()

dataset_info = {
    'cifar10': [[32, 32, 3], [32, 32, 3], False],
    'mnist': [[28, 28, 1], [64, 64, 1], True]
}
data_dim, new_dim, resize = dataset_info[args.dataset]

# Random seeds
key = jax.random.PRNGKey(args.random_seed)

# Define beta, which is required everywhere in DDPM
beta = jnp.linspace(args.beta_0, args.beta_T, args.time_steps)

# Initialize the model
model_args = [args.ch, args.groups, tuple(args.scale), tuple(args.add_attn), args.dropout_rate]
state = init_UNet(new_dim, model_args, args.lr, key)

# Train
if args.mode == 'train':
    os.makedirs(args.checkpoint, exist_ok=False)

    # Print initial training settings
    print(ctime(time()))
    print(f"dataset={args.dataset} lr={args.lr} batch_size={args.batch} epochs={args.epochs}")
    print(f"checkpoint={args.checkpoint}")
    print(f"train_further={args.train_further} old_checkpoint={args.old_checkpoint}")
    print(f"Beta scheduling : time_steps={args.time_steps} beta_0={args.beta_0} beta_T={args.beta_T}")
    print(f"U-Net Parameters : ch={args.ch} groups={args.groups} scale={tuple(args.scale)} add_attn={tuple(args.add_attn)} dropout_rate={args.dropout_rate}")

    # Load dataset
    ds = load_dataset(args.dataset, args.batch, resize, new_dim)

    # Train
    if args.train_further:
        state = checkpoints.restore_checkpoint(ckpt_dir=args.old_checkpoint, target=state)
        print(f"Checkpoint restored from {args.old_checkpoint}")
    state = execute_train(args.epochs, ds, state, beta, key, args.checkpoint, args.save_period)

    print("")

# Sample
else:
    os.makedirs(args.sample_dir, exist_ok=False)

    restored_state = checkpoints.restore_checkpoint(ckpt_dir=args.checkpoint, target=state)
    print(f"Loaded trained model from {args.checkpoint}")

    backward_img = execute_sample(restored_state, beta, new_dim, key, resize, data_dim)

    for t in range(args.time_steps):
        if t % 100 == 0:
            plt.imshow(backward_img[t], cmap='gray')
            plt.savefig(f"{args.sample_dir}/step_{t}.png")

    assert t == args.time_steps - 1
    plt.imshow(backward_img[t], cmap='gray')
    plt.savefig(f"{args.sample_dir}/step_{t}.png")