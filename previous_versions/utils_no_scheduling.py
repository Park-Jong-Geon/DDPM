import tensorflow as tf
import tensorflow_datasets as tfds
import jax
import jax.numpy as jnp
from flax.training import train_state
import optax
from UNet import UNet
from functools import partial
from matplotlib import pyplot as plt

@jax.jit
def calculate_necessary_values(beta):
    alpha_ = jnp.cumprod(1.-beta)
    sqrt_alpha_ = jnp.sqrt(alpha_)
    sqrt_1_alpha_ = jnp.sqrt(1.-alpha_)
    
    return alpha_, sqrt_alpha_, sqrt_1_alpha_

def normalize(image, label):
    image = 2 * (image / 255) - 1
    return image

def img_resize(image, resize, size1, size2):
    image = tf.image.resize(image, [size1, size2])
    return image

def load_dataset(dataset, batch_size, resize, new_dim):
    ds = tfds.load(dataset, as_supervised=True, split='train+test')
    print(f"Loaded {dataset} dataset", flush=True)
    ds = ds.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
    if resize:
        ds = ds.map(partial(img_resize, resize=resize, size1=new_dim[0], size2=new_dim[1]), num_parallel_calls=tf.data.AUTOTUNE)
    ds = tfds.as_numpy(ds.shuffle(60000).batch(batch_size).prefetch(tf.data.AUTOTUNE))
    return ds

def init_UNet(new_dim, model_args, lr, key):
    dummy_x = jnp.ones(shape=(1, *new_dim))
    dummy_t = jnp.ones(shape=(1, ))
    
    model = UNet(*model_args)
    params = model.init(key, dummy_x, dummy_t)['params']
    tx = optax.adam(lr)

    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    return state

def save_imgs(samples, data_dim, sample_dir, random_seed):
    img_num = 1
    for batch in samples:
        sample_num = jnp.size(batch, axis=0)
        for i in range(sample_num):
            if data_dim[2] == 1:
                plt.imshow(jnp.take(batch, i, axis=0), cmap='gray')
                plt.savefig(f"{sample_dir}/seed{random_seed}_{sample_num}samples_img{img_num}.png")
            else:
                plt.imsave(f"{sample_dir}/seed{random_seed}_{sample_num}samples_img{img_num}.png", jnp.take(batch, i, axis=0))
            
            img_num += 1