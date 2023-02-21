import tensorflow as tf
import tensorflow_datasets as tfds

import jax
import jax.numpy as jnp

from flax import linen as nn

from flax.training import train_state
from tqdm import tqdm
import optax

from matplotlib import pyplot as plt

# Hyperparameters

color = 1
dataset_name = 'mnist'
# 32 is the closest integer to 28 of the form 2^n
img_size = 32

# training details
# lr = 
batch_size = 64
epochs = 100
#sampling_steps = 60000

# beta scheduling
time_steps = 1000
beta_0 = 0.0001
beta_T = 0.02

# U-Net Parameters
ch = 32
groups = 8
scale = (1, 2, 4, 8)
add_attn = (16,)
dropout_rate = 0.5

"""<font size="5">1. Preprocess and load the data"""

# Load MNIST dataset

# Rescale the image to [-1,1] scale
# and resize it to [img_size, img_size] 
def normalize_and_resize(image, label):
    image = 2 * (image / 255) - 1
    image = tf.image.resize(image, [img_size, img_size])
    return image

# Load both train and test set since we are implementing a generative model
ds = tfds.load(dataset_name, as_supervised=True, split='train+test')
ds = ds.map(normalize_and_resize, num_parallel_calls=tf.data.AUTOTUNE).shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
# Convert tensor to numpy array, as the latter is handier
ds = tfds.as_numpy(ds)

"""<font size="5">2. Define forward process"""

beta = jnp.linspace(beta_0, beta_T, time_steps)

@jax.jit
def calculate_necessary_values(beta):
    alpha_ = jnp.cumprod(1.-beta)
    sqrt_alpha_ = jnp.sqrt(alpha_)
    sqrt_1_alpha_ = jnp.sqrt(1.-alpha_)
    
    return alpha_, sqrt_alpha_, sqrt_1_alpha_

# Define forward process using q(x_t | x_0)
@jax.jit
def forward_process(x_0, t, beta, key):
    eps = jax.random.normal(key, x_0.shape)
    alpha_, sqrt_alpha_, sqrt_1_alpha_ = calculate_necessary_values(beta)
    x_t = jnp.reshape(jnp.take(sqrt_alpha_, t), (-1, 1, 1, 1)) * x_0 + jnp.reshape(jnp.take(sqrt_1_alpha_, t), (-1, 1, 1, 1)) * eps
    
    assert x_t.shape == eps.shape
    return x_t, eps

"""<font size="5">3. Define U-Net and DDPM loss function"""

# Architecture details were adopted from the original DDPM implementation

# Sinusoidal embedding
class sin_embedding(nn.Module):
    dim: int
    
    @nn.compact
    def __call__(self, t):
        assert self.dim % 2 == 0
        half_dim = self.dim // 2
        w = jnp.power(10000., - jnp.arange(half_dim) / (half_dim - 1))
        arg = t[:, None] * w[None, :]
        t_emb = jnp.concatenate([jnp.sin(arg), jnp.cos(arg)], 1)
        return t_emb
    
# ResNet-style block at each resolution
class resnet_block(nn.Module):
    ch: int
    groups: int
    dropout_rate: float = 0.5
    
    @nn.compact
    def __call__(self, x, t_emb):
        res = nn.Conv(self.ch, (1, 1))(x)
        
        x = nn.silu(nn.GroupNorm(num_groups=self.groups)(x))
        x = nn.Conv(self.ch, (3, 3))(x)
        
        t_emb = nn.silu(nn.Dense(self.ch)(t_emb))
        
        x = nn.silu(x + jnp.expand_dims(t_emb, (1, 2)))
        x = nn.silu(nn.GroupNorm(num_groups=self.groups)(x))
        x = nn.Dropout(rate=dropout_rate, deterministic=True)(x)
        x = nn.Conv(self.ch, (3, 3))(x)
        
        assert(x.shape == res.shape)
        return x + res

# U-Net
class UNet(nn.Module):
    ch: int
    groups: int
    scale: tuple
    add_attn: tuple = (16,)
    dropout_rate: float = 0.5
    
    @nn.compact
    def __call__(self, x, t):        
        t_emb = sin_embedding(self.ch)(t)
        
        # Why are the following two lines necessary?
        t_emb = nn.silu(nn.Dense(4 * self.ch)(t_emb))
        t_emb = nn.Dense(4 * self.ch)(t_emb)
        
        # Initial layer
        ft = nn.Conv(self.ch, (3, 3))(x)
        
        # Downsampling
        residual = []
        for scale in self.scale:
            ft = resnet_block(self.ch * scale, self.groups, self.dropout_rate)(ft, t_emb)
            
            if self.ch * scale in add_attn:
                attn = nn.SelfAttention(num_heads = 8)(ft, deterministic=True)
                assert ft.shape == attn.shape
                ft += nn.GroupNorm(num_groups=self.groups)(attn)
            
            residual.append(ft)
            
            if scale is not self.scale[-1]:
                ft = nn.avg_pool(ft, (2, 2), (2, 2))
            
            # print(f"Feature dimension at 'downsampling' part: {ft.shape}")
                
        # Middle
        ft = resnet_block(self.ch * scale, self.groups, self.dropout_rate)(ft, t_emb)
        attn = nn.SelfAttention(num_heads = 8)(ft, deterministic=True)
        assert ft.shape == attn.shape
        ft += nn.GroupNorm(num_groups=self.groups)(attn)
        ft = resnet_block(self.ch * scale, self.groups, self.dropout_rate)(ft, t_emb)
        # print(f"Feature dimension at 'middle' part: {ft.shape}")
        
        # Upsampling
        reversed_scale = list(reversed(self.scale))
        for scale in reversed_scale:
            ft = jnp.concatenate([residual.pop(), ft], 3)
            ft = resnet_block(self.ch * scale, self.groups, self.dropout_rate)(ft, t_emb)
            
            if self.ch * scale in add_attn:
                attn = nn.SelfAttention(num_heads = 8)(ft, deterministic=True)
                assert ft.shape == attn.shape
                ft += nn.GroupNorm(num_groups=self.groups)(attn)
            
            if scale is not reversed_scale[-1]:
                B, H, W, C = ft.shape
                ft = jax.image.resize(ft, (B, 2*H, 2*W, C), "nearest")
                ft = nn.Conv(self.ch * scale, (3, 3))(ft)
            
            # print(f"Feature dimension at 'upsampling' part: {ft.shape}")
        
        assert not residual
        
        # Terminal layer
        ft = nn.silu(nn.GroupNorm(num_groups=self.groups)(ft))
        out = nn.Conv(color, (3, 3))(ft)
        
        return out

"""<font size="5">4. Train"""

# Initialize the model and create a train state
key = jax.random.PRNGKey(0) # PRNG Key
dummy_x = jnp.ones(shape=(1, img_size, img_size, color))
dummy_t = jnp.ones(shape=(1,))

model = UNet(ch, groups, scale, add_attn, dropout_rate)
params = model.init(key, dummy_x, dummy_t)['params']
tx = optax.adam(1e-4)

jax.tree_map(lambda x: x.shape, params)

state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

@jax.jit
def train(state, x_t, t, eps):

    def loss_fn(params):
        eps_theta = state.apply_fn({'params': params}, x_t, t)
        loss = jnp.mean((eps - eps_theta) ** 2)
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    
    return loss, state.apply_gradients(grads=grads)


#state_per_epoch = []
for epoch in range(1, epochs+1):    
    print(f"Epoch: {epoch}")
    loss_per_epoch = []

    for label, x_0 in enumerate(tqdm(ds)):
        time_key, noise_key = jax.random.split(key)
        
        t = jax.random.randint(time_key, shape=(x_0.shape[0],), minval=0, maxval=time_steps)
        x_t, eps = forward_process(x_0, t, beta, noise_key)
        
        loss, state = train(state, x_t, t, eps)
        
        loss_per_epoch.append(loss)
    #state_per_epoch.append(state)
    print(f"Loss after {epoch} epoch(s): ", np.mean(loss_per_epoch))
    #if epoch % 50 == 0:
    #    print(f"Loss after {epoch} epoch(s): ", loss)
    
print(f"Finished training after {epochs} epochs")

"""<font size="5">5. Sample"""

@jax.jit
def backward_process(x_t, t, eps_theta, beta, key):
    assert x_t.shape == eps_theta.shape

    alpha_, sqrt_alpha_, sqrt_1_alpha_ = calculate_necessary_values(beta)
    beta_t = jnp.take(beta, t)
    coef = beta_t / (jnp.take(sqrt_1_alpha_, t) ** 0.5) 
    
    mean = 1 / ((1-beta_t) ** 0.5) * (x_t - coef * eps_theta)
    z = jax.random.normal(key = key, shape = x_t.shape)
    
    return mean + (beta_t ** 0.5) * z

@jax.jit
def apply_trained_model(trained_state, x_t, t):
    return trained_state.apply_fn({'params': trained_state.params}, x_t, jnp.array([t,]))

time_key, noise_key = jax.random.split(key)
dummy_key, noise_key = jax.random.split(noise_key)

eps = jax.random.normal(noise_key, (1, 32, 32, 1))
backward_img = [eps]

for t in tqdm(reversed(range(0, time_steps))):
    x_t = backward_img[-1]

    eps_theta = apply_trained_model(state_per_epoch[-1], x_t, t)

    dummy_key, noise_key = jax.random.split(noise_key)
    x_t_1 = backward_process(x_t, t, eps_theta, beta, noise_key)

    backward_img.append(x_t_1)

backward_img = jax.lax.map(jnp.squeeze, jnp.array(backward_img))

for i, img in enumerate(backward_img):
  if i%100 == 0:
    plt.imshow(img, cmap='gray')
    plt.savefig(f"ddpm_mnist_{i}.png")

plt.imshow(backward_img[-1], cmap='gray')
plt.savefig(f"ddpm_mnist_final.png")