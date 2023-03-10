import jax
import jax.numpy as jnp
import flax.linen as nn
import flax.linen.initializers as init

# Architecture details were adopted from the original DDPM implementation

# Sinusoidal embedding
class sin_embedding(nn.Module):
    dim: int
    
    @nn.compact
    def __call__(self, t):
        assert self.dim % 2 == 0
        assert len(t.shape) == 1
        
        half_dim = self.dim // 2
        emb = jnp.log(10000) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=1)
        
        assert len(emb.shape) == 2
        assert emb.shape[1] == self.dim
        return emb

# ResNet-style block at each resolution
class resnet_block(nn.Module):
    ch: int
    groups: int
    dropout_rate: float
    
    @nn.compact
    def __call__(self, x, t_emb):
        ft = nn.silu(nn.GroupNorm(num_groups=self.groups)(x))
        ft = nn.Conv(self.ch, (3, 3))(ft)
        
        t_emb = nn.Dense(self.ch)(nn.silu(t_emb))
        ft = ft + jnp.expand_dims(t_emb, (1, 2))

        ft = nn.silu(nn.GroupNorm(num_groups=self.groups)(ft))
        ft = nn.Dropout(rate=self.dropout_rate, deterministic=True)(ft)
        ft = nn.Conv(self.ch, (3, 3))(ft)
        
        if x.shape != ft.shape:
            x = nn.Conv(self.ch, (1, 1))(x)
        
        assert ft.shape == x.shape
        return ft + x

def contract_inner(x, y):
  """tensordot(x, y, 1)."""
  x_chars = list(string.ascii_lowercase[:len(x.shape)])
  y_chars = list(string.ascii_uppercase[:len(y.shape)])
  assert len(x_chars) == len(x.shape) and len(y_chars) == len(y.shape)
  y_chars[0] = x_chars[-1]  # first axis of y and last of x get summed
  out_chars = x_chars[:-1] + y_chars[1:]
  return _einsum(x_chars, y_chars, out_chars, x, y)

def nin(x, *, name, num_units, init_scale=1.):
  with tf.variable_scope(name):
    in_dim = int(x.shape[-1])
    W = tf.get_variable('W', shape=[in_dim, num_units], initializer=default_init(scale=init_scale), dtype=DEFAULT_DTYPE)
    b = tf.get_variable('b', shape=[num_units], initializer=tf.constant_initializer(0.), dtype=DEFAULT_DTYPE)
    y = contract_inner(x, W) + b
    assert y.shape == x.shape[:-1] + [num_units]
    return y

# Attention block
class SelfAttention(nn.Module):
    num_groups: int
    
    @nn.compact
    def __call__(self, x):
        B, H, W, C = x.shape
        ft = nn.GroupNorm(num_groups=self.num_groups)(x)
        q = nn.Dense(C)(ft)
        k = nn.Dense(C)(ft)
        v = nn.Dense(C)(ft)
        
        w = jnp.einsum('bhwc,bHWc->bhwHW', q, k) * (C ** (-0.5))
        w = jnp.reshape(w, [B, H, W, H*W])
        w = nn.softmax(w)
        w = jnp.reshape(w, [B, H, W, H, W])

        attn = jnp.einsum('bhwHW,bHWc->bhwc', w, v)
        attn = nn.Dense(C)(attn)

        assert x.shape == attn.shape
        return x + attn

# U-Net
class UNet(nn.Module):
    ch: int
    groups: int
    scale: tuple
    add_attn: tuple
    dropout_rate: float
    num_heads: int
    num_res_blocks : int
    
    @nn.compact
    def __call__(self, x, t):        
        t_emb = sin_embedding(self.ch)(t)
        
        # Why are the following two lines necessary?
        t_emb = nn.silu(nn.Dense(4 * self.ch)(t_emb))
        t_emb = nn.Dense(4 * self.ch)(t_emb)
        
        # Initial layer
        ft = nn.Conv(self.ch, (3, 3))(x)
        
        # Downsampling
        assert ft.shape[1] == ft.shape[2]
        scale_len = len(self.scale)
        residual = [ft]
        for i, scale in enumerate(self.scale):
            for j in range(self.num_res_blocks):
                ft = resnet_block(self.ch * scale, self.groups, self.dropout_rate)(ft, t_emb)
                
                if ft.shape[1] in self.add_attn:
                    ft = SelfAttention(num_groups=self.groups)(ft)
                
                residual.append(ft)
            
            if i != scale_len-1:
                #ft = nn.avg_pool(ft, (2, 2), (2, 2))
                ft = nn.Conv(self.ch * scale, (3, 3), 2, (1, 1))(ft)
                residual.append(ft)
            # print(f"Feature dimension at 'downsampling' part: {ft.shape}")
                
        # Middle
        ft = resnet_block(self.ch * scale, self.groups, self.dropout_rate)(ft, t_emb)
        ft = SelfAttention(num_groups=self.groups)(ft)
        ft = resnet_block(self.ch * scale, self.groups, self.dropout_rate)(ft, t_emb)
        # print(f"Feature dimension at 'middle' part: {ft.shape}")
        
        # Upsampling
        for i, scale in enumerate(reversed(self.scale)):
            for j in range(self.num_res_blocks + 1):
                assert residual[-1].shape[0:3] == ft.shape[0:3]
                ft = jnp.concatenate([ft, residual.pop()], 3)
                ft = resnet_block(self.ch * scale, self.groups, self.dropout_rate)(ft, t_emb)
                
                if ft.shape[1] in self.add_attn:
                    ft = SelfAttention(num_groups=self.groups)(ft)
            
            if i != scale_len-1:
                B, H, W, C = ft.shape
                ft = jax.image.resize(ft, (B, 2*H, 2*W, C), "nearest")
                ft = nn.Conv(self.ch * scale, (3, 3))(ft)
            
            # print(f"Feature dimension at 'upsampling' part: {ft.shape}")
        
        assert not residual
        
        # Terminal layer
        ft = nn.silu(nn.GroupNorm(num_groups=self.groups)(ft))
        out = nn.Conv(x.shape[3], (3, 3))(ft)
        
        return out