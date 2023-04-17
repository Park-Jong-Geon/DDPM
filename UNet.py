import jax
import jax.numpy as jnp
import flax.linen as nn
import flax.linen.initializers as init
from typing import Callable

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
    def __call__(self, x, t_emb, training):
        ft = nn.swish(nn.GroupNorm(num_groups=self.groups)(x))
        ft = nn.Conv(self.ch, (3, 3))(ft)
        
        t_emb = nn.Dense(self.ch)(nn.swish(t_emb))
        ft += t_emb[:, None, None, :]

        ft = nn.swish(nn.GroupNorm(num_groups=self.groups)(ft))
        ft = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(ft)
        ft = nn.Conv(self.ch, (3, 3))(ft)
        
        if x.shape[-1] != self.ch:
            x = nn.Conv(self.ch, (1, 1))(x)
        
        assert ft.shape == x.shape
        return ft + x

# NiN layer used for SelfAttention layer
class nin(nn.Module):
    ch: int
    kernel_init: Callable = nn.initializers.xavier_normal()
    bias_init: Callable = nn.initializers.zeros_init()
    
    @nn.compact
    def __call__(self, x):
        W = self.param('kernel', self.kernel_init, (x.shape[-1], self.ch))
        b = self.param('bias', self.bias_init, (self.ch,))
        
        return jnp.tensordot(x, W, 1) + b

# Attention block
class SelfAttention(nn.Module):
    num_groups: int
    
    @nn.compact
    def __call__(self, x):
        B, H, W, C = x.shape
        h = nn.GroupNorm(num_groups=self.num_groups)(x)
        
        q = nin(C)(h)
        k = nin(C)(h)
        v = nin(C)(h)
        
        w = jnp.einsum('bhwc,bHWc->bhwHW', q, k) * (C ** (-0.5))
        w = jnp.reshape(w, [B, H, W, H*W])
        w = nn.softmax(w, axis=-1)
        w = jnp.reshape(w, [B, H, W, H, W])

        h = jnp.einsum('bhwHW,bHWc->bhwc', w, v)
        
        h = nin(C)(h)
        
        assert x.shape == h.shape
        return x + h

# U-Net
class UNet(nn.Module):
    ch: int
    groups: int
    scale: tuple
    add_attn: tuple
    dropout_rate: float
    num_res_blocks : int
    
    @nn.compact
    def __call__(self, x, t, training):       
        t_emb = sin_embedding(self.ch)(t)
        t_emb = nn.swish(nn.Dense(4 * self.ch)(t_emb))
        t_emb = nn.Dense(4 * self.ch)(t_emb)
        
        # Initial layer
        ft = nn.Conv(self.ch, (3, 3))(x)
        
        # Downsampling
        assert ft.shape[1] == ft.shape[2]
        scale_len = len(self.scale)
        residual = [ft]
        
        for i, scale in enumerate(self.scale):
            for j in range(self.num_res_blocks):
                ft = resnet_block(self.ch * scale, self.groups, self.dropout_rate)(ft, t_emb, training)
                
                if ft.shape[1] in self.add_attn:
                    ft = SelfAttention(num_groups=self.groups)(ft)
                
                residual.append(ft)
            
            if i < scale_len-1:
                ft = nn.Conv(self.ch * scale, (3, 3), 2, (1, 1))(ft)
                residual.append(ft)
            # print(f"Feature dimension at 'downsampling' part: {ft.shape}")
                
        # Middle
        assert scale == self.scale[-1]
        ft = resnet_block(self.ch * scale, self.groups, self.dropout_rate)(ft, t_emb, training)
        ft = SelfAttention(num_groups=self.groups)(ft)
        ft = resnet_block(self.ch * scale, self.groups, self.dropout_rate)(ft, t_emb, training)
        # print(f"Feature dimension at 'middle' part: {ft.shape}")
        
        # Upsampling
        for i, scale in enumerate(reversed(self.scale)):
            for j in range(self.num_res_blocks + 1):
                assert residual[-1].shape[0:3] == ft.shape[0:3]
                ft = jnp.concatenate([ft, residual.pop()], 3)
                ft = resnet_block(self.ch * scale, self.groups, self.dropout_rate)(ft, t_emb, training)
                
                if ft.shape[1] in self.add_attn:
                    ft = SelfAttention(num_groups=self.groups)(ft)
            
            if i < scale_len-1:
                B, H, W, C = ft.shape
                ft = jax.image.resize(ft, (B, 2*H, 2*W, C), "nearest")
                ft = nn.Conv(C, (3, 3))(ft)
            # print(f"Feature dimension at 'upsampling' part: {ft.shape}")
        
        assert not residual
        
        # Terminal layer
        ft = nn.swish(nn.GroupNorm(num_groups=self.groups)(ft))
        out = nn.Conv(x.shape[-1], (3, 3))(ft)
        
        return out