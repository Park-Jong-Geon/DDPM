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
        half_dim = self.dim // 2
        w = jnp.power(10000., - jnp.arange(half_dim) / (half_dim - 1))
        arg = t[:, None] * w[None, :]
        t_emb = jnp.concatenate([jnp.sin(arg), jnp.cos(arg)], 1)
        return t_emb
    
# ResNet-style block at each resolution
class resnet_block(nn.Module):
    ch: int
    groups: int
    dropout_rate: float
    
    @nn.compact
    def __call__(self, x, t_emb):
        x = nn.silu(nn.GroupNorm(num_groups=self.groups)(x))
        x = nn.Conv(self.ch, (3, 3), kernel_init=init.kaiming_normal(), bias_init=init.zeros_init())(x)
        
        t_emb = nn.Dense(self.ch)(nn.silu(t_emb))
        x = x + jnp.expand_dims(t_emb, (1, 2))

        x = nn.silu(nn.GroupNorm(num_groups=self.groups)(x))
        x = nn.Dropout(rate=self.dropout_rate, deterministic=True)(x)
        x = nn.Conv(self.ch, (3, 3), kernel_init=init.kaiming_normal(), bias_init=init.zeros_init())(x)
        
        res = nn.Conv(self.ch, (1, 1))(x)
        
        assert(x.shape == res.shape)
        return x + res

# U-Net
class UNet(nn.Module):
    ch: int
    groups: int
    scale: tuple
    add_attn: tuple
    dropout_rate: float
    num_heads: int
    
    @nn.compact
    def __call__(self, x, t):        
        t_emb = sin_embedding(self.ch)(t)
        
        # Why are the following two lines necessary?
        t_emb = nn.silu(nn.Dense(4 * self.ch, kernel_init=init.kaiming_normal(), bias_init=init.zeros_init())(t_emb))
        t_emb = nn.Dense(4 * self.ch, kernel_init=init.kaiming_normal(), bias_init=init.zeros_init())(t_emb)
        
        # Initial layer
        ft = nn.Conv(self.ch, (3, 3), kernel_init=init.kaiming_normal(), bias_init=init.zeros_init())(x)
        
        # Downsampling
        scale_len = len(self.scale)
        residual = []
        for i, scale in enumerate(self.scale):
            ft = resnet_block(self.ch * scale, self.groups, self.dropout_rate)(ft, t_emb)
            
            '''
            if i+1 in self.add_attn:
                attn = nn.GroupNorm(num_groups=self.groups)(ft)
                attn = nn.SelfAttention(num_heads=self.num_heads, kernel_init=init.kaiming_normal(), bias_init=init.zeros_init())(attn, deterministic=True)
                assert ft.shape == attn.shape
                #ft += nn.GroupNorm(num_groups=self.groups)(attn)
                ft += attn
            '''

            attn = nn.GroupNorm(num_groups=self.groups)(ft)
            attn = nn.SelfAttention(num_heads=self.num_heads, kernel_init=init.kaiming_normal(), bias_init=init.zeros_init())(attn, deterministic=True)
            assert ft.shape == attn.shape
            ft += attn

            residual.append(ft)
            
            if i != scale_len-1:
                #ft = nn.avg_pool(ft, (2, 2), (2, 2))
                ft = nn.Conv(self.ch * scale, (3, 3), 2, (1, 1), kernel_init=init.kaiming_normal(), bias_init=init.zeros_init())(ft)
            
            # print(f"Feature dimension at 'downsampling' part: {ft.shape}")
                
        # Middle
        ft = resnet_block(self.ch * scale, self.groups, self.dropout_rate)(ft, t_emb)
        attn = nn.GroupNorm(num_groups=self.groups)(ft)
        attn = nn.SelfAttention(num_heads=self.num_heads, kernel_init=init.kaiming_normal(), bias_init=init.zeros_init())(attn, deterministic=True)
        assert ft.shape == attn.shape
        #ft += nn.GroupNorm(num_groups=self.groups)(attn)
        ft += attn
        ft = resnet_block(self.ch * scale, self.groups, self.dropout_rate)(ft, t_emb)
        # print(f"Feature dimension at 'middle' part: {ft.shape}")
        
        # Upsampling
        for i, scale in enumerate(reversed(self.scale)):
            assert residual[-1].shape[0:3] == ft.shape[0:3]
            ft = jnp.concatenate([residual.pop(), ft], 3)
            ft = resnet_block(self.ch * scale, self.groups, self.dropout_rate)(ft, t_emb)
            '''
            if scale_len-i in self.add_attn:
                attn = nn.SelfAttention(num_heads=self.num_heads, kernel_init=init.kaiming_normal(), bias_init=init.zeros_init())(ft, deterministic=True)
                assert ft.shape == attn.shape
                #ft += nn.GroupNorm(num_groups=self.groups)(attn)
                ft += attn
            '''

            attn = nn.GroupNorm(num_groups=self.groups)(ft)
            attn = nn.SelfAttention(num_heads=self.num_heads, kernel_init=init.kaiming_normal(), bias_init=init.zeros_init())(attn, deterministic=True)
            assert ft.shape == attn.shape
            ft += attn

            if i != scale_len-1:
                B, H, W, C = ft.shape
                ft = jax.image.resize(ft, (B, 2*H, 2*W, C), "nearest")
                ft = nn.Conv(self.ch * scale, (3, 3), kernel_init=init.kaiming_normal(), bias_init=init.zeros_init())(ft)
            
            # print(f"Feature dimension at 'upsampling' part: {ft.shape}")
        
        assert not residual
        
        # Terminal layer
        ft = nn.silu(nn.GroupNorm(num_groups=self.groups)(ft))
        out = nn.Conv(x.shape[3], (3, 3), kernel_init=init.kaiming_normal(), bias_init=init.zeros_init())(ft)
        
        return out