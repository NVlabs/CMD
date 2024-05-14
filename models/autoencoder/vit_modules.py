# ---------------------------------------------------------------
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# This file has been modified from PVDM.
#
# Source:
#  https://github.com/sihyun-yu/PVDM/blob/main/models/autoencoder/vit_modules.py
#
# The license for the original version of this file can be
# found in https://github.com/NVlabs/CMD/blob/main/third_party_licenses/LICENSE_PVDM
# The modifications to this file are subject to the same license.
# ---------------------------------------------------------------
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from math import log, pi

from timm.models.vision_transformer import Attention
from typing import Tuple

if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
    ATTENTION_MODE = 'flash'
else:
    try:
        import xformers
        import xformers.ops
        ATTENTION_MODE = 'xformers'
    except:
        ATTENTION_MODE = 'math'


def rotate_every_two(x: torch.Tensor) -> torch.Tensor:
    x = rearrange(x, '... (d j) -> ... d j', j = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d j -> ... (d j)')

def apply_rot_emb(
        q: torch.Tensor, k: torch.Tensor, rot_emb: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
    sin, cos = rot_emb
    rot_dim = sin.shape[-1]
    (q, q_pass), (k, k_pass) = map(lambda t: (t[..., :rot_dim], t[..., rot_dim:]), (q, k))
    q, k = map(lambda t: t * cos + rotate_every_two(t) * sin, (q, k))
    q, k = map(lambda t: torch.cat(t, dim = -1), ((q, q_pass), (k, k_pass)))
    return q, k

class AxialRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_freq: int = 10):
        super().__init__()
        self.dim = dim
        scales = torch.logspace(0., log(max_freq / 2) / log(2), self.dim // 4, base = 2)
        self.register_buffer('scales', scales)

    def forward(self, h: torch.Tensor, w: torch.Tensor, device: torch.cuda.device):
        scales = rearrange(self.scales, '... -> () ...')
        scales = scales.to(device)

        h_seq = torch.linspace(-1., 1., steps = h, device = device)
        h_seq = h_seq.unsqueeze(-1)

        w_seq = torch.linspace(-1., 1., steps = w, device = device)
        w_seq = w_seq.unsqueeze(-1)

        h_seq = h_seq * scales * pi
        w_seq = w_seq * scales * pi

        x_sinu = repeat(h_seq, 'i d -> i j d', j = w)
        y_sinu = repeat(w_seq, 'j d -> i j d', i = h)

        sin = torch.cat((x_sinu.sin(), y_sinu.sin()), dim = -1)
        cos = torch.cat((x_sinu.cos(), y_sinu.cos()), dim = -1)

        sin, cos = map(lambda t: rearrange(t, 'i j d -> (i j) d'), (sin, cos))
        sin, cos = map(lambda t: repeat(t, 'n d -> () n (d j)', j = 2), (sin, cos))
        return sin, cos

class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freqs = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freqs', inv_freqs)

    def forward(self, n, device):
        seq = torch.arange(n, device = device)
        freqs = einsum('i, j -> i j', seq, self.inv_freqs)
        freqs = torch.cat((freqs, freqs), dim = -1)
        freqs = rearrange(freqs, 'n d -> () n d')
        return freqs.sin(), freqs.cos()

def exists(val):
    return val is not None

def shift(t, amt):
    if amt == 0:
        return t
    return F.pad(t, (0, 0, 0, 0, amt, -amt))

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

class TimeSformerEncoder(nn.Module):
    def __init__(
        self,
        *,
        dim: int = 512,
        num_frames: int = 16,
        image_size: Tuple[int, int] = (128, 128),
        patch_size: int = 8,
        channels: int = 3,
        depth: int = 8,
        heads: int = 8,
        dim_head: int = 64,
        ff_dropout: int = 0.,
        rotary_emb: bool = True,
    ):
        super().__init__()
        assert image_size[0] % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        assert image_size[1] % patch_size == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_size[0] // patch_size) * (image_size[1] // patch_size)
        num_positions = num_frames * num_patches
        patch_dim = channels * patch_size ** 2

        self.heads = heads
        self.patch_size = patch_size
        self.to_patch_embedding = nn.Linear(patch_dim, dim)

        self.use_rotary_emb = rotary_emb
        if rotary_emb:
            self.frame_rot_emb = RotaryEmbedding(dim_head)
            self.image_rot_emb = AxialRotaryEmbedding(dim_head)
        else:
            self.pos_emb = nn.Embedding(num_positions, dim)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            ff = FeedForward(dim, dropout = ff_dropout)
            time_attn = Attention(dim, num_heads=heads, qkv_bias=True)
            spatial_attn = Attention(dim, num_heads=heads, qkv_bias=True)
            time_attn, spatial_attn, ff = map(
                lambda t: PreNorm(dim, t), (time_attn, spatial_attn, ff)
                )
            self.layers.append(nn.ModuleList([time_attn, spatial_attn, ff]))

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        _, f, _, h, w, *_, device, p = *video.shape, video.device, self.patch_size
        assert (h % p == 0 and w % p == 0), f'height {h} and width {w} of video must be divisible by the patch size {p}'

        # calculate num patches in height and width dimension, and number of total patches (n)
        hp, wp = (h // p), (w // p)
        n = hp * wp

        # video to patch embeddings
        video = rearrange(video, 'b f c (h p1) (w p2) -> b (f h w) (p1 p2 c)', p1 = p, p2 = p)
        x = self.to_patch_embedding(video)

        if not self.use_rotary_emb:
            x += self.pos_emb(torch.arange(x.shape[1], device = device))

        # time and space attention
        for (time_attn, spatial_attn, ff) in self.layers:
            x = rearrange(x, 'b (f n) d -> (b n) f d', f=f)
            x = time_attn(x) + x
            x = rearrange(x, '(b n) f d -> (b f) n d', n=n)
            x = spatial_attn(x) + x
            x = rearrange(x, '(b f) n d -> b (f n) d', f=f)
            x = ff(x) + x

        return x

class TimeSformerDecoder(nn.Module):
    def __init__(
        self,
        *,
        dim: int = 512,
        num_frames: int = 16,
        image_size: Tuple[int, int] = (128, 128),
        patch_size: int = 8,
        channels: int = 3,
        depth: int = 8,
        heads: int = 8,
        dim_head: int = 64,
        ff_dropout: int = 0.,
        rotary_emb: bool = True,
    ):
        super().__init__()
        assert image_size[0] % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        assert image_size[1] % patch_size == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_size[0] // patch_size) * (image_size[1] // patch_size)
        num_positions = num_frames * num_patches

        self.heads = heads
        self.patch_size = patch_size

        self.use_rotary_emb = rotary_emb
        if rotary_emb:
            self.frame_rot_emb = RotaryEmbedding(dim_head)
            self.image_rot_emb = AxialRotaryEmbedding(dim_head)
        else:
            self.pos_emb = nn.Embedding(num_positions, dim)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            ff = FeedForward(dim, dropout = ff_dropout)
            time_attn = Attention(dim, num_heads=heads, qkv_bias=True)
            spatial_attn = Attention(dim, num_heads=heads, qkv_bias=True)
            time_attn, spatial_attn, ff = map(
                lambda t: PreNorm(dim, t), (time_attn, spatial_attn, ff)
                )
            self.layers.append(nn.ModuleList([time_attn, spatial_attn, ff]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        f, hp, wp = x.size(2), x.size(3), x.size(4)
        n = hp * wp
        x = rearrange(x, 'b c f h w -> b (f h w) c')

        if not self.use_rotary_emb:
            x += self.pos_emb(torch.arange(x.shape[1], device = device))

        # time and space attention
        for (time_attn, spatial_attn, ff) in self.layers:
            x = rearrange(x, 'b (f n) d -> (b n) f d', f=f)
            x = time_attn(x) + x
            x = rearrange(x, '(b n) f d -> (b f) n d', n=n)
            x = spatial_attn(x) + x
            x = rearrange(x, '(b f) n d -> b (f n) d', f=f)
            x = ff(x) + x

        return x
