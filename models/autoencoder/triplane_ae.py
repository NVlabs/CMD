# ---------------------------------------------------------------
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# This file has been modified from PVDM.
#
# Source:
#  https://github.com/sihyun-yu/PVDM/blob/main/models/autoencoder/autoencoder_vit.py
#
# The license for the original version of this file can be
# found in https://github.com/NVlabs/CMD/blob/main/third_party_licenses/LICENSE_PVDM
# The modifications to this file are subject to the same license.
# ---------------------------------------------------------------


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from models.autoencoder.vit_modules import TimeSformerEncoder, TimeSformerDecoder
from einops import rearrange
from einops.layers.torch import Rearrange

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


class ProjectedAutoencoder(nn.Module):
    def __init__(
            self,
            dim: int = 384,
            image_size: Tuple[int, int] = (64, 64),
            patch_size: int = 2,
            channels: int = 3,
            n_frames: int = 16,  
            embed_dim: int = 8,
            depth: int = 8,
            mode: str = 'pixel',
            ):

        super().__init__()
        self.s = n_frames
        self.patch_size = patch_size
        self.image_size = image_size
        self.embed_dim = embed_dim
        self.down = int(math.log(patch_size, 2))
        self.latent_h = self.image_size[0] // patch_size
        self.latent_w = self.image_size[1] // patch_size
        self.mode = mode
        patch_dim = channels * patch_size ** 2

        self.encoder = TimeSformerEncoder(
            dim=dim,
            image_size=image_size,
            num_frames=n_frames,
            depth=depth,
            patch_size=patch_size,
            channels=channels
            )
        self.decoder = TimeSformerDecoder(
            dim=dim,
            image_size=image_size,
            num_frames=n_frames,
            depth=depth,
            patch_size=patch_size
            )
        self.to_output = nn.Sequential(
            Rearrange(
                'b (t h w) c -> (b t) c h w', h=self.latent_h, w=self.latent_w,
                ),
            nn.ConvTranspose2d(
                dim, channels, kernel_size=(patch_size, patch_size), stride=patch_size
                ),
            )
        self.act = nn.Sigmoid()
        self.to_logit = nn.Sequential(
            Rearrange(
                'b (t h w) c -> (b t) c h w', h=self.latent_h, w=self.latent_w
                ),
            nn.ConvTranspose2d(
                dim, channels, kernel_size=(patch_size, patch_size), stride=patch_size
                ),
            )
        self.pre_xt = nn.Conv2d(dim, self.embed_dim, 1)
        self.pre_yt = nn.Conv2d(dim, self.embed_dim, 1)
        self.post_xt = nn.Conv2d(self.embed_dim, dim, 1)
        self.post_yt = nn.Conv2d(self.embed_dim, dim, 1)
        self.to_patch_embedding_xy = nn.Linear(patch_dim, dim)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: b c t h w
        b = x.size(0)
        x = rearrange(x, 'b c t h w -> b t c h w')
        h = self.encoder(x)

        # z_c
        h_xy_logit = rearrange(self.to_logit(h), '(b t) c h w -> b t c h w', b=b)
        h_xy_logit = torch.softmax(h_xy_logit, dim=1)
        z_c = rearrange(x * h_xy_logit, 'b t c h w -> b c t h w').sum(dim=2)

        # z_m
        h = rearrange(h, 'b (t h w) c -> b c t h w', t=self.s, h=self.latent_h)
        h_xt, h_yt = self.pre_xt(h.mean(dim=-2)), self.pre_yt(h.mean(dim=-1))
        z_m = torch.tanh(torch.cat([h_xt, h_yt], dim=-1))

        return z_m, z_c

    def decode(self, z_m: torch.Tensor, z_c: torch.Tensor) -> torch.Tensor:
        h_xy = rearrange(
            z_c, 'b c (h p1) (w p2) -> b h w (p1 p2 c)', p1 = self.patch_size, p2 = self.patch_size
            )
        h_xy = self.to_patch_embedding_xy(h_xy)
        h_xy = rearrange(h_xy, 'b h w c -> b c h w')
        h_xt, h_yt = z_m[:, :, :, :self.latent_h], z_m[:, :, :, self.latent_h:]
        h_xt, h_yt = self.post_xt(h_xt), self.post_yt(h_yt)

        h_xy = h_xy.unsqueeze(-3).expand(-1, -1, self.s, -1, -1)
        h_xt = h_xt.unsqueeze(-2).expand(-1, -1, -1, self.latent_h, -1)
        h_yt = h_yt.unsqueeze(-1).expand(-1, -1, -1, -1, self.latent_w)

        dec = self.decoder(h_xy + h_yt + h_xt)
        if self.mode == 'pixel':
            return 2*self.act(self.to_output(dec)).contiguous() -1
        return self.to_output(dec)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.ndim == 4:
            x = rearrange(x, '(b t) c h w -> b c t h w', t = self.s)
        z_m, z_c = self.encode(x)
        dec = self.decode(z_m, z_c)

        return dec, z_c
