# ---------------------------------------------------------------
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# This file has been modified from DiT.
#
# Source:
#  https://github.com/facebookresearch/DiT/blob/main/models.py
#
# The license for the original version of this file can be
# found in https://github.com/NVlabs/CMD/blob/main/third_party_licenses/LICENSE_DIT
# The modifications to this file are subject to the same license.
# ---------------------------------------------------------------


import torch
import torch.nn as nn
import numpy as np
from timm.models.vision_transformer import PatchEmbed
from models.diffusion.content_dit import (
    TimestepEmbedder,
    DiTBlock,
    FinalLayer,
    get_2d_sincos_pos_embed,
)

if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
    ATTENTION_MODE = 'flash'
else:
    try:
        import xformers
        import xformers.ops
        ATTENTION_MODE = 'xformers'
    except:
        ATTENTION_MODE = 'math'
#print(f'attention mode is {ATTENTION_MODE}')

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#                                 Core DiT Model                                #
#################################################################################å
class MotionDiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size: tuple = (16, 64),
        patch_size: int = 2,
        in_channels: int = 8,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        learn_sigma: bool = True,
        keyframe_size: int =  64,
        keyframe_channel: int = 3,
        keyframe_patch_size: int = 4,
    ):
        super().__init__()

        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.input_size = input_size
        self.x_embedder = PatchEmbed(
            input_size, (patch_size, patch_size), in_channels, hidden_size, bias=True
            )
        self.keyframe_embedder = PatchEmbed(
            keyframe_size, keyframe_patch_size, keyframe_channel, hidden_size, bias=True
            )
        self.keyframe_size = (keyframe_size, keyframe_size)
        self.keyframe_patch_size = keyframe_patch_size
        self.t_embedder = TimestepEmbedder(hidden_size)
        num_patches = self.x_embedder.num_patches
        keyframe_num_patches = self.keyframe_embedder.num_patches

        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, hidden_size), requires_grad=False
            )
        self.keyframe_pos_embed = nn.Parameter(
            torch.zeros(1, keyframe_num_patches, hidden_size), requires_grad=False
            )
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5)
            )
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0)
            )
        keyframe_pos_embed = get_2d_sincos_pos_embed(
            self.keyframe_pos_embed.shape[-1], int(self.keyframe_embedder.num_patches ** 0.5)
            )
        self.keyframe_pos_embed.data.copy_(
            torch.from_numpy(keyframe_pos_embed).float().unsqueeze(0)
            )

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.keyframe_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.keyframe_embedder.proj.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = self.input_size[0] // p
        w = self.input_size[1] // p
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def forward(self, x, t, keyframe, y):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        keyframe = self.keyframe_embedder(keyframe) + self.keyframe_pos_embed
        num_patches = x.size(1)

        t = self.t_embedder(t)                   # (N, D)
        c = t

        x = torch.cat([x, keyframe], dim=1)
        for block in self.blocks:
            x = block(x, c)                      # (N, T, D)
        x = self.final_layer(x[:, :num_patches], c)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)

        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def MotionDiT_XL_2(**kwargs):
    return MotionDiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def MotionDiT_XL_4(**kwargs):
    return MotionDiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def MotionDiT_XL_8(**kwargs):
    return MotionDiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def MotionDiT_L_2(**kwargs):
    return MotionDiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def MotionDiT_L_4(**kwargs):
    return MotionDiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def MotionDiT_L_8(**kwargs):
    return MotionDiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def MotionDiT_B_2(**kwargs):
    return MotionDiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def MotionDiT_B_4(**kwargs):
    return MotionDiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def MotionDiT_B_8(**kwargs):
    return MotionDiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def MotionDiT_S_2(**kwargs):
    return MotionDiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def MotionDiT_S_4(**kwargs):
    return MotionDiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def MotionDiT_S_8(**kwargs):
    return MotionDiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


MotionDiT_models = {
    'MotionDiT-XL/2': MotionDiT_XL_2,  'MotionDiT-XL/4': MotionDiT_XL_4,  'MotionDiT-XL/8': MotionDiT_XL_8,
    'MotionDiT-L/2':  MotionDiT_L_2,   'MotionDiT-L/4':  MotionDiT_L_4,   'MotionDiT-L/8':  MotionDiT_L_8,
    'MotionDiT-B/2':  MotionDiT_B_2,   'MotionDiT-B/4':  MotionDiT_B_4,   'MotionDiT-B/8':  MotionDiT_B_8,
    'MotionDiT-S/2':  MotionDiT_S_2,   'MotionDiT-S/4':  MotionDiT_S_4,   'MotionDiT-S/8':  MotionDiT_S_8,
}