# ---------------------------------------------------------------
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for CMD. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------


import torch
import torch.nn.functional as F
    
from einops import rearrange
import numpy as np

from eval.fvd.fvd import get_fvd_logits, frechet_distance
from tools.utils import convert_to_grid
from diffusion import create_diffusion
from torch.cuda.amp import autocast

from einops import rearrange
from typing import Any, List


def maybe_pixels_to_sd_latents(
        x: torch.Tensor,
        sd_vae: torch.nn.Module,
        scale_factor: float,
        reduce_memory: bool = True
        ) -> torch.Tensor:
    if sd_vae is None:
        return x
    with torch.no_grad():
        b = x.size(0)
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        if reduce_memory:
            chunk_size = x.size(0) // 4
            x = torch.cat(
                [sd_vae.encode(z).latent_dist.sample().mul_(scale_factor)
                 for z in x.chunk(x.size(0) // chunk_size)]
                 )
        else:
            chunk_size = x.size(0) // 8
            x = torch.cat(
                [sd_vae.encode(z).latent_dist.sample().mul_(scale_factor)
                 for z in x.chunk(x.size(0) // chunk_size)]
                 )
        x = rearrange(x, '(b t) c h w -> b c t h w', b=b)

    return x

def maybe_sd_latents_to_pixels(
        x: torch.Tensor,
        sd_vae: torch.nn.Module,
        scale_factor: float
        ) -> torch.Tensor:
    if sd_vae is None:
        return x
    return torch.cat([sd_vae.decode((z / scale_factor)).sample for z in x.chunk(x.size(0))])

def get_real_embeddings(
        test_batches: List[torch.Tensor],
        i3d: torch.nn.Module,
        device: torch.cuda.device,
        num_samples: int =  -1
        ) -> List[torch.Tensor]:
    curr_samples = 0
    real_embeddings = []
    for test_batch in test_batches:
        if num_samples > 0 and curr_samples >= num_samples:
            break
        curr_samples += test_batch["x"].size(0)
        real_embeddings.append(
        get_fvd_logits(
            ((1 + test_batch["x"]) * 127.5).round().type(torch.uint8).numpy(), 
            i3d=i3d,
            device=device)
            )
    return real_embeddings

def postprocess_videos(videos: torch.Tensor) -> torch.Tensor:
    videos = torch.cat(videos)[:16].float().clamp(-1, 1).contiguous() # b c t h w

    videos = convert_to_grid(videos.numpy()) # t ph pw c, [0, 255]
    return np.expand_dims(videos, 0).transpose(0, 1, 4, 2, 3) # b t c ph pw

def postprocess_images(images: torch.Tensor) -> torch.Tensor:
    images = torch.cat(images)[:16].float().clamp(-1, 1).contiguous()
    _, C, H, W = images.shape

    images = rearrange(images, '(ph pw) c h w -> ph pw c h w', ph=4, pw=4)
    images = ((images + 1) * 127.5).cpu().type(torch.uint8).numpy()
    images = rearrange(images, 'ph pw c h w -> (ph h) (pw w) c')
    return np.expand_dims(images, 0)


def eval_with_visualize(
        test_batches: Any,
        autoencoder: torch.nn.Module, # require
        content_pipeline: Any = None,
        motion_pipeline: Any = None, 
        device: str = "cpu",
        mode: str ='pixel', 
        sd_vae: Any = None, 
        scale_factor: Any = None, 
        i3d: Any = None,
        real_embeddings: Any = None,
        visualize: bool = True,
        sample_fn: Any = None,
        num_samples: int = -1,
        use_ddp: bool = False,
        ) -> Any:
    
    ret = dict(recons=[], keyframes=[], gen_videos=[], gen_keyframes=[], pred_videos=[], fvd=None,)
    fake_embeddings = []
    val_diffusion = create_diffusion(timestep_respacing="50", diffusion_steps=1000)
    curr_samples = 0

    if real_embeddings is None:
        real_embeddings = get_real_embeddings(test_batches, i3d, device, num_samples=num_samples)

    for test_batch in test_batches:
        if num_samples > 0 and curr_samples >= num_samples:
            break
        curr_samples += test_batch["x"].size(0)
        if content_pipeline == None and motion_pipeline == None:
            with torch.no_grad(), autocast():
                # reconstruction
                x = test_batch["x"].to(device)
                x = maybe_pixels_to_sd_latents(x, sd_vae, scale_factor)
                x_tilde, z_c = autoencoder(x)

                if 'latent' in mode:
                    x_tilde = maybe_sd_latents_to_pixels(x_tilde, sd_vae, scale_factor)
                    z_c = maybe_sd_latents_to_pixels(z_c, sd_vae, scale_factor)
                x_tilde = rearrange(x_tilde, '(b t) c h w -> b c t h w', b=x.size(0))

                if visualize and len(ret["recons"]) < 16:
                    ret["recons"].append(x_tilde.cpu())
                    ret["keyframes"].append(z_c.cpu())

                fake_embedding = get_fvd_logits(
                                   videos = ((1+x_tilde)*127.5).cpu().type(torch.uint8).numpy(), 
                                   i3d=i3d, 
                                   device=device)
                fake_embeddings.append(fake_embedding)

        elif content_pipeline == None:
            with torch.no_grad():
                # motion model
                x = test_batch["x"].to(device)
                y = test_batch["y"].to(device)

                x = (x if mode == 'pixel' \
                     else maybe_pixels_to_sd_latents(x, sd_vae, scale_factor))
                with autocast():
                    z_m, z_c = autoencoder.encode(x)
                z_m, z_c = z_m.float(), z_c.float()
                z_m = torch.randn_like(z_m, device = device)
                model_kwargs = dict(y=y, keyframe=z_c)
                z_m = val_diffusion.p_sample_loop(
                    motion_pipeline.forward,
                    z_m.shape,
                    z_m,
                    clip_denoised=False,
                    model_kwargs=model_kwargs,
                    progress=True,
                    device=device
                )
                with autocast():
                    pred_video = autoencoder.decode(z_m, z_c)
                pred_video = pred_video.float()
                if 'latent' in mode:
                    pred_video = maybe_sd_latents_to_pixels(pred_video, sd_vae, scale_factor)
                pred_video = rearrange(pred_video, '(b t) c h w -> b c t h w', b=x.size(0))    
                if visualize:
                    ret["pred_videos"].append(pred_video.cpu())
                fake_embeddings.append(
                    get_fvd_logits(
                        ((1+pred_video)*127.5).cpu().type(torch.uint8).numpy(),
                        i3d=i3d,
                        device=device
                        ))
        else:   
            with torch.no_grad():
                y = test_batch["y"].to(device)
                # Setup classifier-free guidance:
                n = test_batch["x"].size(0)
                z = torch.randn(n, 3, 64, 64).to(device)
                z = torch.cat([z, z], 0)
                y_null = torch.tensor([101] * n, device=device)
                y = torch.cat([y, y_null], 0)
                model_kwargs = dict(y=y, cfg_scale=4.0)
                z_c = sample_fn(
                    content_pipeline.forward_with_cfg,
                    (2*n, 3, 64, 64),
                    z,
                    clip_denoised=True,
                    model_kwargs=model_kwargs, progress=True, device=device
                )
                z_c, _ = z_c.chunk(2, dim=0)
                y = None

            with torch.no_grad():
                z_m = torch.randn(z_c.size(0), 8, 16, 64).to(device)
                model_kwargs = dict(y=y, keyframe=z_c)
                z_m = val_diffusion.p_sample_loop(
                    motion_pipeline.forward,
                    z_m.shape,
                    z_m,
                    clip_denoised=False,
                    model_kwargs=model_kwargs,
                    progress=True,
                    device=device
                )
                gen_video = autoencoder.decode(z_m, z_c)
                if 'latent' in mode:
                    with autocast():
                        z_c = maybe_sd_latents_to_pixels(z_c, sd_vae, scale_factor)
                        gen_video = maybe_sd_latents_to_pixels(
                            gen_video, sd_vae, scale_factor
                            ).clamp(-1, 1)
                gen_video = rearrange(gen_video, '(b t) c h w -> b c t h w', b=z_c.size(0))

                if visualize:
                    ret["gen_keyframes"].append(z_c.cpu())
                    ret["gen_videos"].append(gen_video.cpu())

                fake_embedding = get_fvd_logits(
                    videos=((1+gen_video)*127.5).cpu().type(torch.uint8).numpy(),
                    i3d=i3d,
                    device=device
                    )
                fake_embeddings.append(fake_embedding)

    # if image, output should be (b c h w) with range [-1, 1]
    # if video, output should be (b c t h w) iwth range [-1, 1]
    if visualize:
        if len(ret["recons"]) > 0:
            ret["recons"] = postprocess_videos(ret["recons"])
        if len(ret["pred_videos"]) > 0:
            ret["pred_videos"] = postprocess_videos(ret["pred_videos"])
        if len(ret["gen_videos"]) > 0:
            ret["gen_videos"] = postprocess_videos(ret["gen_videos"])
        if len(ret["keyframes"]) > 0:
            ret["keyframes"] = postprocess_images(ret["keyframes"])
        if len(ret["gen_keyframes"]) > 0:
            ret["gen_keyframes"] = postprocess_images(ret["gen_keyframes"])

    if real_embeddings != None:
        real_embeddings = torch.cat(real_embeddings)
        fake_embeddings = torch.cat(fake_embeddings)
        if use_ddp:
            num_devices = torch.cuda.device_count
            import torch.distributed as dist
            dist.barrier()
            gather_fake = [
                torch.zeros_like(fake_embeddings) for _ in range(num_devices)
                ]
            gather_real = [
                torch.zeros_like(real_embeddings) for _ in range(num_devices)
                ]
            torch.distributed.all_gather(gather_fake, fake_embeddings)
            torch.distributed.all_gather(gather_real, real_embeddings)
            fake_embeddings = torch.cat(gather_fake)
            real_embeddings = torch.cat(gather_real)
        ret["fvd"] =  frechet_distance(fake_embeddings.clone().float().detach(), 
                                       real_embeddings.clone().float().detach())
    
    print(real_embeddings.shape, fake_embeddings.shape)
    return ret
