# ---------------------------------------------------------------
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for CMD. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------


import argparse
from time import time

import torch
from torch.cuda.amp import autocast
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from einops import rearrange
import numpy as np
import wandb

from eval_functions import *
from tools.utils import *

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = True


def main(args):
    assert (torch.cuda.is_available(),
            "Training currently requires at least one GPU.")
    if args.dataset_name == 'webvid':
        raise NotImplementedError('Current code only support training on UCF-101.')
    # Setup DDP:
    dist.init_process_group("nccl")
    assert (args.global_batch_size % dist.get_world_size() == 0,
            f"Batch size must be divisible by world size.")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # load model
    sd_vae = None
    logger, checkpoint_dir, i3d = get_logger(rank, args, device)
    autoenc, _, _, sd_dict = get_models(args, logger)
    autoenc = autoenc.to(device)
    if "sd_vae" in sd_dict.keys():
        sd_vae = sd_dict["sd_vae"].to(device)
    criterion, opt, scaler = get_loss_optimizer(autoenc, args, device, mode="autoenc")
    autoenc_path = (
        f"./ckpts/{args.dataset_name}/{args.mode}/autoencoder/last.pt" if args.resume else None
    )
    load_model_checkpoints(
        opt_target='autoenc', opt=opt, autoenc=autoenc, autoenc_path=autoenc_path
        )
    autoenc = DDP(autoenc, device_ids=[rank])
    sampler, loader = get_dataloader(rank, args, logger)

    # Variables for monitoring/logging purposes:
    train_steps, log_steps, running_loss = 0, 0, 0
    eval_num_videos, test_batch_size = 2048, 16
    start_time = time()
    test_batches = []
    logger.info(f"Training for {args.epochs} epochs...")

    # training
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for train_batch in loader:
            x = train_batch["x"].to(device)
            b = x.size(0)
            if rank == 0 and len(test_batches) * b < eval_num_videos:
                test_batches.append(dict(x=x.cpu()))
            with autocast():
                x = maybe_pixels_to_sd_latents(x, sd_vae, args.scale_factor)
                x_recon, _ = autoenc(x)
                x_recon = rearrange(x_recon, '(b t) c h w -> b c t h w', b=b)
                loss = (
                    criterion(x_recon, x) if args.mode == 'latent'
                    else criterion(
                        x, x_recon, optimizer_idx=0, global_step=train_steps,
                        )
                    )
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if rank == 0 and len(test_batches) * b == eval_num_videos:
                print("Saving real test video")
                test_vids = torch.cat(
                    [test_batch["x"] for test_batch in test_batches]
                    )
                real_vid = np.expand_dims(
                    convert_to_grid(test_vids[:16].numpy()), 0
                    ).transpose(0, 1, 4, 2, 3)
                wandb.log({"real": wandb.Video(real_vid, fps=4, format="mp4")})
                test_vids = torch.chunk(
                    test_vids, eval_num_videos // test_batch_size
                    )
                test_batches = [dict(x=test_vid) for test_vid in test_vids]
                print("completed.")

            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(
                    f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}"
                )
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save checkpoint:
            if train_steps % args.ckpt_every == 0 and rank == 0:
                checkpoint = {
                    "autoenc": autoenc.module.state_dict(),
                    "opt": opt.state_dict(),
                    "args": args
                }
                checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
                eval_results = eval_with_visualize(
                    test_batches=test_batches,
                    autoencoder=autoenc,
                    content_pipeline=None,
                    motion_pipeline=None,
                    device=device,
                    mode=args.mode,
                    sd_vae=sd_vae,
                    scale_factor=args.scale_factor,
                    i3d=i3d,
                    real_embeddings=None,
                    visualize=True,
                    num_samples=-1,
                )
                wandb.log(
                    {"recon": wandb.Video(eval_results["recons"], fps=4, format="mp4"),
                     "frames": wandb.Image(eval_results["keyframes"]),
                     "fvd": eval_results["fvd"]}
                     )
    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, choices=["UCF101", "webvid"], default="UCF101")
    parser.add_argument("--mode", type=str, choices=["pixel", "latent"], default="pixel")
    parser.add_argument("--scale-factor", type=float, default=0.18215)
    parser.add_argument("--results-dir", type=str, default="./result")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50000)
    parser.add_argument("--embed-dim", type=int, default=8)
    parser.add_argument("--n-frames", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--image-size", nargs='+', type=int, default=[64, 64])

    args = parser.parse_args()
    main(args)
