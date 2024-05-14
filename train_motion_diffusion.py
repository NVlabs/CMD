# ---------------------------------------------------------------
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for CMD. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------


import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast
import torch.multiprocessing as mp

import argparse
import wandb

import numpy as np
from einops import rearrange

from copy import deepcopy
from glob import glob
from time import time

from models.diffusion.motion_dit import MotionDiT_models

from eval_functions import *
from tools.utils import *


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(rank, gpu, args):
    if args.dataset_name == 'webvid':
        raise NotImplementedError('Current code only support training on UCF-101.')
    """
    Trains a new DiT model.
    """
    seed = args.global_seed * dist.get_world_size() + rank
    print('start training...')
    print(f'rank: {rank}, gpu: {gpu}, local_rank: {args.local_rank}')
    print(f'seed: {seed}, world_size: {dist.get_world_size()}')
    print(f'torch.cuda.device_count(): {torch.cuda.device_count()}')
    device = torch.device("cuda")
    torch.manual_seed(seed)

    logger, checkpoint_dir, i3d = get_logger(rank, args, device, exp_name="MotionDiffusion")
    autoenc, mdiff_model, _, sd_dict = get_models(args, logger, mode="mdiff_model")
    sd_vae, text_encoder, tokenizer = None, None, None
    if "sd_vae" in sd_dict.keys():
        sd_vae, text_encoder, tokenizer = (
            sd_dict["sd_vae"].to(device), sd_dict["text_encoder"].to(device), sd_dict["tokenzier"]
            )
    autoenc = autoenc.to(device)
    mdiff_model = mdiff_model.to(device)
    ema = deepcopy(mdiff_model)  # Create an EMA of the model for use after training
    criterion, opt, scaler = get_loss_optimizer(mdiff_model, args, device, mode="mdiff_model")

    autoenc_path = f"./ckpts/{args.dataset_name}/{args.mode}/autoencoder/last.pt"
    mdiff_model_path = (
        f"./ckpts/{args.dataset_name}/{args.mode}/motion-diffusion/last.pt" if args.resume else None
    )

    load_model_checkpoints(
        opt_target="mdiff_model",
        opt=opt,
        autoenc=autoenc,    
        autoenc_path=autoenc_path,
        mdiff_model=mdiff_model,
        mdiff_model_ema=ema,
        mdiff_model_path=mdiff_model_path,
        )

    requires_grad(ema, False)
    dist.barrier()
    ema = ema.to(device)
    mdiff_model = DDP(mdiff_model, device_ids=[gpu])
    sampler, loader = get_dataloader(rank, args, logger)

    #############################################################################

    # Prepare models for training:
    if not args.resume:
        update_ema(ema, mdiff_model.module, decay=0)  # Ensure EMA is initialized with synced weights
    mdiff_model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps, log_steps, running_loss = 0, 0, 0
    eval_num_videos, test_batch_size = 128, 16

    start_time = time()
    test_batches = []
    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for train_batch in loader:
            x = train_batch["x"]
            y = train_batch["y"]
            b = x.size(0)
            if rank == 0 and len(test_batches) * b < eval_num_videos:
                test_batches.append(dict(x=x, y=y))
            x = x.to(device)
            y = y.to(device)

            with autocast():
                if 'latent' in args.mode:
                    x = maybe_pixels_to_sd_latents(
                        x, sd_vae, args.scale_factor, reduce_memory=False
                        )
                with torch.no_grad():
                    if text_encoder is not None:
                        y = text_encoder(y).last_hidden_state.mean(dim=1)
                    z_m, z_c = autoenc.encode(x)

                t = torch.randint(
                    0, criterion.num_timesteps, (z_m.shape[0],), device=device
                    )
                model_kwargs = dict(y=y, keyframe=z_c)
                loss_dict = criterion.training_losses(
                    mdiff_model, z_m, t, model_kwargs=model_kwargs
                    )
                loss = loss_dict["loss"].mean()

            opt.zero_grad()
            scaler.scale(loss).backward()

            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(mdiff_model.parameters(), 1.)

            scaler.step(opt)
            scaler.update()

            update_ema(ema, mdiff_model.module)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1

            if rank == 0 and len(test_batches) * b == eval_num_videos:
                print("Saving real test video")
                test_vids = torch.cat([test_batch["x"] for test_batch in test_batches])
                real_vid = np.expand_dims(
                    convert_to_grid(test_vids[:16].numpy()), 0
                    ).transpose(0, 1, 4, 2, 3)
                wandb.log({"real": wandb.Video(real_vid, fps=4, format="mp4")})
                test_batches_x = torch.chunk(
                    torch.concat([test_batch['x'] for test_batch in test_batches]),
                    eval_num_videos // test_batch_size
                    )
                test_batches_y = torch.chunk(
                    torch.concat([test_batch['y'] for test_batch in test_batches]),
                    eval_num_videos // test_batch_size
                    )
                test_batches = [
                    dict(x=test_batch_x, y=test_batch_y) 
                    for test_batch_x, test_batch_y in zip(test_batches_x, test_batches_y)
                    ]
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

            # Save DiT checkpoint:
            if rank == 0 and train_steps % args.ckpt_every == 0:
                checkpoint = {
                    "mdiff_model": mdiff_model.module.state_dict(),
                    "ema": ema.state_dict(),
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
                    motion_pipeline=ema,
                    device=device,
                    mode=args.mode,
                    sd_vae=sd_vae if 'latent' == args.mode else None,
                    scale_factor=args.scale_factor if 'latent' == args.mode else None,
                    i3d=i3d,
                    real_embeddings=None,
                    )
                wandb.log(
                    {"pred_videos": wandb.Video(eval_results["pred_videos"], fps=4, format="mp4")}
                    )

                wandb.log({"fvd": eval_results["fvd"]})


    mdiff_model.eval()
    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, choices=["UCF101", "webvid"], default="UCF101")
    parser.add_argument("--mode", type=str, choices=["pixel"], default="pixel")

    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--motion-model-config", type=str, choices=list(MotionDiT_models.keys()), default="DiT-B/2")
    parser.add_argument("--keyframe-patch-size", type=int, default=4)
    parser.add_argument("--use-text-embed", action='store_true')
    parser.add_argument("--image-size", nargs='+', type=int, default=[64, 64])
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
한과영0217

    # AE options
    parser.add_argument("--embed-dim", type=int, default=8)
    parser.add_argument("--n-frames", type=int, default=16)
    parser.add_argument("--scale-factor", type=float, default=0.18215)

    # ddp
    parser.add_argument('--nnodes', type=int, default=1,
                        help='The number of nodes in multi node env.')
    parser.add_argument('--nproc_per_node', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--node_rank', type=int, default=0,
                        help='The index of node.')
    parser.add_argument('--local-rank', type=int, default=0,
                        help='rank of process in the node')
    parser.add_argument('--master_addr', type=str, default='localhost',
                        help='address for master')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-4)


    args = parser.parse_args()
    processes = []
    args.world_size = args.nnodes * args.nproc_per_node
    size = args.nproc_per_node

    mp.set_start_method('spawn', force=True)

    if size > 1:
        for rank in range(size):
            args.local_rank = rank
            global_rank = rank + args.node_rank * args.nproc_per_node
            global_size = args.nnodes * args.nproc_per_node
            args.global_rank = global_rank
            print('Node rank %d, local proc %d, global proc %d' % (args.node_rank, rank, global_rank))
            p = mp.Process(target=init_processes, args=(global_rank, global_size, main, args))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        print('starting in debug mode')

        init_processes(0, size, main, args)
