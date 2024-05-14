# ---------------------------------------------------------------
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for CMD. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------


from glob import glob
import os
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb
from eval.fvd.download import load_i3d_pretrained

import numpy as np
import logging

from collections import OrderedDict
from models.diffusion.motion_dit import MotionDiT_models
from models.autoencoder.triplane_ae import ProjectedAutoencoder
from diffusers.pipelines import DiffusionPipeline
from models.diffusion.content_dit import ContentDiT_models

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from reconstruction.lossperceptual import LPIPSWithDiscriminator
from tools.dataset import get_dataset
from torch.cuda.amp import GradScaler
from diffusion import create_diffusion

#################################################################################
#                             Training Helper Functions                         #
#################################################################################

def init_processes(rank, size, fn, args):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = '6020'
    print(f'MASTER_ADDR = {os.environ["MASTER_ADDR"]}')
    print(f'MASTER_PORT = {os.environ["MASTER_PORT"]}')
    torch.cuda.set_device(args.local_rank)
    gpu = args.local_rank
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=size)
    fn(rank, gpu, args)
    dist.barrier()
    cleanup()

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid
        # small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()

def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger

def convert_to_grid(vid, drange=[-1,1], grid_size=(4,4), normalize=True):
    if normalize:
        lo, hi = drange
        vid = np.asarray(vid, dtype=np.float32)
        vid = (vid - lo) * (255 / (hi - lo))
        vid = np.rint(vid).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, T, H, W = vid.shape
    vid = vid.reshape(gh, gw, C, T, H, W)
    vid = vid.transpose(3, 0, 4, 1, 5, 2)
    vid = vid.reshape(T, gh * H, gw * W, C)

    return vid

def get_logger(rank, args, device, exp_name="ProjectedAutoencoder"):
    # Setup an experiment folder:
    if rank == 0:
        # Make results folder (holds all experiment subfolders)
        os.makedirs(args.results_dir, exist_ok=True)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = exp_name

        # Create an experiment folder
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"

        # Stores saved model checkpoints
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment dassertirectory created at {experiment_dir}")
        wandb.init(project=model_string_name)
        i3d = load_i3d_pretrained().to(device)
    else:
        logger = create_logger(None)
        i3d, checkpoint_dir = None, None

    return logger, checkpoint_dir, i3d

def get_models(args, logger, mode='autoenc'):
    dist.barrier()
    autoenc, mdiff_model, cdiff_model, latent_models_dict = None, None, None, dict()

    # Autoencoder
    # TODO(sihyun): adjust hyperparameters
    input_size = (
        args.image_size if args.mode == 'pixel' 
        else (args.image_size[0] // 8, args.image_size[0] // 8)
        )
    autoenc = ProjectedAutoencoder(
        dim=384,
        image_size=input_size,
        patch_size=2,
        channels=4 if 'latent' == args.mode else 3,
        n_frames=args.n_frames,
        embed_dim=args.embed_dim,
        depth=8,
        mode=args.mode,
    )
    logger.info(
        f"AE Parameters: {sum(p.numel() for p in autoenc.parameters()):,}"
        )

    if mode != 'autoenc':
        mdiff_model = MotionDiT_models[args.motion_model_config](
            input_size=(16, input_size[0] + input_size[1]),
            keyframe_channel=4 if 'latent' == args.mode else 3,
            keyframe_patch_size=args.keyframe_patch_size,
            )
        logger.info(
            f"Motion diffusion Parameters: {sum(p.numel() for p in mdiff_model.parameters()):,}"
            )
    if mode == 'cdiff_model':
        cdiff_model = ContentDiT_models['ContentDiT-L/2'](
                        input_size=input_size,
                        num_classes=args.num_classes,
                        )
        logger.info(
            f"Content Model Trainable Parameters: {sum(p.numel() for p in cdiff_model.parameters()):,}"
            )
    if args.mode == 'latent':
        pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1",
            variant="fp16",
            torch_dtype=torch.float16,
        )
        sd_vae = pipe.vae
        sd_vae.eval()
        logger.info(
            f"SD VAE Parameters: {sum(p.numel() for p in sd_vae.parameters()):,}"
            )
        tokenizer = pipe.tokenizer
        text_encoder = pipe.text_encoder
        latent_models_dict = {
            "sd_vae": sd_vae, "tokenizer": tokenizer, "text_encoder": text_encoder
            }
    dist.barrier()
    return autoenc, mdiff_model, cdiff_model, latent_models_dict


def get_loss_optimizer(model, args, device, mode):
    # Optimization related
    if mode == "autoenc":
        criterion = (
            torch.nn.MSELoss() if args.mode == 'latent'
            else LPIPSWithDiscriminator(disc_start=10000000).to(device)
            )
        opt = torch.optim.AdamW(
            model.parameters(), lr=args.lr, betas=(0.5, 0.9), weight_decay=0
            )
    elif mode == 'mdiff_model':
        criterion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
    elif mode == 'cdiff_model':
        criterion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
    dist.barrier()

    scaler = GradScaler()

    return criterion, opt, scaler


def get_dataloader(rank, args, logger):
    # Setup data:
    kwargs = {
        'root': args.data_path,
        'resolution': args.image_size,
        'n_frames': args.n_frames,
        }

    dataset = get_dataset(args.dataset_name, **kwargs)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    return sampler, loader


def load_model_checkpoints(
        opt_target,
        opt=None,
        autoenc=None,
        mdiff_model=None,
        mdiff_model_ema=None,
        cdiff_model=None,
        cdiff_model_ema=None,
        autoenc_path=None,
        mdiff_model_path=None,
        cdiff_model_path=None,
        ):

    if autoenc_path is not None:
        ckpt = torch.load(autoenc_path, map_location='cpu')
        autoenc.load_state_dict(ckpt["autoenc"])
        if opt_target == 'autoenc':
            opt.load_state_dict(ckpt["opt"])

    if mdiff_model_path is not None:
        ckpt = torch.load(mdiff_model_path, map_location='cpu')
        if mdiff_model is not None:
            mdiff_model.load_state_dict(ckpt["mdiff_model"])
            mdiff_model_ema.load_state_dict(ckpt["ema"])
        else:
            mdiff_model_ema.load_state_dict(ckpt["ema"])
        if opt_target == 'mdiff_model':
            opt.load_state_dict(ckpt["opt"])

    if cdiff_model_path is not None:
        ckpt = torch.load(cdiff_model_path, map_location='cpu')
        cdiff_model.load_state_dict(ckpt["cdiff_model"])
        cdiff_model_ema.load_state_dict(ckpt["ema"])
        if opt_target == 'cdiff_model':
            opt.load_state_dict(ckpt["opt"])


