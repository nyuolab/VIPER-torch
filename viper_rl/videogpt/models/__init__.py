from functools import cached_property, partial
import os.path as osp
import glob
import re
import pickle
import numpy as np
import torch
import torch.nn as nn
# from torch.nn.parallel import DistributedDataParallel as DDP

from .vqgan import VQGAN
from .videogpt import VideoGPT
from .stylegan_disc import StyleGANDisc
from .vqgan import VQGAN

def extract_iteration(filename):
    match = re.search(r"checkpoint_(\d+).pth", filename)
    return int(match.group(1)) if match else 0


def load_videogpt(path, config, ae_config, ae=None, replicate=True):
    # Load configuration
    # config = pickle.load(open(osp.join(path, 'args'), 'rb'))
    
    # Initialize the AE model if not provided
    if ae is None:
        ae = AE(config.ae_ckpt, ae_config)  # Adjust AE initialization as per your PyTorch implementation
    
    # config.ae = ae_config
    # Initialize the VideoGPT model
    gpt = VideoGPT(config, ae)

    # Load class mapping file if exists
    class_file = osp.join(path, 'class_map.pkl')
    if osp.exists(class_file):
        class_map = pickle.load(open(class_file, 'rb'))
        class_map = {k: int(v) for k, v in class_map.items()}
        
    else:
        class_map = None

    # Load model checkpoint
    model_files = glob.glob(f"{path}/checkpoints/*.pth")
    if len(model_files):
        checkpoint_path = sorted(model_files, key=extract_iteration)[-1]
        print("load videogpt weights from {}".format(checkpoint_path))
        gpt.model.load_state_dict(torch.load(checkpoint_path)["model_state_dict"])

        # If replicate is True and using distributed training, replicate the model as needed
        # if replicate:
        #     # Adjust this part based on your distributed training setup in PyTorch
        #     pass

    return gpt, class_map

def load_vqgan(path, ae_config):
    # Load configuration
    # config = pickle.load(open(osp.join(path, 'args'), 'rb'))
    
    # Initialize the VQGAN model with the loaded configuration
    model = VQGAN(**ae_config) # .to(device)  # Replace VQGAN with your PyTorch implementation
    # if ae_config['ddp']:
    #    model = DDP(model, device_ids=[device])

    # Load mask map if exists
    mask_file = osp.join(path, 'mask_map.pkl')
    if osp.exists(mask_file):
        mask_map = pickle.load(open(mask_file, 'rb'))
        mask_map = {k: torch.tensor(v, dtype=torch.uint8) for k, v in mask_map.items()}
    else:
        mask_map = None

    # Load model checkpoint
    # checkpoint_path = osp.join(path, 'checkpoints')
    # if osp.exists(checkpoint_path):
    #     model.load_state_dict(torch.load(checkpoint_path))
    model_files = glob.glob(f"{path}/checkpoints/*.pth")

    if len(model_files):
        checkpoint_path = sorted(model_files, key=extract_iteration)[-1]
        print("load vqgan weights from {}".format(checkpoint_path))
        # if ae_config['ddp']:
        #     model.module.load_state_dict(torch.load(checkpoint_path)["model_state_dict"])
        # else:
        model.load_state_dict(torch.load(checkpoint_path)["model_state_dict"])

    return model, mask_map


class AE:
    def __init__(self, path, ae_config):
        self.ddp = ae_config["ddp"]
        path = osp.expanduser(path)
        self.ae, self.mask_map = load_vqgan(path, ae_config)  # Assuming load_vqgan is adapted for PyTorch
        # PyTorch doesn't have a direct equivalent of JAX's 'pmap' or 'jit' mode
        # if self.ddp:
        #     self.ae = self.ae.module

    def latent_shape(self, image_size):
        if isinstance(self.ae, nn.DataParallel):
            return self.ae.module.latent_shape(image_size) # (8, 8)
        else:
            return self.ae.latent_shape(image_size) # (8, 8)

    @property
    def channels(self):
        if isinstance(self.ae, nn.DataParallel):
            return self.ae.module.codebook_embed_dim
        else:
            return self.ae.codebook_embed_dim
    
    @property
    def n_embed(self): 
        if isinstance(self.ae, nn.DataParallel):
            return self.ae.module.n_embed
        else:
            return self.ae.n_embed

    def encode(self, video):
        # Assuming the AE model has 'encode' method
        T = video.shape[1]
        video = video.reshape(-1, *video.shape[2:])
        out = self.ae(video)
        encodings = out['encodings']
        return encodings.reshape(-1, T, *encodings.shape[1:])
    
    def decode(self, encodings):
        # Assuming the AE model has 'decode' method
        T = encodings.shape[1]
        encodings = encodings.reshape(-1, *encodings.shape[2:])
        recon = self.ae(encodings, encode=False, is_embed=False)
        
        recon = torch.clip(recon, -1, 1)
        return recon.reshape(-1, T, *recon.shape[1:])
    
    def lookup(self, encodings, permute=True):
        # Assuming the AE model has 'codebook_lookup' method
        # if isinstance(self.ae, nn.DataParallel):
        #     return self.ae.module.codebook_lookup(encodings, permute=permute)
        # else:
        #     return self.ae.codebook_lookup(encodings, permute=permute)
        return self.ae(encodings, lookup=True, permute=permute)

    def prepare_batch(self, batch):
        # Prepare the batch for training, similar logic as in JAX
        if 'encodings' in batch:
            encodings = batch.pop('encodings').to(self.device)
        else:
            video = batch.pop('video').to(self.device) 
            # (..., H, W, C) -> (..., C, H, W)
            axis_order = tuple(range(video.ndim - 3)) + (video.ndim - 1, video.ndim - 3, video.ndim - 2)
            video = torch.permute(video, axis_order)
            # print(video.shape) # [64, 16, 3, 64, 64]
            encodings = self.encode(video)
            # print(encodings.shape) # [64, 16, 8, 8]

        if 'embeddings' in batch:
            embeddings = batch.pop('embeddings').to(self.device)
        else:
            embeddings = self.lookup(encodings, permute=False)
        # print(embeddings.shape)
        # print(encodings.shape)
        batch.update(embeddings=embeddings, encodings=encodings)
        return batch

