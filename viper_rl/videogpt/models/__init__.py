from functools import cached_property, partial
import os.path as osp
import pickle
import numpy as np
import torch

from .vqgan import VQGAN
from .videogpt import VideoGPT
from .stylegan_disc import StyleGANDisc
from .vqgan import VQGAN


def load_videogpt(path, replicate=True, ae=None):
    # Load configuration
    config = pickle.load(open(osp.join(path, 'args'), 'rb'))
    
    # Initialize the AE model if not provided
    if ae is None:
        ae = AE(config.ae_ckpt)  # Adjust AE initialization as per your PyTorch implementation

    # Initialize the VideoGPT model
    model = VideoGPT(config, ae)

    # Load class mapping file if exists
    class_file = osp.join(path, 'class_map.pkl')
    if osp.exists(class_file):
        class_map = pickle.load(open(class_file, 'rb'))
        class_map = {k: int(v) for k, v in class_map.items()}
    else:
        class_map = None

    # Load model checkpoint
    checkpoint_path = osp.join(path, 'checkpoints')
    if osp.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        # If replicate is True and using distributed training, replicate the model as needed
        if replicate:
            # Adjust this part based on your distributed training setup in PyTorch
            pass

    return model, class_map

def load_vqgan(path):
    # Load configuration
    config = pickle.load(open(osp.join(path, 'args'), 'rb'))
    
    # Initialize the VQGAN model with the loaded configuration
    model = VQGAN(**config.ae)  # Replace VQGAN with your PyTorch implementation

    # Load mask map if exists
    mask_file = osp.join(path, 'mask_map.pkl')
    if osp.exists(mask_file):
        mask_map = pickle.load(open(mask_file, 'rb'))
        mask_map = {k: torch.tensor(v, dtype=torch.uint8) for k, v in mask_map.items()}
    else:
        mask_map = None

    # Load model checkpoint
    checkpoint_path = osp.join(path, 'checkpoints')
    if osp.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))

    return model, mask_map


class AE:
    def __init__(self, path):
        path = osp.expanduser(path)
        self.ae, self.mask_map = load_vqgan(path)  # Assuming load_vqgan is adapted for PyTorch
        # PyTorch doesn't have a direct equivalent of JAX's 'pmap' or 'jit' mode

    def latent_shape(self, image_size):
        return self.ae.latent_shape(image_size)

    @property
    def channels(self):
        return self.ae.codebook_embed_dim
    
    @property
    def n_embed(self):
        return self.ae.n_embed

    def encode(self, video):
        # Assuming the AE model has 'encode' method
        encodings = self.ae.encode(video)
        return encodings
    
    def decode(self, encodings):
        # Assuming the AE model has 'decode' method
        recon = self.ae.decode(encodings)
        recon = torch.clip(recon, -1, 1)
        return recon
    
    def lookup(self, encodings):
        # Assuming the AE model has 'codebook_lookup' method
        return self.ae.codebook_lookup(encodings)

    def prepare_batch(self, batch):
        # Prepare the batch for training, similar logic as in JAX
        if 'encodings' in batch:
            encodings = batch.pop('encodings')
        else:
            video = batch.pop('video')
            encodings = self.encode(video)

        if 'embeddings' in batch:
            embeddings = batch.pop('embeddings')
        else:
            embeddings = self.lookup(encodings)
        batch.update(embeddings=embeddings, encodings=encodings)
        return batch

