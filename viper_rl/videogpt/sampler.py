# from functools import cached_property
from tqdm import tqdm
import numpy as np
import torch

class VideoGPTSampler:
    def __init__(self, model):
        self.model = model
        self.config = model.config

    # @cached_property
    def _model_step(self, embeddings, label, decode_step, training=False):
        # Implement the logic to get logits and cache from the model
        # PyTorch manages cache (states) differently, so this needs to be adapted
        with torch.no_grad():
            # print(label.dtype)
            logits = self.model(embeddings, label=label, decode_step=decode_step, training=False)
        return logits  # Cache handling will depend on your model's architecture

    # @cached_property
    def _sample_step(self, logits):
        probabilities = torch.softmax(logits, dim=-1)
        samples = torch.multinomial(probabilities.squeeze(-2), num_samples=1) # .unsqueeze(-1)
        return samples
        
    def __call__(self, batch, log_tqdm=True, open_loop_ctx=None, decode=True):
        # Prepare the batch
        # batch = {k: v.clone().detach().cpu() for k, v in batch.items()}
        with torch.no_grad():
            batch = self.model.ae.prepare_batch(batch)
        encodings = batch.pop('encodings') # [batch_size, seq_len, height, width]
        label = batch['label'].to(self.config.device)
        # print(label.dtype)

        # print("embeddings shape is {}".format(batch["embeddings"].shape))
        # [batch_size, seq_len, height, width, embed_dim]

        # Initialize the random number generator
        # torch.manual_seed(seed)

        # Setup for sampling
        samples = torch.zeros_like(encodings).to(self.config.device) # [batch_size, seq_len, height, width]
        latent_shape = samples.shape[-3:]
        ctx = open_loop_ctx or self.config.open_loop_ctx
        samples[:, :ctx] = encodings[:, :ctx]
        samples = samples.reshape(*samples.shape[:-3], -1)

        # print("samples shape is {}".format(samples.shape))

        # Define sampling range
        n_cond = np.prod(latent_shape[1:]) * ctx
        n_tokens = np.prod(latent_shape)
        itr = range(n_cond, n_tokens)
        if log_tqdm:
            itr = tqdm(itr)

        # Sampling loop
        for i in itr:
            # Retrieve logits from the model
            # Assuming self._model_step is adapted for PyTorch
            # embed_dim = [batch_size, 1, embed_dim=64]
            logits = self._model_step(self.model.ae.lookup(samples[...,  i-1, None], permute=False), label, i)
            # print(logits.shape) # [72, 1, 256]
            # Sample from logits
            # Assuming self._sample_step is adapted for PyTorch
            s = self._sample_step(logits)
            samples[..., i, None] = s # .detach.cpu()

        # Reshape and decode samples if required
        samples = samples.reshape(*samples.shape[:-1], *latent_shape)
        if decode:
            samples = self.model.ae.decode(samples) * 0.5 + 0.5
        samples = samples.detach().cpu().numpy()

        return samples