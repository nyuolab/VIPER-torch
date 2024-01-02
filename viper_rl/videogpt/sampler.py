# from functools import cached_property
from tqdm import tqdm
import numpy as np
import torch

class VideoGPTSampler:
    def __init__(self, model):
        self.ae = model.ae
        self.model = model
        self.mode = mode
        self.config = model.config

    # @cached_property
    def _model_step(self, embeddings, label, decode_step):
        # Implement the logic to get logits and cache from the model
        # PyTorch manages cache (states) differently, so this needs to be adapted
        logits = self.model(embeddings, label=label, decode_step=decode_step, training=False)
        return logits  # Cache handling will depend on your model's architecture

    # @cached_property
    def _sample_step(self, logits):
        probabilities = torch.softmax(logits, dim=-1)
        samples = torch.multinomial(probabilities, num_samples=1).squeeze(-1)
        return samples
        
    def __call__(self, variables, batch, seed=0, log_tqdm=True, open_loop_ctx=None, decode=True):
        # Prepare the batch
        # Assuming self.ae.prepare_batch is adapted for PyTorch
        batch = {k: v.clone().detach().cpu() for k, v in batch.items()}
        batch = self.ae.prepare_batch(batch)
        encodings = batch.pop('encodings')
        label = batch.pop('label', None)

        # Initialize the random number generator
        torch.manual_seed(seed)

        # Setup for sampling
        samples = torch.zeros_like(encodings)
        latent_shape = samples.shape[-3:]
        ctx = open_loop_ctx or self.config.open_loop_ctx
        samples[..., :ctx] = encodings[..., :ctx]
        samples = samples.reshape(*samples.shape[:-3], -1)

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
            logits = self._model_step(samples[..., i - 1:i], label, 0)

            # Sample from logits
            # Assuming self._sample_step is adapted for PyTorch
            s = self._sample_step(logits)
            samples[..., i, None] = s # .detach.cpu()

        # Reshape and decode samples if required
        samples = samples.reshape(*samples.shape[:-1], *latent_shape)
        if decode:
            samples = self.ae.decode(samples) * 0.5 + 0.5

        return samples.detach.cpu()