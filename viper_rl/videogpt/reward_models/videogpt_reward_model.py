from typing import Any, Union, Dict, Tuple, Optional, List
import os
import pickle
import functools
import numpy as np
import torch
# import torch.nn as nn
import torch.nn.functional as F
import tqdm
import argparse

from ..models import load_videogpt, AE
from .. import sampler

# tree_map = jax.tree_util.tree_map
# sg = lambda x: tree_map(jax.lax.stop_gradient, x)

def stop_gradient_on_structure(structure):
    if isinstance(structure, torch.Tensor):
        return structure.detach()
    elif isinstance(structure, (list, tuple)):
        return type(structure)(stop_gradient_on_structure(item) for item in structure)
    else:
        raise TypeError("Unsupported type for stop_gradient_on_structure")

class InvalidSequenceError(Exception):
    def __init__(self, message):            
        super().__init__(message)


class VideoGPTRewardModel:

    PRIVATE_LIKELIHOOD_KEY = 'logimmutabledensity'
    PUBLIC_LIKELIHOOD_KEY = 'density'

    def __init__(self, task: str, 
                 ae_config, config,
                 vqgan_path: str, videogpt_path: str,
                 camera_key: str='image',
                 reward_scale: Optional[Union[Dict[str, Tuple], Tuple]]=None,
                 reward_model_device: int=0,
                 nll_reduce_sum: bool=True,
                 compute_joint: bool=True,
                 minibatch_size: int=64,
                 encoding_minibatch_size: int=64):
        """VideoGPT likelihood model for reward computation.

        Args:
            task: Task name, used for conditioning when class_map.pkl exists in videogpt_path.
            vqgan_path: Path to vqgan weights.
            videogpt_path: Path to videogpt weights.
            camera_key: Key for camera observation.
            reward_scale: Range to scale logits from [0, 1].
            reward_model_device: Device to run reward model on.
            nll_reduce_sum: Whether to reduce sum over nll.
            compute_joint: Whether to compute joint likelihood or use conditional.
            minibatch_size: Minibatch size for VideoGPT.
            encoding_minibatch_size: Minibatch size for VQGAN.
        """
        self.domain, self.task = task.split('_', 1)
        self.vqgan_path = vqgan_path
        self.videogpt_path = videogpt_path
        self.camera_key = camera_key
        self.reward_scale = reward_scale
        self.nll_reduce_sum = nll_reduce_sum
        self.compute_joint = compute_joint
        self.minibatch_size = minibatch_size
        self.encoding_minibatch_size = encoding_minibatch_size

        # Load VQGAN and VideoGPT weights.
        self.device = reward_model_device
        print(f'Reward model devices: {self.device}')
        self.ae = AE(path=vqgan_path, ae_config=ae_config)
        self.ae.ae = self.ae.ae.to(self.device)
        self.model, self.class_map = load_videogpt(videogpt_path, config=config, ae_config=ae_config, ae=self.ae, replicate=False)
        # print(self.class_map)
        config = self.model.config
        self.sampler = sampler.VideoGPTSampler(self.model)
        self.model = self.model.to(self.device)

        self.model_name = config.model
        self.n_skip = getattr(config, 'frame_skip', 1)
        self.class_cond = getattr(config, 'class_cond', False)
        self.seq_len = config.seq_len * self.n_skip
        self.seq_len_steps = self.seq_len

        # Load frame mask.
        if self.ae.mask_map is not None:
            self.mask = self.ae.mask_map[self.task]
            print(f'Loaded mask for task {self.task} from mask_map with keys: {self.ae.mask_map.keys()}')
            self.mask = self.mask.to(torch.uint8).to(self.device)
        else:
            self.mask = None

        # Load task id.
        if self.class_cond and self.class_map is not None:
            self.task_id = None
            print(f'Available tasks: {list(self.class_map.keys())}')
            assert (self.task in self.class_map,
                    f'{self.task} not found in class map.')
            self.task_id = int(self.class_map[self.task])
            print(f'Loaded conditioning information for task {self.task}')
        elif self.class_cond:
            raise ValueError(
                f'No class_map for class_conditional model. '
                f'VideoGPT loaded class_map? {self.class_map is not None}')
        else:
            self.task_id = None

        print(
            f'finished loading {self.__class__.__name__}:'
            f'\n\tseq_len: {self.seq_len}'
            f'\n\tclass_cond: {self.class_cond}'
            f'\n\ttask: {self.task}'
            f'\n\tmodel: {self.model_name}'
            f'\n\tcamera_key: {self.camera_key}'
            f'\n\tseq_len_steps: {self.seq_len_steps}'
            f'\n\tmask? {self.mask is not None}'
            f'\n\ttask_id: {self.task_id}'
            f'\n\tn_skip? {self.n_skip}')

    def __call__(self, seq, **kwargs):
        return self.process_seq(self.compute_reward(seq, **kwargs), **kwargs)

    def rollout_video(self, init_frames, video_length, seed=0, open_loop_ctx=4, inputs_are_codes=False, decode=True, pbar=False):
        torch.manual_seed(seed)

        if inputs_are_codes:
            encodings = init_frames
        else:
            init_frames = self.process_images(init_frames)  # This should be defined as per your model
            encodings = self.ae.encode(init_frames)  # Assuming ae has an encode method

        rollout_length = min(video_length, self.seq_len // self.n_skip)
        batch = {'encodings': encodings, 'label': self.task_id or 0}  # self.task_id needs to be defined

        if rollout_length > init_frames.shape[1]:
            padding = (0, 0, 0, 0, 0, rollout_length - init_frames.shape[1])
            encodings = F.pad(encodings, padding)
            batch['encodings'] = encodings
            encodings = self.sampler(batch, open_loop_ctx=init_frames.shape[1], decode=False)  # Assuming sampler is defined
            batch['encodings'] = encodings

        all_samples = [encodings]

        remaining_frames = video_length - encodings.shape[1]
        extra_sample_steps = max(remaining_frames // ((self.seq_len // self.n_skip) - open_loop_ctx), 0)
        vid_range = tqdm.tqdm(range(extra_sample_steps)) if pbar else range(extra_sample_steps)

        for _ in vid_range:
            batch['encodings'] = torch.roll(encodings, -((self.seq_len // self.n_skip) - open_loop_ctx), dims=1)
            encodings = self.sampler(batch, open_loop_ctx=open_loop_ctx, decode=False)
            all_samples.append(encodings[:, open_loop_ctx:])
        
        all_samples = torch.cat(all_samples, dim=1)

        if decode:
            decoded_samples = self.ae.decode(all_samples)  # Assuming ae has a decode method
            return (255 * (decoded_samples * 0.5 + 0.5)).byte().numpy()
        else:
            return all_samples.numpy()
        

    def _compute_likelihood(self, embeddings, encodings, label):
        # print(f'Tracing likelihood: Original embeddings shape: {embeddings.shape}, Encodings shape: {encodings.shape}')
        if self.n_skip > 1:
            encodings = encodings[:, self.n_skip - 1::self.n_skip]
            embeddings = embeddings[:, self.n_skip - 1::self.n_skip]
            print(f'\tAfter applying frame skip: Embeddings shape: {embeddings.shape}, Encodings shape: {encodings.shape}')
        # print(label)
        # Assuming the log_prob method is implemented in your PyTorch model
        likelihoods = self.model.log_prob(embeddings, encodings, label=label, reduce_sum=self.nll_reduce_sum)

        if self.compute_joint:
            ll = likelihoods.sum(dim=-1)
        else:
            ll = likelihoods[:, -1]

        return ll
    

    def _compute_likelihood_for_initial_elements(self, embeddings, encodings, label):
        # print(f'Tracing init frame likelihood: Embeddings shape: {embeddings.shape}, Encodings shape: {encodings.shape}')
        if self.n_skip > 1:
            first_encodings = torch.cat([encodings[:1, i::self.n_skip] for i in range(self.n_skip)], dim=0)
            first_embeddings = torch.cat([embeddings[:1, i::self.n_skip] for i in range(self.n_skip)], dim=0)
            print(f'\tAfter applying frame skip: Embeddings shape: {first_embeddings.shape}, Encodings shape: {first_encodings.shape}')
        else:
            first_encodings, first_embeddings = encodings[:1], embeddings[:1]
        likelihoods = self.model.log_prob(first_embeddings, first_encodings, label=label, reduce_sum=self.nll_reduce_sum)
        
        
        if self.n_skip > 1:
            idxs = np.arange(len(likelihoods.shape))
            idxs[0], idxs[1] = idxs[1], idxs[0]
            ll = likelihoods.permute(idxs.tolist()).reshape((-1,) + likelihoods.shape[2:])[:-1]
        else:
            ll = likelihoods[0, :-1]
        if self.compute_joint:
            ll = torch.cumsum(ll, dim=0) / torch.arange(1, len(ll) + 1).to(self.device)
        return ll

    
    def _reward_scaler(self, reward):
        if self.reward_scale:
            if isinstance(self.reward_scale, dict) and (self.task not in self.reward_scale):
                return reward
            rs = self.reward_scale[self.task] if isinstance(self.reward_scale, dict) else self.reward_scale
            reward = np.array(np.clip((reward - rs[0]) / (rs[1] - rs[0]), 0.0, 1.0))
            return reward
        else:
            return reward
    
    def compute_reward(self, seq):
        """Use VGPT model to compute likelihoods for input sequence.
        Args:
            seq: Input sequence of states.
        Returns:
            seq: Input sequence with additional keys in the state dict.
        """
        l = len(seq[self.camera_key])

        seq[VideoGPTRewardModel.PRIVATE_LIKELIHOOD_KEY] = [0] * l

        if l < self.seq_len_steps:
            raise InvalidSequenceError(f'Input sequence must be at least {self.seq_len_steps} steps long. Seq len is {l}')
        label = self.task_id if self.class_cond else None
        # print(label)
        # Where in sequence to start computing likelihoods. Don't perform redundant likelihood computations.
        # start_idx = 0
        # for i in range(self.seq_len_steps - 1, l):
        #     # if not self.is_step_processed(seq[i]):
        #         start_idx = i
        #         break
        # start_idx = int(max(start_idx - self.seq_len_steps + 1, 0))

        start_idx = 0
        T = l - start_idx

        # Compute encodings and embeddings for image sequence.
        # image_batch = torch.stack([seq[i][self.camera_key] for i in range(start_idx, l)])
        image_batch = np.stack(seq[self.camera_key][start_idx:])
        # print(image_batch.shape)
        image_batch = self.process_images(image_batch)
        encodings = self.ae.encode(torch.unsqueeze(image_batch, 0))
        embeddings = self.ae.lookup(encodings)
        encodings, embeddings = encodings[0], embeddings[0]

        # Compute batch of encodings and embeddings for likelihood computation.
        idxs = list(range(T - self.seq_len + 1))
        batch_encodings = [encodings[idx:(idx + self.seq_len)] for idx in idxs]
        batch_embeddings = [embeddings[idx:(idx + self.seq_len)] for idx in idxs]
        batch_encodings = torch.stack(batch_encodings) # .to(self.device)
        batch_embeddings = torch.stack(batch_embeddings) # .to(self.device)

        rewards = []
        for i in range(0, len(idxs), self.minibatch_size):
            mb_encodings = batch_encodings[i: i+self.minibatch_size]
            mb_embeddings = batch_embeddings[i: i+self.minibatch_size]
            mb_label = self.expand_scalar(label, mb_encodings.shape[0], torch.int64)
            reward = self._compute_likelihood(mb_embeddings, mb_encodings, mb_label) # .detach().cpu().numpy()
            rewards.append(reward)
            # print(reward.shape)
        # print(len(rewards))
        # progress
        rewards = torch.cat(rewards, dim=0).detach().cpu().numpy()
        # print(rewards.shape)
        if len(rewards) <= 1:
            rewards = self._reward_scaler(rewards)
        assert len(rewards) == (T - self.seq_len_steps + 1), f'{len(rewards)} != {T - self.seq_len_steps + 1}'
        for i, rew in enumerate(rewards):
            idx = start_idx + self.seq_len_steps - 1 + i
            # assert not self.is_step_processed(seq[idx])
            seq[VideoGPTRewardModel.PRIVATE_LIKELIHOOD_KEY][idx] = rew

        if seq['is_first'][0]:
            first_encodings = batch_encodings[:1]
            first_embeddings = batch_embeddings[:1]
            first_label = self.expand_scalar(label, first_encodings.shape[0], torch.int32)
            first_rewards = self._compute_likelihood_for_initial_elements(
                first_embeddings, first_encodings, first_label).detach().cpu().numpy()
            if len(first_rewards.shape) <= 1:
                first_rewards = self._reward_scaler(first_rewards)
            assert len(first_rewards) == self.seq_len_steps - 1, f'{len(first_rewards)} != {self.seq_len_steps - 1}'
            for i, rew in enumerate(first_rewards):
                # assert not self.is_step_processed(seq[i]), f'Step {i} already processed'
                seq[VideoGPTRewardModel.PRIVATE_LIKELIHOOD_KEY][i] = rew

        return seq

    def expand_scalar(self, scalar, size, dtype):
        if scalar is None: return None
        return torch.full((size,), scalar, dtype=dtype)
    
    def is_step_processed(self, step):
        return VideoGPTRewardModel.PRIVATE_LIKELIHOOD_KEY in step.keys()

    def is_seq_processed(self, seq):
        for step in seq:
            if not self.is_step_processed(step):
                return False
        return True

    def process_images(self, image_batch):
        if image_batch.shape[-1] <= 3:
            new_axes = tuple(range(image_batch.ndim - 3)) + (image_batch.ndim - 1, image_batch.ndim - 3, image_batch.ndim - 2)
            image_batch = np.transpose(image_batch, new_axes)
        image_batch = torch.tensor(image_batch, dtype=torch.uint8).to(self.device)
        # if image_batch.shape[-1] <= 3:
        #     image_batch = torch.permute(image_batch, (0, 3, 1, 2))
        image_batch = image_batch * self.mask if self.mask is not None else image_batch
        return image_batch.float() / 127.5 - 1.0

    def process_seq(self, seq):
        # for step in seq:
        #     if not self.is_step_processed(step):
        #         continue
        seq[VideoGPTRewardModel.PUBLIC_LIKELIHOOD_KEY] = seq[VideoGPTRewardModel.PRIVATE_LIKELIHOOD_KEY]
        return seq # [self.seq_len_steps - 1:]
    
    



    


    

