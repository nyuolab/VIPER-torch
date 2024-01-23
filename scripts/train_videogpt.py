import os
import os.path as osp
import pathlib
import sys
import numpy as np
import re
import time
import argparse
import yaml
import pickle
import random
import wandb
from datetime import datetime
import glob

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group(backend='nccl')

global is_master_process

# check: find . -maxdepth 1 -type d -name '*dmc_videogpt*'
# find . -maxdepth 1 -type d -name '*dmc_videogpt*' -exec rm -rf {} \;

directory = pathlib.Path(__file__).resolve()
directory = directory.parent
sys.path.append(str(directory.parent))

from viper_rl.videogpt.models import AE, VideoGPT
from viper_rl.videogpt.sampler import VideoGPTSampler
from viper_rl.videogpt.data import load_dataset
from viper_rl.videogpt.train_utils import init_model_state_videogpt, get_first_device, ProgressMeter, \
    save_video_grid, add_border, save_video

def extract_iteration(filename):
    match = re.search(r"checkpoint_(\d+).pth", filename)
    return int(match.group(1)) if match else 0

def main():
    global model
    global ckpt_dir
    # global best_loss
    
    seed = config.seed
    torch.manual_seed(seed)
    num_device = torch.cuda.device_count()

    config.best_loss = float('inf')

    if config.ddp:
        torch.cuda.manual_seed_all(seed)
        rank = dist.get_rank()
        print(f"Start running basic DDP on rank {rank}.")
        
        # create model and move it to GPU with id rank
        device = rank % num_device
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create a new generator (equivalent to a new stream of random numbers)
    new_generator = torch.Generator()
    new_generator.manual_seed(config.seed)  # Optionally, seed the new generator

    config.ckpt = config.output_dir if osp.exists(config.output_dir) else None
    # config.ckpt = None
    config.device = device
    config.ae['device'] = device

    ckpt_dir = osp.join(config.output_dir, 'checkpoints')

    if is_master_process:
        wandb.init(project='videogpt', config=config,
                   id=config.run_id, resume='allow', mode='online')
        wandb.run.name = config.run_id
        wandb.run.save()

    train_loader, class_map, _ = load_dataset(config, train=True, num_ds_shards=dist.get_world_size(), ds_shard_id=dist.get_rank(), modality='video')
    test_loader, class_map_test, _ = load_dataset(config, train=False, num_ds_shards=dist.get_world_size(), ds_shard_id=dist.get_rank(), modality='video')

    rev_class_map = {val:key for key,val in class_map.items()}
    config.rev_class_map = rev_class_map

    if config.class_cond:
        assert class_map == class_map_test, (class_map, class_map_test)
        pickle.dump(class_map, open(osp.join(config.output_dir, 'class_map.pkl'), 'wb'))

    # print(config.ae)
    ae = AE(config.ae_ckpt, config.ae)


    batch = next(iter(train_loader))
    print(batch.keys())

    with torch.no_grad():
        batch = ae.prepare_batch(batch)
    batch = get_first_device(batch)
    
    model = VideoGPT(config, ae)
    model.to(device)
    if config.ddp:
        model = DDP(model, device_ids=[device])
    sampler = VideoGPTSampler(model, ddp=config.ddp)
   
    state = init_model_state_videogpt(model, batch, config)

    start_iteration = 0
    if config.ckpt is not None:
        model_files = glob.glob(f"{ckpt_dir}/*.pth")
        if len(model_files):
            checkpoint_path = sorted(model_files, key=extract_iteration)[-1]
            print("load videogpt weights from {}".format(checkpoint_path))
            checkpoint = torch.load(checkpoint_path)
            if config.ddp:
                state.model.module.load_state_dict(checkpoint['model_state_dict'])
                state.model.module.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            else:
                state.model.load_state_dict(checkpoint['model_state_dict'])
                state.model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_iteration = checkpoint['iteration']
            print(f'Restored from checkpoint {os.path.join(config.ckpt)}, at iteration {start_iteration}')

    config.start_iter = start_iteration

    iteration = start_iteration

    while iteration < config.total_steps:
        torch.manual_seed(iteration + random.randint(0, 100000))
        iteration, state = train(iteration, ae, state, train_loader, test_loader, sampler, config, device)
        


def train_step(batch, state, device, step, ddp=False):
    if ddp:    
        state_model = state.model.module
    else:
        state_model = state.model

    state_model.train()

    batch = {k: v.to(device) for k, v in batch.items()}

    state_model.optimizer.zero_grad()

    # Forward pass with dropout
    # outputs = state.model(**batch)
    
    loss = state_model.loss(batch)

    # Backward pass and optimize
    loss["loss"].backward()
    state_model.optimizer.step()

    # Update EMA parameters if enabled
    if config.ema:
        decay = config.ema if step > 0 else 0.0
        with torch.no_grad():
            state.update_ema(decay)

    return state, loss



def train(iteration, ae, state, train_loader, test_loader, sampler, config, device):
    progress = ProgressMeter(
        config.total_steps,
        ['time', 'data'] + (state.model.module.metrics if config.ddp else state.model.metrics)
    )

    end = time.time()
    for batch in train_loader:
        batch_size = batch[list(batch.keys())[0]].shape[0]
        # print(batch[list(batch.keys())[0]].shape) # [64, 16, 64, 64, 3]
        if iteration % config.viz_interval == 0:
            visualize(sampler, ae, iteration, state, test_loader)

        progress.update(data=time.time() - end)

        with torch.no_grad():
            batch = ae.prepare_batch(batch)
        # print(batch.keys())
        state, return_dict = train_step(batch=batch, state=state, device=device, step=iteration, ddp=config.ddp)
        if config.ddp:
            state.model.module.scheduler.step(iteration)
            metrics = state.model.module.metrics
        else:
            state.model.scheduler.step(iteration)
            metrics = state.model.metrics
        metrics = {k: return_dict[k].detach().cpu().numpy().mean() for k in metrics}
        metrics = {k: v.astype(np.float32) for k, v in metrics.items()}
        progress.update(n=batch_size, **{k: v for k, v in metrics.items()})

        if is_master_process:
            scheduler = state.model.module.scheduler if config.ddp else state.model.scheduler
            wandb.log({'train/lr': scheduler.get_last_lr()[0]}, step=iteration)
            wandb.log({**{f'train/{metric}': val
                        for metric, val in metrics.items()}
                    }, step=iteration)

        progress.update(time=time.time() - end)
        end = time.time()

        if iteration % config.log_interval == 0:
            progress.display(iteration)

        if iteration % config.test_interval == 0:
            val_loss = validate(iteration, ae, state, test_loader, device, config.ddp)
            # is_best = val_loss < config.best_loss
            config.best_loss = min(config.best_loss, val_loss)
    
        
        if iteration % config.save_interval == 0 and is_master_process and iteration > config.start_iter: # and is_best:
            save_path = os.path.join(ckpt_dir, f'checkpoint_{iteration}.pth')
            torch.save({
                'iteration': iteration,
                'model_state_dict': state.model.module.state_dict() if config.ddp else state.model.state_dict(),
                'optimizer_state_dict': state.optimizer.state_dict()
            }, save_path)
            print('Saved checkpoint to', save_path)
            print('Saved checkpoint at iteration', iteration)

        # if iteration % config.viz_interval == 0 or \
        # iteration % config.test_interval == 0 or \
        # iteration % config.save_interval == 0 or \
        # iteration >= config.total_steps:
        #     return iteration, state, rngs

        iteration += 1
    
    return iteration, state


def val_step(batch, state, device, ddp=False):
    if ddp:    
        state.model.module.eval()
    else:
        state.model.eval()

    batch = {k: v.to(device) for k, v in batch.items()}

    with torch.no_grad():
        # Forward pass
        loss = state.model.module.loss(batch, training=False) if ddp else state.model.loss(batch, training=False)

        # If using Distributed Data Parallel, the averaging across devices is handled automatically.
        # If not, and you need to average outputs across devices, you would need to do it manually here.

    return loss


def validate(iteration, ae, state, test_loader, device, ddp):
    if ae.ddp:
        metrics = state.model.module.metrics
    else:
        metrics = state.model.metrics
    
    progress = ProgressMeter(
        50,
        ['time', 'data'] + metrics,
        prefix='\tTest:'
    )

    end = time.time()
    for i in range(50):
        batch = next(iter(test_loader))
        batch_size = batch[list(batch.keys())[0]].shape[1]
        progress.update(data=time.time() - end)
        with torch.no_grad():
            batch = ae.prepare_batch(batch)
        return_dict = val_step(batch=batch, state=state, device=device, ddp=ddp)
        
        metrics = {k: return_dict[k].detach().cpu().numpy().mean() for k in metrics}
        metrics = {k: v.astype(np.float32) for k, v in metrics.items()}
        progress.update(n=batch_size, **{k: v for k, v in metrics.items()})
        progress.update(time=time.time() - end)
        end = time.time()

        # if i % config.log_eval_interval == 0:
        #     progress.display(i)

    progress.display(i)

    
    metrics = {metric: progress.meters[metric].avg
                for metric in metrics}

    if is_master_process:
        wandb.log({**{f'val/{metric}': val
                      for metric, val in metrics.items()}
                  }, step=iteration)
                  
    return metrics['loss']


def visualize(sampler, ae, iteration, state, test_loader):
    batch = next(iter(test_loader))
    real = batch['video'].detach().cpu().numpy()
    labels = batch['label'].detach().cpu().numpy()
    # print(label.shape)
    # print(label)
    str_labels = [config.rev_class_map[l] for l in labels]
    print(str_labels)

    # print(video.shape) # (batch_size, seq_len, height, width, channels)
    # if len(video.shape) == 5: # NBTHW
    #     video = ae.decode(video)
    # variables = {'params': state.ema_params if hasattr(state, 'ema_params') else state.params}
    samples = sampler(batch) # .copy()
    samples = samples.reshape(-1, *samples.shape[-4:])
    new_axes = tuple(range(samples.ndim - 3)) + (samples.ndim - 2, samples.ndim - 1, samples.ndim - 3)
    samples = np.transpose(samples, new_axes)
    
    real = (real * 0.5 + 0.5).reshape(-1, *real.shape[-4:])
    add_border(samples[:, :config.open_loop_ctx], (0., 1., 0.))
    add_border(samples[:, config.open_loop_ctx:], (1., 0., 0.))
    # print(samples.shape)
    # print(real.shape)
    
    videos = np.stack((samples, real), axis=1)
    videos = videos.reshape(-1, *videos.shape[2:])
    videos = (videos * 255).astype(np.uint8)

    videos = save_video_grid(videos)
    if is_master_process:
        videos = np.transpose(videos, (0, 3, 1, 2))
        table = wandb.Table(columns=["class"])
        for l in str_labels:
            table.add_data(l)
        wandb.log({'viz/sample': wandb.Video(videos, format='gif'),  'label': table}, step=iteration)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_dir', type=str, required=True)
    parser.add_argument('-c', '--config', type=str, required=True)
    args = parser.parse_args()

    args.run_id = args.output_dir.split('/')[-1] + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    world_size = torch.cuda.device_count()

    if torch.cuda.is_available():
        if dist.is_available() and dist.is_initialized():
            # If using PyTorch Distributed Data Parallel (DDP)
            print(f'PyTorch process: {dist.get_rank()} / {dist.get_world_size()}')
        else:
            # If not using DDP, the concept of process index/count doesn't directly apply
            print('PyTorch process: N/A (not using Distributed Data Parallel)')
        
        print(f'Total CUDA devices: {world_size}')
        print(f'Current CUDA device index: {torch.cuda.current_device()}')
        print(f'Current CUDA device name: {torch.cuda.get_device_name(torch.cuda.current_device())}')
    else:
        print('CUDA is not available. Using CPU only.')


    config = yaml.safe_load(open(args.config, 'r'))
    if os.environ.get('DEBUG') == '1':
        config['viz_interval'] = 10
        config['log_interval'] = 1
        config['test_interval'] = 10
        args.output_dir = osp.join(osp.dirname(args.output_dir), f'DEBUG_{osp.basename(args.output_dir)}')
        args.run_id = f'DEBUG_{args.run_id}'

    print(f"Logging to {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    args_d = vars(args)
    args_d.update(config)
    pickle.dump(args, open(osp.join(args.output_dir, 'args'), 'wb'))
    config = args
    
    is_master_process = dist.get_rank() == 0

    main()
    # mp.spawn(main, args=(config,), nprocs=world_size, join=True)