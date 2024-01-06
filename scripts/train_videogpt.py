import os
import os.path as osp
import pathlib
import sys
import numpy as np
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


def main():
    global model
    global ckpt_dir
    
    seed = config.seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create a new generator (equivalent to a new stream of random numbers)
    new_generator = torch.Generator()
    new_generator.manual_seed(config.seed)  # Optionally, seed the new generator

    # config.ckpt = config.output_dir if osp.exists(config.output_dir) else None
    config.ckpt = None
    config.device = device
    ckpt_dir = osp.join(config.output_dir, 'checkpoints')

    if is_master_process:
        wandb.init(project='videogpt', config=config,
                   id=config.run_id, resume='allow', mode='online')
        wandb.run.name = config.run_id
        wandb.run.save()

    train_loader, class_map, _ = load_dataset(config, train=True, num_ds_shards=dist.get_world_size(), ds_shard_id=dist.get_rank(), modality='video')
    test_loader, class_map_test, _ = load_dataset(config, train=False, num_ds_shards=dist.get_world_size(), ds_shard_id=dist.get_rank(), modality='video')

    if config.class_cond:
        assert class_map == class_map_test, (class_map, class_map_test)
        pickle.dump(class_map, open(osp.join(config.output_dir, 'class_map.pkl'), 'wb'))

    # print(config.ae)
    ae = AE(config.ae_ckpt, config.ae)


    batch = next(iter(train_loader))
    # print(batch.keys())
    batch = ae.prepare_batch(batch)
    batch = get_first_device(batch)
    
    model = VideoGPT(config, ae)
    model.to(device)
    sampler = VideoGPTSampler(model)
   
    state = init_model_state_videogpt(model, batch, config)

    if config.ckpt is not None:
        checkpoint = torch.load(glob.glob(config.ckpt)[-1])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_iteration = checkpoint['iteration']
        print(f'Restored from checkpoint {os.path.join(config.ckpt)}, at iteration {start_iteration}')
    else:
        start_iteration = 0


    best_loss = float('inf')

    iteration = start_iteration

    while iteration < config.total_steps:
        torch.manual_seed(iteration + random.randint(0, 100000))
        iteration, state = train(iteration, ae, state, train_loader, config, device)
        if iteration % config.test_interval == 0:
            val_loss = validate(iteration, ae, state, test_loader, device)
            is_best = val_loss < best_loss
            best_loss = min(best_loss, val_loss)
        # if iteration % config.viz_interval == 0:
        #     visualize(sampler, ae, iteration, state, test_loader)
        
        


def train_step(batch, state, device, step):
    state.model.train()

    batch = {k: v.to(device) for k, v in batch.items()}

    state.model.optimizer.zero_grad()

    # Forward pass with dropout
    # outputs = state.model(**batch)
    
    loss = state.model.loss(batch)

    # Backward pass and optimize
    loss["loss"].backward()
    state.model.optimizer.step()

    # Update EMA parameters if enabled
    if config.ema:
        decay = config.ema if step > 0 else 0.0
        with torch.no_grad():
            state.update_ema(decay)

    return state, loss



def train(iteration, ae, state, train_loader, config, device):
    progress = ProgressMeter(
        config.total_steps,
        ['time', 'data'] + model.metrics
    )

    end = time.time()
    for batch in train_loader:
        batch_size = batch[list(batch.keys())[0]].shape[1]
        # print(batch[list(batch.keys())[0]].shape) # [64, 16, 64, 64, 3]
        progress.update(data=time.time() - end)

        batch = ae.prepare_batch(batch)
        # print(batch.keys())
        state, return_dict = train_step(batch=batch, state=state, device=device, step=iteration)

        metrics = {k: return_dict[k].detach().cpu().numpy().mean() for k in model.metrics}
        metrics = {k: v.astype(np.float32) for k, v in metrics.items()}
        progress.update(n=batch_size, **{k: v for k, v in metrics.items()})

        if is_master_process:
            wandb.log({'train/lr': state.model.scheduler.get_last_lr()[-1]}, step=iteration)
            wandb.log({**{f'train/{metric}': val
                        for metric, val in metrics.items()}
                    }, step=iteration)

        progress.update(time=time.time() - end)
        end = time.time()

        if iteration % config.log_interval == 0:
            progress.display(iteration)
        
        if iteration % config.save_interval == 0 and is_master_process: # and is_best:
            save_path = os.path.join(ckpt_dir, f'checkpoint_{iteration}.pth')
            torch.save({
                'iteration': iteration,
                'model_state_dict': state.model.state_dict(),
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


def val_step(batch, state, device):
    state.model.eval()

    batch = {k: v.to(device) for k, v in batch.items()}

    with torch.no_grad():
        # Forward pass
        loss = state.model.loss(batch, training=False)

        # If using Distributed Data Parallel, the averaging across devices is handled automatically.
        # If not, and you need to average outputs across devices, you would need to do it manually here.

    return loss


def validate(iteration, ae, state, test_loader, device):
    progress = ProgressMeter(
        50,
        ['time', 'data'] + model.metrics,
        prefix='\tTest:'
    )

    end = time.time()
    for i in range(50):
        batch = next(iter(test_loader))
        batch_size = batch[list(batch.keys())[0]].shape[1]
        progress.update(data=time.time() - end)

        batch = ae.prepare_batch(batch)
        return_dict = val_step(batch=batch, state=state, device=device)

        metrics = {k: return_dict[k].detach().cpu().numpy().mean() for k in model.metrics}
        metrics = {k: v.astype(np.float32) for k, v in metrics.items()}
        progress.update(n=batch_size, **{k: v for k, v in metrics.items()})
        progress.update(time=time.time() - end)
        end = time.time()

        if i % config.log_interval == 0:
            progress.display(i)

    progress.display(i)

    metrics = {metric: progress.meters[metric].avg
               for metric in model.metrics}

    if is_master_process:
        wandb.log({**{f'val/{metric}': val
                      for metric, val in metrics.items()}
                  }, step=iteration)
    return metrics['loss'], rngs


def visualize(sampler, ae, iteration, state, test_loader):
    batch = next(test_loader)
    video = batch['video']
    if len(video.shape) == 5: # NBTHW
        video = ae.decode(video)
    variables = {'params': state.ema_params if hasattr(state, 'ema_params') else state.params}
    samples = sampler(variables, batch).copy()
    samples = samples.reshape(-1, *samples.shape[-4:])
    real = video.detach().cpu()
    real = (real * 0.5 + 0.5).reshape(-1, *real.shape[-4:])
    add_border(samples[:, :config.open_loop_ctx], (0., 1., 0.))
    add_border(samples[:, config.open_loop_ctx:], (1., 0., 0.))

    videos = np.stack((samples, real), axis=1)
    videos = videos.reshape(-1, *videos.shape[2:])
    videos = (videos * 255).astype(np.uint8)

    videos = save_video_grid(videos)
    if is_master_process:
        videos = np.transpose(videos, (0, 3, 1, 2))
        wandb.log({'viz/sample': wandb.Video(videos, format='gif')}, step=iteration)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_dir', type=str, required=True)
    parser.add_argument('-c', '--config', type=str, required=True)
    args = parser.parse_args()

    args.run_id = args.output_dir.split('/')[-1] + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    if torch.cuda.is_available():
        if dist.is_available() and dist.is_initialized():
            # If using PyTorch Distributed Data Parallel (DDP)
            print(f'PyTorch process: {dist.get_rank()} / {dist.get_world_size()}')
        else:
            # If not using DDP, the concept of process index/count doesn't directly apply
            print('PyTorch process: N/A (not using Distributed Data Parallel)')

        print(f'Total CUDA devices: {torch.cuda.device_count()}')
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