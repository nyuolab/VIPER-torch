import os
import os.path as osp
import pathlib
import sys
import numpy as np
import re
import time
import copy
import argparse
import yaml
import pickle
import random
import wandb
import datetime
from datetime import timedelta
import glob

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
# from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import DataParallel as DP

# global is_master_process

# check: find . -maxdepth 1 -type d -name '*dmc_videogpt*'
# find . -maxdepth 1 -type d -name '*dmc_videogpt*' -exec rm -rf {} \;

# os.environ['NCCL_DEBUG'] = 'INFO'
os.environ['GLOO_SOCKET_IFNAME'] = 'eth0'  # Replace 'eth0' with your interface name
os.environ['GLOO_LOG_LEVEL'] = 'DEBUG'
os.environ['NCCL_DEBUG'] = 'INFO'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
        
directory = pathlib.Path(__file__).resolve()
directory = directory.parent
sys.path.append(str(directory.parent))

from viper_rl.videogpt.models import AE, VideoGPT
from viper_rl.videogpt.sampler import VideoGPTSampler
from viper_rl.videogpt.data import load_dataset, prepare
from viper_rl.videogpt.train_utils import print_model_size, get_first_device, ProgressMeter, \
    save_video_grid, add_border, save_video

def extract_iteration(filename):
    match = re.search(r"checkpoint_(\d+).pth", filename)
    return int(match.group(1)) if match else 0

def main(config):
    global model
    global ckpt_dir
    # global best_loss
    rank = int(os.getenv('LOCAL_RANK', 0))
    seed = config.seed
    torch.manual_seed(seed)

    config.best_loss = float('inf')
    config.ae["ddp"] = config.ddp
    config.num_device = torch.cuda.device_count()
    
    if config.ddp or config.dp:
        config.batch_size *= config.num_device

    # Create a new generator (equivalent to a new stream of random numbers)
    new_generator = torch.Generator()
    new_generator.manual_seed(config.seed)  # Optionally, seed the new generator

    config.ckpt = config.output_dir if osp.exists(config.output_dir) else None
    # config.ckpt = None

    ckpt_dir = osp.join(config.output_dir, 'checkpoints')

    # if is_master_process:
    # if rank == 0:
    wandb.init(project='videogpt', config=config,
                id=config.run_id, resume='allow', mode='online')
    wandb.run.name = config.run_id
    wandb.run.save()

    train_dataset, class_map, _ = load_dataset(config, train=True, modality='video')
    test_dataset, class_map_test, _ = load_dataset(config, train=False, modality='video')
    
    rev_class_map = {val:key for key, val in class_map.items()}
    config.rev_class_map = rev_class_map

    if config.class_cond:
        assert class_map == class_map_test, (class_map, class_map_test)
        pickle.dump(class_map, open(osp.join(config.output_dir, 'class_map.pkl'), 'wb'))

    # batch = next(iter(train_loader))
    # print(batch.keys())

    # with torch.no_grad():
    #     batch = ae.prepare_batch(batch)
    # batch = get_first_device(batch)
    ae = AE(config.ae_ckpt, config.ae)
    gpt = VideoGPT(config, ae)

    if config.ckpt is not None:
        model_files = glob.glob(f"{ckpt_dir}/*.pth")
        if len(model_files):
            checkpoint_path = sorted(model_files, key=extract_iteration)[-1]
            print("load videogpt weights from {}".format(checkpoint_path))
            checkpoint = torch.load(checkpoint_path)
            gpt.model.load_state_dict(checkpoint['model_state_dict'])
            # gpt.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            config.start_iter = checkpoint['iteration']
            print(f'Restored from checkpoint {os.path.join(config.ckpt)}, at iteration {config.start_iter}')

    # if config.ddp:
    #     # dist.init_process_group(backend='nccl', world_size=world_size, timeout=datetime.timedelta(minutes=5))
    #     # mp.spawn(train_videogpt, args=(config, gpt, train_dataset, test_dataset), nprocs=world_size, join=True)
    # else:
    if config.ddp or config.dp:
        eval_device = "cuda:0"
        gpt_eval = copy.deepcopy(gpt)
        gpt_eval.device = eval_device
        gpt_eval.model.device = eval_device
        gpt_eval.ae.device = eval_device
    
        gpt_eval.ae.ae.to(eval_device)
        gpt_eval.model.to(eval_device)
        gpt_eval.model.position_bias_to_device(device=eval_device)

    if gpt_eval is None:
        sampler = VideoGPTSampler(gpt)
    else:
        sampler = VideoGPTSampler(gpt_eval)

    train_videogpt(rank, config, gpt, sampler, train_dataset, test_dataset)


def train_videogpt(rank, config, gpt, sampler, train_dataset, test_dataset):
    world_size = config.num_device
    if config.ddp:
        # rank = dist.get_rank()
        print(f"Start running basic DDP on rank {rank}.")
        # os.environ['MASTER_ADDR'] = 'localhost'
        # os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size, timeout=datetime.timedelta(minutes=5))
        # dist.init_process_group(backend='gloo', rank=rank, world_size=world_size, timeout=datetime.timedelta(minutes=5))
        # create model and move it to GPU with id rank
        torch.cuda.set_device(rank)
        device = rank % config.num_device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu" # torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.cuda.manual_seed_all(config.seed)

    train_loader = prepare(train_dataset, config.batch_size, world_size, rank, ddp=config.ddp)
    test_loader = prepare(test_dataset, config.batch_size, world_size, rank, ddp=config.ddp)

    config.device = device
    gpt.device = device
    gpt.model.device = device
    gpt.ae.device = device
    
    if config.ddp:
        gpt.ae.ae = DDP(gpt.ae.ae, device_ids=[device])
        # gpt.ae.ae = gpt.ae.ae.module
        gpt.model = DDP(gpt.model, device_ids=[device])
        # gpt.optimizer = torch.optim.AdamW(gpt.model.module.parameters(), lr=config.lr)
        gpt.model.module.position_bias_to_device()
    elif config.dp:
        gpt.ae.ae = DP(gpt.ae.ae)
        gpt.ae.ae.to(device)
        # gpt.ae.ae = gpt.ae.ae.module
        gpt.model = DP(gpt.model)
        gpt.model.to(device)
        # gpt.optimizer = torch.optim.AdamW(gpt.model.module.parameters(), lr=config.lr)
        gpt.model.module.position_bias_to_device()
    else:
        gpt.ae.ae.to(device)
        gpt.model.to(device)
        # gpt.optimizer = torch.optim.AdamW(gpt.model.parameters(), lr=config.lr)
        gpt.model.position_bias_to_device()
    
    gpt.optimizer = torch.optim.AdamW(gpt.model.parameters(), lr=config.lr)
    # gpt.mask.to(device)
    gpt.init_ema_params()

    if config.ddp or config.dp:
        print_model_size(gpt.model.module)
    else:   
        print_model_size(gpt.model)

    iteration = config.start_iter

    while iteration < config.total_steps:
        # torch.manual_seed(iteration + random.randint(0, 100000))
        iteration, gpt = train(iteration, gpt, train_loader, test_loader, sampler, config, device)   

    dist.destroy_process_group()     
    

def train_step(batch, gpt, device, step):
    gpt.train()

    batch = {k: v.to(device) for k, v in batch.items()}

    gpt.optimizer.zero_grad()

    loss = gpt.loss(batch)

    # Backward pass and optimize
    loss["loss"].backward()
    gpt.optimizer.step()

    # Update EMA parameters if enabled
    if config.ema:
        decay = config.ema if step > 0 else 0.0
        with torch.no_grad():
            gpt.update_ema(decay)
    return gpt, loss

def train(iteration, gpt, train_loader, test_loader, sampler, config, device):
    progress = ProgressMeter(
        config.total_steps,
        ['time', 'data'] + gpt.metrics
    )

    end = time.time()
    for batch in train_loader:
        batch_size = batch[list(batch.keys())[0]].shape[0]
        # print(batch[list(batch.keys())[0]].shape) # [64, 16, 64, 64, 3]
        if iteration % config.viz_interval == 0:
            visualize(sampler, iteration, gpt, test_loader)

        progress.update(data=time.time() - end)

        with torch.no_grad():
            batch = gpt.ae.prepare_batch(batch)
        # print(batch.keys())
        gpt, return_dict = train_step(batch=batch, gpt=gpt, device=device, step=iteration)

        gpt.scheduler.step()
        # gpt.scheduler.step(iteration)
        metrics = gpt.metrics
        
        metrics = {k: return_dict[k].detach().cpu().numpy().mean() for k in metrics}
        metrics = {k: v.astype(np.float32) for k, v in metrics.items()}
        progress.update(n=batch_size, **{k: v for k, v in metrics.items()})

        # if is_master_process:
        scheduler = gpt.scheduler
        wandb.log({'train/lr': scheduler.get_last_lr()[0]}, step=iteration)
        wandb.log({**{f'train/{metric}': val
                    for metric, val in metrics.items()}
                }, step=iteration)

        progress.update(time=time.time() - end)
        end = time.time()

        if iteration % config.log_interval == 0:
            progress.display(iteration)

        if iteration % config.test_interval == 0:
            val_loss = validate(iteration, gpt, test_loader, device)
            # is_best = val_loss < config.best_loss
            config.best_loss = min(config.best_loss, val_loss)
    
        
        if iteration % config.save_interval == 0 and iteration > config.start_iter: # and is_best:
            save_path = os.path.join(ckpt_dir, f'checkpoint_{iteration}.pth')
            torch.save({
                'iteration': iteration,
                'model_state_dict': gpt.model.module.state_dict() if config.ddp or config.dp else gpt.model.state_dict(),
                'optimizer_state_dict': gpt.optimizer.state_dict()
            }, save_path)
            print('Saved checkpoint to', save_path)
            print('Saved checkpoint at iteration', iteration)

        # if iteration % config.viz_interval == 0 or \
        # iteration % config.test_interval == 0 or \
        # iteration % config.save_interval == 0 or \
        # iteration >= config.total_steps:
        #     return iteration, gpt, rngs

        iteration += 1
    
    return iteration, gpt


def val_step(batch, gpt, device):
    gpt.eval()

    batch = {k: v.to(device) for k, v in batch.items()}

    with torch.no_grad():
        # Forward pass
        loss = gpt.loss(batch, training=False)

        # If using Distributed Data Parallel, the averaging across devices is handled automatically.
        # If not, and you need to average outputs across devices, you would need to do it manually here.

    return loss


def validate(iteration, gpt, test_loader, device):
    metrics = gpt.metrics
    
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
            batch = gpt.ae.prepare_batch(batch)
        return_dict = val_step(batch=batch, gpt=gpt, device=device)
        
        metrics = {k: return_dict[k].detach().cpu().numpy().mean() for k in metrics}
        metrics = {k: v.astype(np.float32) for k, v in metrics.items()}
        progress.update(n=batch_size, **{k: v for k, v in metrics.items()})
        progress.update(time=time.time() - end)
        end = time.time()

        # if i % config.log_eval_interval == 0:
        #     progress.display(i)

    progress.display(i)
    metrics = {metric: progress.meters[metric].avg for metric in metrics}

    # if is_master_process:
    # if rank == 0:
    wandb.log({**{f'val/{metric}': val
                    for metric, val in metrics.items()}
                }, step=iteration)
                  
    return metrics['loss']

def copy_gpt_weight(dp_gpt, target_gpt):
    target_gpt.ae.ae.load_state_dict(dp_gpt.ae.ae.module.state_dict())
    target_gpt.model.load_state_dict(dp_gpt.model.module.state_dict())

def visualize(sampler, iteration, gpt, test_loader):
    batch = next(iter(test_loader))
    real = batch['video'].detach().cpu().numpy()
    labels = batch['label'].detach().cpu().numpy()
    # print(label.shape)
    # print(label)
    str_labels = [config.rev_class_map[l] for l in labels]
    print(str_labels)

    # print(video.shape) # (batch_size, seq_len, height, width, channels)
    # if len(video.shape) == 5: # NBTHW
    #     video = gpt.ae.decode(video)
    # variables = {'params': model.ema_params if hasattr(model, 'ema_params') else model.parameters()}
    copy_gpt_weight(gpt, sampler.model)
    # sampler.model = gpt

    ws = gpt.config.num_device

    if ws > 1:
        split_batches = [{} for _ in range(ws)]
        # Iterate through each item in the batch
        for key, value in batch.items():
            # Split the tensor into k parts along the batch dimension
            chunks = torch.chunk(value, ws, dim=0)
            
            # Distribute the chunks into the corresponding new dictionaries
            for i, chunk in enumerate(chunks):
                split_batches[i][key] = chunk
        
        sample_list = []
        for b in split_batches:
            sample_list.append(sampler(b))
        samples = np.concatenate(sample_list, axis=0)

    else:
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
    # if is_master_process:
    
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

    args.run_id = args.output_dir.split('/')[-1] + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    world_size = torch.cuda.device_count()

    if torch.cuda.is_available():
        # if dist.is_available() and dist.is_initialized():
        #     # If using PyTorch Distributed Data Parallel (DDP)
        #     print(f'PyTorch process: {dist.get_rank()} / {dist.get_world_size()}')
        # else:
        #     # If not using DDP, the concept of process index/count doesn't directly apply
        #     print('PyTorch process: N/A (not using Distributed Data Parallel)')
        
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
    
    # is_master_process = dist.get_rank() == 0

    main(config)