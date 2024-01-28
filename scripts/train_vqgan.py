import os
import os.path as osp
import pathlib
import sys
import re
import glob
from unittest.mock import NonCallableMagicMock
import numpy as np
import time
import argparse
import yaml
import pickle
import wandb
import random
from datetime import datetime


import torch
import torch.distributed as dist


# import jax
# import jax.numpy as jnp
# from flax.training import checkpoints
# from flax import jax_utils

directory = pathlib.Path(__file__).resolve()
directory = directory.parent
sys.path.append(str(directory.parent))

from viper_rl.videogpt.loss_vqgan import VQPerceptualWithDiscriminator
from viper_rl.videogpt.data import load_dataset, prepare
from viper_rl.videogpt.train_utils import init_model_state_vqgan, ProgressMeter, save_image_grid, get_first_device, print_model_size
from viper_rl.videogpt import weight_init


def extract_iteration(filename):
    match = re.search(r"checkpoint_(\d+).pth", filename)
    return int(match.group(1)) if match else 0

def main():
    # print("The world size is {}".format(dist.get_world_size()))
    seed = config.seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    world_size = torch.cuda.device_count()
    rank = 0
    
    if config.ddp:
        dist.init_process_group(backend='nccl')
        # rank = dist.get_rank()
        print(f"Start running basic DDP on rank {rank}.")
        dist.init_process_group(backend='gloo', rank=rank, world_size=world_size, timeout=datetime.timedelta(minutes=5))
        # create model and move it to GPU with id rank
        device = rank % world_size
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.cuda.manual_seed_all(config.seed)

    
    global model
    global ckpt_dir

    num_device = torch.cuda.device_count()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # rng = jax.random.PRNGKey(config.seed)
    # rng, init_rng = jax.random.split(rng)

    config.ckpt = config.output_dir if osp.exists(config.output_dir) else None
    # config.ckpt = None
    ckpt_dir = osp.join(config.output_dir, 'checkpoints')
    
    wandb.init(project='vqgan', config=config,
                id=config.run_id, resume='allow', mode='online')
    wandb.run.name = config.run_id
    wandb.run.save()
    
    model = VQPerceptualWithDiscriminator(config)
    print_model_size(model.vqgan, name='vqgan')
    print_model_size(model.disc, name='disc')


    config.device = device
    config.ae['device'] = device

    train_dataset, _, mask_map = load_dataset(config, train=True, modality='image')
    test_dataset, _, _ = load_dataset(config, train=False, modality='image')

    train_loader = prepare(train_dataset, config.batch_size, world_size, rank, ddp=config.ddp)
    test_loader = prepare(test_dataset, config.batch_size, world_size, rank, ddp=config.ddp)

    if mask_map is not None:
        pickle.dump(mask_map, open(osp.join(config.output_dir, 'mask_map.pkl'), 'wb'))
    
    
    # batch = next(iter(train_loader))
    # print(batch.keys())
    # batch = get_first_device(batch)

    state = init_model_state_vqgan(model, config)
    state.model.vqgan.apply(weight_init)
    state.model.disc.apply(weight_init)
    state.model.use_device(device)
    
    start_iteration = 0

    if config.ckpt is not None:
        model_files = glob.glob(f"{ckpt_dir}/*.pth")
        if len(model_files):
            checkpoint_path = sorted(model_files, key=extract_iteration)[-1]
            checkpoint = torch.load(checkpoint_path)
            state.model.vqgan.load_state_dict(checkpoint['model_state_dict'])
            state.G_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            state.model.disc.load_state_dict(checkpoint['disc_state_dict'])
            state.D_optimizer.load_state_dict(checkpoint['D_optimizer_state_dict'])
            start_iteration = checkpoint['iteration']
            print(f'Restored from checkpoint {os.path.join(checkpoint_path)}, at iteration {start_iteration}')

    

    # Randomize RNG so we get different rngs when restarting after preemptions
    # Otherwise we get the same sequence of noise
    iteration = start_iteration

    while iteration < config.total_steps:
        # Randomize RNG
        torch.manual_seed(iteration + random.randint(0, 100000))
        
        # Training function
        iteration, state = train(iteration, state, train_loader, test_loader, config)
        
        # Visualization
        # if iteration % config.viz_interval == 0:
        #     visualize(iteration, state, test_loader, device)

        # Update learning rate
        
def train_step(batch, state, device):
    # Assuming 'model' includes both VQGAN and discriminator sub-models
    state.model.train()  # Set the model to training mode

    # Move batch to the same device as the model
    batch = {k: v.to(device) for k, v in batch.items()}

    # Generator update
    state.G_optimizer.zero_grad()
    loss_G, aux_G = state.model.loss_G(batch)
    loss_G.backward()  # Backpropagation to calculate gradients
    state.G_optimizer.step()  # Update VQGAN parameters using its optimizer
    # state.G_scheduler.step()

    # Discriminator update
    state.D_optimizer.zero_grad()
    loss_D, aux_D = state.model.loss_D(batch)
    loss_D.backward()  # Backpropagation to calculate gradients
    state.D_optimizer.step()  # Update discriminator parameters using its optimizer
    # state.D_scheduler.step()

    # Combine auxiliary outputs
    aux = {**aux_G, **aux_D}

    return state, aux


def train(iteration, state, train_loader, test_loader, config):
    progress = ProgressMeter(
        config.total_steps,
        ['time', 'data'] + state.model.metrics
    )

    end = time.time()
    for batch in train_loader:
        # batch = next(train_loader)
        batch_size = batch['image'].shape[0]
        progress.update(data=time.time() - end)

        # Visualization
        if iteration % config.viz_interval == 0:
            visualize(iteration, state, test_loader, config.device)

        state, metrics = train_step(batch, state, config.device)

        state.G_scheduler.step(iteration)
        state.D_scheduler.step(iteration)

        metrics = {k: metrics[k].detach().cpu().numpy().mean() for k in state.model.metrics}
        metrics = {k: v.astype(np.float32) for k, v in metrics.items()}
        progress.update(n=batch_size, **{k: v for k, v in metrics.items()})

        wandb.log({'train/lr': state.G_scheduler.get_last_lr()[0]}, step=iteration)
        wandb.log({**{f'train/{metric}': val
                    for metric, val in metrics.items()}
                }, step=iteration)

        progress.update(time=time.time() - end)
        end = time.time()

        # Checkpoint saving
        if iteration % config.save_interval == 0:
            save_path = os.path.join(ckpt_dir, f'checkpoint_{iteration}.pth')
            torch.save({
                'iteration': iteration,
                'model_state_dict': state.model.vqgan.state_dict(),
                'optimizer_state_dict': state.G_optimizer.state_dict(),
                'disc_state_dict': state.model.disc.state_dict(),
                'D_optimizer_state_dict': state.D_optimizer.state_dict(),
            }, save_path)
            print('Saved checkpoint to', save_path)
            print('Saved checkpoint at iteration', iteration)

        if iteration % config.log_interval == 0:
            progress.display(iteration)

        # if iteration % config.save_interval == 0 or \
        # iteration % config.viz_interval == 0 or \
        # iteration >= config.total_steps:
        #     return iteration, state

        iteration += 1

    return iteration, state

        
def viz_step(images, state):
    # Assuming batch is a dictionary with 'image' tensor

    # Perform forward pass (reconstruction) - no need for gradient tracking
    with torch.no_grad():
        recon = state.model.vqgan.reconstruct(images)

    # Clip the reconstructed images to the range [-1, 1]
    recon = torch.clamp(recon, -1, 1)

    return recon


def visualize(iteration, state, test_loader, device):
    # Fetch a batch of data
    batch = next(iter(test_loader))
    images = batch['image'].to(device)

    # Perform reconstruction using the model
    state.model.eval()  # Set the model to evaluation mode
    # with torch.no_grad():
    recon = viz_step(images, state)  # Replace 'viz_step' with your model's method

    # Prepare images for visualization
    images_np = images.detach().cpu().numpy() # .reshape(-1, *images.shape[2:])
    recon_np = recon.detach().cpu().numpy() # .reshape(-1, *recon.shape[2:])
    viz = np.stack((recon_np, images_np), axis=1)
    
    viz = viz.reshape(-1, *viz.shape[2:]) * 0.5 + 0.5  # Adjusting range if necessary
    viz = np.transpose(viz, (0, 2, 3, 1)) # .astype(np.int32)

    # Save image grid and log to WandB
    viz_image = save_image_grid(viz)  # Define save_image_grid as needed
    viz_wandb = wandb.Image(viz_image)

    # Log visualization in WandB
    wandb.log({'eval/recon': viz_wandb}, step=iteration)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_dir', type=str, required=True)
    parser.add_argument('-c', '--config', type=str, required=True)
    args = parser.parse_args()

    args.run_id = args.output_dir.split('/')[-1] + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    if torch.cuda.is_available():
        print(f'Total CPUs: {os.cpu_count()}')
        print(f'Total CUDA devices: {torch.cuda.device_count()}')
        print(f'Current CUDA device index: {torch.cuda.current_device()}')
        print(f'Current CUDA device name: {torch.cuda.get_device_name(torch.cuda.current_device())}')
    else:
        print('CUDA is not available. Using CPU only.')

    config = yaml.safe_load(open(args.config, 'r'))
    if os.environ.get('DEBUG') == '1':
        config['save_interval'] = 10
        config['viz_interval'] = 10
        config['log_interval'] = 1
        args.output_dir = osp.join(osp.dirname(args.output_dir), f'DEBUG_{osp.basename(args.output_dir)}')
        args.run_id = f'DEBUG_{args.run_id}'
    
    print(f"Logging to {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    args_d = vars(args)
    args_d.update(config)
    pickle.dump(args, open(osp.join(args.output_dir, 'args'), 'wb'))
    config = args

    # is_master_process = dist.get_rank() == 0

    main()