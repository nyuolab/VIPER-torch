from typing import Any
from collections import OrderedDict
import numpy as np
import math
import tempfile
from PIL import Image
import ffmpeg
from moviepy.editor import ImageSequenceClip
import torch
import lpips


class TrainStateEMA:
    def __init__(self, model, optimizer, ema_decay):
        self.model = model
        self.optimizer = optimizer
        self.ema_params = OrderedDict(model.named_parameters())
        self.ema_decay = ema_decay
        self.ema_params = self._initialize_ema_params(model)
    
    def _initialize_ema_params(self, model):
        # Initialize EMA parameters to be the same as the model's parameters
        ema_params = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                ema_params[name] = param.data.clone().detach()
        return ema_params

    def update_ema(self, ema_decay=None):
        if not ema_decay:
            ema_decay = self.ema_decay

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.ema_params[name].mul_(ema_decay).add_(param.data, alpha=1 - ema_decay)

class TrainStateVQ:
    def __init__(self, model, config):
        self.step = 0
        self.model = model
        self.vqgan_model = model.vqgan
        self.disc_model = model.disc
        self.lpips_model = model.lpips
        self.G_optimizer, self.G_scheduler = get_optimizer(self.vqgan_model, config)
        self.D_optimizer, self.D_scheduler = get_optimizer(self.disc_model, config)
        

    # def apply_vqgan_gradients(self, vqgan_grads):
    #     # Assume vqgan_grads is a dictionary of gradients for VQGAN model parameters
    #     for name, param in self.vqgan_model.named_parameters():
    #         if name in vqgan_grads:
    #             param.grad = vqgan_grads[name].detach()
    #     self.vqgan_optimizer.step()
    #     self.vqgan_optimizer.zero_grad()
    #     self.step += 1

    # def apply_disc_gradients(self, disc_grads):
    #     # Similar approach for discriminator
    #     for name, param in self.disc_model.named_parameters():
    #         if name in disc_grads:
    #             param.grad = disc_grads[name].detach()
    #     self.disc_optimizer.step()
    #     self.disc_optimizer.zero_grad()

    # @classmethod
    # def create(self, cls, vqgan_model, disc_model, lpips_model, vqgan_optimizer, disc_optimizer):
    #     return cls(vqgan_model, disc_model, lpips_model, vqgan_optimizer, disc_optimizer)

def torch_tree_map(fn, tree):
    if isinstance(tree, dict):
        return {k: torch_tree_map(fn, v) for k, v in tree.items()}
    elif isinstance(tree, list):
        return [torch_tree_map(fn, v) for v in tree]
    else:
        return fn(tree)

def get_first_device(pytree):
    def get_first_and_to_cpu(tensor):
        if isinstance(tensor, torch.Tensor):
            return tensor[0].cpu()
        else:
            return tensor

    return torch_tree_map(get_first_and_to_cpu, pytree)

def n2str(x):
    suffix = ''
    if x > 1e9:
        x /= 1e9
        suffix = 'B'
    elif x > 1e6:
        x /= 1e6
        suffix = 'M'
    elif x > 1e3:
        x /= 1e3
        suffix = 'K'
    return f'{x:.2f}{suffix}'

def print_model_size(model, name=''):
    total_params = sum(p.numel() for p in model.parameters())
    if name:
        print(f'{name} parameter count:', total_params)
    else:
        print('model parameter count:', total_params)

def get_optimizer(model, config):
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: min((epoch + 1) / config.warmup_steps, 1)
    )
    return optimizer, scheduler

def init_model_state_videogpt(model, batch, config):
    # Initialize model parameters (assuming model is an instance of nn.Module)
    # In PyTorch, model parameters are initialized when the model is created
    print_model_size(model)

    train_state = TrainStateEMA(
        model=model,
        optimizer=model.optimizer,
        ema_decay=config.ema_decay  # Assuming ema_decay is a config attribute
    )
    return train_state, model.scheduler

def init_model_state_vqgan(model, batch, config):
    # In PyTorch, models are initialized when they are created
    print_model_size(model.vqgan, name='vqgan')
    print_model_size(model.disc, name='disc')

    # Initialize LPIPS
    # lpips_model = lpips.LPIPS(net='vgg').to(batch['image'].device)

    # Initialize optimizers
    # optimizer, scheduler = get_optimizer(model.vqgan, config)
    # disc_optimizer, disc_scheduler = get_optimizer(model.disc, config)
    # lpips_optimizer, lpips_scheduler = get_optimizer(lpips_model, config)

    # Create and return the TrainStateVQ instance
    train_state = TrainStateVQ(
        model=model,
        config=config
    )
    return train_state #, optimizer, scheduler, disc_optimizer, disc_scheduler

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, total_iters, meter_names, prefix=""):
        self.iter_fmtstr = self._get_iter_fmtstr(total_iters)
        self.meters = OrderedDict({mn: AverageMeter(mn, ':6.3f')
                                   for mn in meter_names})
        self.prefix = prefix

    def update(self, n=1, **kwargs):
        for k, v in kwargs.items():
            self.meters[k].update(v, n=n)

    def display(self, iteration):
        entries = [self.prefix + self.iter_fmtstr.format(iteration)]
        entries += [str(meter) for meter in self.meters.values()]
        print('\t'.join(entries))

    def _get_iter_fmtstr(self, total_iters):
        num_digits = len(str(total_iters // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(total_iters) + ']'

def save_video_grid(video, fname=None, nrow=None, fps=10):
    b, t, h, w, c = video.shape

    if nrow is None:
        nrow = math.ceil(math.sqrt(b))
        if b % nrow != 0:
            nrow = 8
    ncol = math.ceil(b / nrow)
    padding = 1
    new_h = (padding + h) * ncol + padding
    new_h += new_h % 2
    new_w = (padding + w) * nrow + padding
    new_w += new_w % 2
    video_grid = np.zeros((t, new_h, new_w, c), dtype='uint8')
    for i in range(b):
        r = i // nrow
        c = i % nrow

        start_r = (padding + h) * r
        start_c = (padding + w) * c
        video_grid[:, start_r:start_r + h, start_c:start_c + w] = video[i]

    if fname is not None:
        clip = ImageSequenceClip(list(video_grid), fps=fps)
        clip.write_gif(fname, fps=fps)
        print('saved videos to', fname)
    
    return video_grid # THWC, uint8

    
def save_video(video, fname=None, fps=10):
    # video: TCHW, uint8
    T, H, W, C = video.shape
    if fname is None:
        fname = tempfile.NamedTemporaryFile().name + '.mp4'
    process = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{W}x{H}')
        .output(fname, pix_fmt='yuv420p', vcodec='libx264', r=fps)
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )

    for frame in video:
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    process.wait()
    print('Saved video to', fname)
    return fname
    

def add_border(video, color, width=0.025):
    # video: BTHWC in [0, 1]
    S = math.ceil(int(video.shape[3] * width))

    # top
    video[:, :, :S, :, 0] = color[0]
    video[:, :, :S, :, 1] = color[1]
    video[:, :, :S, :, 2] = color[2]

    # bottom
    video[:, :, -S:, :, 0] = color[0]
    video[:, :, -S:, :, 1] = color[1]
    video[:, :, -S:, :, 2] = color[2]

    # left
    video[:, :, :, :S, 0] = color[0]
    video[:, :, :, :S, 1] = color[1]
    video[:, :, :, :S, 2] = color[2]

    # right
    video[:, :, :, -S:, 0] = color[0]
    video[:, :, :, -S:, 1] = color[1]
    video[:, :, :, -S:, 2] = color[2]



def save_image_grid(images, fname=None, nrow=None):
    b, h, w, c = images.shape
    images = (images * 255).astype('uint8')

    if nrow is None:
        nrow = math.ceil(math.sqrt(b))
    ncol = math.ceil(b / nrow)
    padding = 1
    image_grid = np.zeros(((padding + h) * ncol + padding,
                          (padding + w) * nrow + padding, c), dtype='uint8')
    for i in range(b):
        r = i // nrow
        c = i % nrow

        start_r = (padding + h) * r
        start_c = (padding + w) * c
        image_grid[start_r:start_r + h, start_c:start_c + w] = images[i]

    if fname is not None:
        image = Image.fromarray(image_grid)
        image.save(fname)
        print('saved image to', fname)

    return image_grid