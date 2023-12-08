"""Extract features for temporal action detection datasets"""
import argparse
import os
import random

import numpy as np
import torch
from torch import nn
from timm.models import create_model
from torchvision import transforms

# NOTE: Do not comment `import models`, it is used to register models
import VideoMAEv2.models as models  # noqa: F401
from VideoMAEv2.dataset.loader import get_video_loader


def to_normalized_float_tensor(vid):
    return vid.permute(3, 0, 1, 2).to(torch.float32) / 255


# NOTE: for those functions, which generally expect mini-batches, we keep them
# as non-minibatch so that they are applied as if they were 4d (thus image).
# this way, we only apply the transformation in the spatial domain
def resize(vid, size, interpolation='bilinear'):
    # NOTE: using bilinear interpolation because we don't work on minibatches
    # at this level
    scale = None
    if isinstance(size, int):
        scale = float(size) / min(vid.shape[-2:])
        size = None
    return torch.nn.functional.interpolate(
        vid,
        size=size,
        scale_factor=scale,
        mode=interpolation,
        align_corners=False)


class ToFloatTensorInZeroOne(object):

    def __call__(self, vid):
        return to_normalized_float_tensor(vid)


class Resize(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, vid):
        return resize(vid, self.size)


class VMAEEncoder(nn.Module):
    def __init__(self, model_name="vit_giant_patch14_224", num_frames=5, img_size=224, 
                pretrained=False, num_classes=710, tubelet_size=2, 
                drop_path_rate=0.1, use_mean_pooling=True, ckpt_path="logdir/vit_g_hybrid_pt_1200e_k710_ft.pth"):
        super(VMAEEncoder, self).__init__()
        self.transform = transforms.Compose(
                [ToFloatTensorInZeroOne(),
                Resize((img_size, img_size))])
        
        self.model = create_model(
        model_name,
        img_size=img_size,
        pretrained=pretrained,
        num_classes=num_classes,
        all_frames=num_frames,
        tubelet_size=tubelet_size,
        drop_path_rate=drop_path_rate,
        use_mean_pooling=use_mean_pooling)
        ckpt = torch.load(ckpt_path, map_location='cpu')
        for model_key in ['model', 'module']:
            if model_key in ckpt:
                ckpt = ckpt[model_key]
                break
        self.model.load_state_dict(ckpt)
        self.model.eval()
        self.model.cuda()
        self.name = "videomae"
        self.outdim = 1408

    def forward(self, frames):
        x = torch.from_numpy(frames)
        # print(x.shape)
        frame_q = self.transform(x)  
        input_data = frame_q.unsqueeze(0).cuda()

        with torch.no_grad():
            features = self.model.forward_features(input_data)

        return features.detach().cpu().numpy()



def get_args():
    parser = argparse.ArgumentParser(
        'Extract TAD features using the videomae model', add_help=False)
    # parser.add_argument('--batch_size', default=64, type=int)
    # parser.add_argument('--epochs', default=300, type=int)
    # parser.add_argument('--save_ckpt_freq', default=50, type=int)

    # Model parameters
    parser.add_argument(
        '--model',
        # default='pretrain_videomae_base_patch16_224',
        default='vit_giant_patch14_224',
        type=str,
        metavar='MODEL',
        help='Name of model to train')
    
    
    parser.add_argument('--tubelet_size', type=int, default=2)
    parser.add_argument(
        '--with_checkpoint', action='store_true', default=False)

    parser.add_argument(
        '--decoder_depth', default=4, type=int, help='depth of decoder')

    parser.add_argument(
        '--mask_type',
        default='tube',
        choices=['random', 'tube'],
        type=str,
        help='encoder masked strategy')
    parser.add_argument(
        '--decoder_mask_type',
        default='run_cell',
        choices=['random', 'run_cell'],
        type=str,
        help='decoder masked strategy')

    parser.add_argument(
        '--mask_ratio', default=0.9, type=float, help='mask ratio of encoder')
    parser.add_argument(
        '--decoder_mask_ratio',
        default=0.0,
        type=float,
        help='mask ratio of decoder')

    parser.add_argument(
        '--input_size',
        default=224,
        type=int,
        help='images input size for backbone')

    parser.add_argument(
        '--drop_path',
        type=float,
        default=0.1,
        metavar='PCT',
        help='Drop path rate (default: 0.1)')

    parser.add_argument(
        '--normlize_target',
        default=True,
        type=bool,
        help='normalized the target patch pixels')

    # Optimizer parameters
    parser.add_argument(
        '--opt',
        default='adamw',
        type=str,
        metavar='OPTIMIZER',
        help='Optimizer (default: "adamw"')
    parser.add_argument(
        '--opt_eps',
        default=1e-8,
        type=float,
        metavar='EPSILON',
        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument(
        '--opt_betas',
        default=None,
        type=float,
        nargs='+',
        metavar='BETA',
        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument(
        '--clip_grad',
        type=float,
        default=None,
        metavar='NORM',
        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument(
        '--momentum',
        type=float,
        default=0.9,
        metavar='M',
        help='SGD momentum (default: 0.9)')
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0.05,
        help='weight decay (default: 0.05)')
    parser.add_argument(
        '--weight_decay_end',
        type=float,
        default=None,
        help="""Final value of the
        weight decay. We use a cosine schedule for WD. 
        (Set the same value with args.weight_decay to keep weight decay no change)"""
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=1.5e-4,
        metavar='LR',
        help='learning rate (default: 1.5e-4)')
    parser.add_argument(
        '--warmup_lr',
        type=float,
        default=1e-6,
        metavar='LR',
        help='warmup learning rate (default: 1e-6)')
    parser.add_argument(
        '--min_lr',
        type=float,
        default=1e-5,
        metavar='LR',
        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument(
        '--warmup_epochs',
        type=int,
        default=40,
        metavar='N',
        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument(
        '--warmup_steps',
        type=int,
        default=-1,
        metavar='N',
        help='epochs to warmup LR, if scheduler supports')

    # Augmentation parameters
    parser.add_argument(
        '--color_jitter',
        type=float,
        default=0.0,
        metavar='PCT',
        help='Color jitter factor (default: 0.4)')
    parser.add_argument(
        '--train_interpolation',
        type=str,
        default='bicubic',
        choices=['random', 'bilinear', 'bicubic'],
        help='Training interpolation')

    # * Finetuning params
    parser.add_argument(
        '--finetune', default='', help='finetune from checkpoint')

    # Dataset parameters
    parser.add_argument(
        '--data_path',
        default='../yt_videos',
        type=str,
        help='dataset path')
    parser.add_argument(
        '--save_path',
        default='logdir/yt_videos/th14_vit_g_16_4',
        type=str,
        help='path for saving features')
    parser.add_argument(
        '--ckpt_path',
        default='logdir/vit_g_hybrid_pt_1200e_k710_ft.pth',
        help='load from checkpoint')

    parser.add_argument(
        '--data_root', default='', type=str, help='dataset path root')
    parser.add_argument(
        '--fname_tmpl',
        default='img_{:05}.jpg',
        type=str,
        help='filename_tmpl for rawframe data')
    parser.add_argument(
        '--imagenet_default_mean_and_std', default=True, action='store_true')
    parser.add_argument('--num_frames', type=int, default=8)
    parser.add_argument('--sampling_rate', type=int, default=4)
    parser.add_argument('--num_sample', type=int, default=1)
    parser.add_argument(
        '--output_dir',
        default='logdir',
        help='path where to load model weights')
    parser.add_argument(
        '--log_dir', default=None, help='path where to tensorboard log')
    parser.add_argument(
        '--device',
        default='cuda',
        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument(
        '--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument(
        '--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument(
        '--pin_mem',
        action='store_true',
        help=
        'Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.'
    )
    parser.add_argument(
        '--no_pin_mem', action='store_false', dest='pin_mem', help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument(
        '--world_size',
        default=1,
        type=int,
        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument(
        '--dist_url',
        default='env://',
        help='url used to set up distributed training')
    
    # parser.add_argument(
    #     '--seq_dim',
    #     default=32,
    #     help='dimension of final logits')
    # parser.add_argument(
    #     '--queries_dim',
    #     default=32,
    #     help='dimension of final logits')
    # parser.add_argument(
    #     '--logits_dim',
    #     default=100,
    #     help='dimension of final logits')
    # parser.add_argument(
    #     '--pio_depth',
    #     default=6,
    #     help='dimension of final logits')
        
    # parser.add_argument(
    #     '--num_latents',
    #     default=256,
    #     help='dimension of final logits')
    # parser.add_argument(
    #     '--cross_heads',
    #     default=1,
    #     help='number of heads for cross attention. paper said 1')
    # parser.add_argument(
    #     '--latent_heads',
    #     default=1,
    #     help='number of heads for latent self attention, 8')
    # parser.add_argument(
    #     '--cross_dim_head',
    #     default=64,
    #     help='number of dimensions per cross attention head')
    # parser.add_argument(
    #     '--latent_dim',
    #     default=512,
    #     help='number of dimensions per latent self attention head')
    # parser.add_argument(
    #     '--latent_dim_head',
    #     default=64,
    #     help='number of dimensions per latent self attention head')
    # parser.add_argument(
    #     '--weight_tie_layers',
    #     default=False,
    #     help='whether to weight tie layers (optional, as indicated in the diagram)')
    # parser.add_argument(
    #     '--seq_dropout_prob',
    #     default=0.2,
    #     help='fraction of the tokens from the input sequence to dropout (structured dropout, for saving compute and regularizing effects)')

    return parser.parse_args()


def get_start_idx_range(data_set):

    def thumos14_range(num_frames):
        return range(0, num_frames - 15, 4)

    def fineaction_range(num_frames):
        return range(0, num_frames - 15, 16)

    if data_set == 'THUMOS14':
        return thumos14_range
    elif data_set == 'FINEACTION':
        return fineaction_range
    else:
        raise NotImplementedError()


def extract_feature(args):
    # preparation
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    video_loader = get_video_loader()
    # start_idx_range = get_start_idx_range(args.data_set)
    transform = transforms.Compose(
        [ToFloatTensorInZeroOne(),
         Resize((224, 224))])

    # get video path
    vid_list = os.listdir(args.data_path)
    random.shuffle(vid_list)

    # get model & load ckpt
    model = create_model(
        args.model,
        img_size=224,
        pretrained=False,
        num_classes=710,
        all_frames=args.num_frames,
        tubelet_size=args.tubelet_size,
        drop_path_rate=args.drop_path,
        use_mean_pooling=True)
    ckpt = torch.load(args.ckpt_path, map_location='cpu')
    for model_key in ['model', 'module']:
        if model_key in ckpt:
            ckpt = ckpt[model_key]
            break
    model.load_state_dict(ckpt)
    model.eval()
    model.cuda()

    # extract feature
    num_videos = len(vid_list)
    for idx, vid_name in enumerate(vid_list):
        url = os.path.join(args.save_path, vid_name.split('.')[0] + '.npy')
        # if os.path.exists(url):
        #     continue

        video_path = os.path.join(args.data_path, vid_name)
        vr = video_loader(video_path)
        total_frames = len(vr)//args.num_frames*args.num_frames

        feature_list = []
        # for start_idx in start_idx_range(len(vr)):
        for start_idx in range(0, total_frames, args.num_frames):
            data = vr.get_batch(np.arange(start_idx, start_idx + args.num_frames)).asnumpy()
            frame = torch.from_numpy(data)  # torch.Size([8, 566, 320, 3])
            frame_q = transform(frame)  # torch.Size([3, 8, 224, 224])
            input_data = frame_q.unsqueeze(0).cuda()

            with torch.no_grad():
                feature = model.forward_features(input_data)
                print(feature.shape)
                feature_list.append(feature.cpu().numpy())

        # [N, C]
        np.save(url, np.vstack(feature_list))
        print(f'[{idx} / {num_videos}]: save feature on {url}')


if __name__ == '__main__':
    args = get_args()
    extract_feature(args)