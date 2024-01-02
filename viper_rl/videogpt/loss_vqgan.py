import torch
from .models import VQGAN, StyleGANDisc
import lpips

from .train_utils import get_optimizer



class VQPerceptualWithDiscriminator:
    def __init__(self, config):
        self.config = config
        self.vqgan = VQGAN(config.image_size, **config.ae)
        self.disc = StyleGANDisc(config.image_size, **config.disc)
        self.lpips = lpips.LPIPS(net='vgg')
        self.G_optimizer, self.G_scheduler = get_optimizer(self.vqgan, config)
        self.D_optimizer, self.D_scheduler = get_optimizer(self.disc, config)

    @property
    def metrics(self):
        return ['loss_G', 'loss_D'] + self.vqgan.metrics + \
            self.disc.metrics + ['d_weight']
    
    def use_device(self, device):
        self.vqgan = self.vqgan.to(device)
        self.disc = self.disc.to(device)
        self.lpips = self.lpips.to(device)
    
    def train(self):
        self.vqgan.train()
        self.disc.train()
    
    def eval(self):
        self.vqgan.eval()
        self.disc.eval()

    def loss_recon(self, batch):
        # In PyTorch, randomness is handled by the framework, especially for dropout
        out = self.vqgan(batch['image'])
        # print(batch['image'].shape)
        # print(out['recon'].shape)
        rec_loss = torch.abs(batch['image'] - out['recon'])
        if self.config.perceptual_weight > 0:
            p_loss = self.lpips(batch['image'], out['recon'])
        else:
            p_loss = torch.tensor(0.0, device=batch['image'].device)

        total_loss = rec_loss + self.config.perceptual_weight * p_loss
        out['ae_loss'] = total_loss.mean()
        return out

    def loss_G(self, batch):
        vqgan_out = self.loss_recon(batch)
        # print(vqgan_out)

        g_out = self.disc.loss_G(vqgan_out['recon'])

        d_weight = torch.tensor(self.config.disc_weight, dtype=torch.float32)

        loss = vqgan_out['ae_loss'] + d_weight * g_out['g_loss'] + \
            self.config.codebook_weight * vqgan_out['vq_loss']

        return loss, {
            'loss_G': loss,
            'd_weight': d_weight,
            **{k: vqgan_out[k] for k in self.vqgan.metrics},
            **g_out, 
        }


    def loss_D(self, batch):
        out = self.vqgan(batch['image'])
        d_out = self.disc.loss_D(batch['image'], out['recon'])

        return d_out['loss_D'], d_out