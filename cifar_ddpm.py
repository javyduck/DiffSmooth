import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), './diffusion'))
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

def cifar_model_and_diffusion_defaults():
    """
    Defaults for cifar ddpm.
    """
    return dict(
        image_size=32,
        num_channels=128,
        num_res_blocks=3,
        num_heads=4,
        num_heads_upsample=-1,
        attention_resolutions="16,8",
        dropout=0.3,
        learn_sigma=True,
        sigma_small=False,
        class_cond=False,
        diffusion_steps=4000,
        noise_schedule="cosine",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=True,
        rescale_learned_sigmas=True,
        use_checkpoint=False,
        use_scale_shift_norm=True,
    )

def create_argparser(ddpm_path):
    defaults = dict(
        clip_denoised=True,
        num_samples=16,
        batch_size=16,
        use_ddim=False,
        model_path=ddpm_path,
    )
    defaults.update(cifar_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser(description='default ddpm')
    add_dict_to_argparser(parser, defaults)
    return parser

class cifar_ddpm(nn.Module):
    def __init__(self, sigma, ddpm_path = 'pretrained_models/cifar10_uncond_50M_500K.pt', device = 'cuda'):
        super(cifar_ddpm,self).__init__()
        self.ddpm_path = ddpm_path
        self.device = device
        self.args = create_argparser(self.ddpm_path).parse_args([])
        self.model, self.diffusion = self.get_model()
        self.sigma = sigma
        self.t = self.get_t(self.sigma)
        self.sqrt_alpha_t = self.diffusion.sqrt_alphas_cumprod[self.t]
        
    @torch.no_grad()
    def forward(self,x):
        ## denoise with one step, currently, I can not run more steps owing to the constrain of the resource.. ##
        ## notice, the range of the input is [0,1] ##
        x = x * 2 - 1 
        t = torch.tensor([self.t]).repeat(x.shape[0]).to(x.device)
        x = self.sqrt_alpha_t * x
        sample = self.diffusion.p_sample(
            self.model,
            x,
            t,
            clip_denoised=True,
        )['pred_xstart']
        return (sample+1)/2
    
    def get_model(self):
        model, diffusion = create_model_and_diffusion(
        **args_to_dict(self.args, model_and_diffusion_defaults().keys())
        )
        model.load_state_dict(
            dist_util.load_state_dict(self.args.model_path, map_location="cpu")
        )
        model.to(self.device)
        model.eval()
        return model, diffusion
    
    def get_t(self, sigma):
        ratio = self.diffusion.sqrt_one_minus_alphas_cumprod/self.diffusion.sqrt_alphas_cumprod
        for i in range(len(ratio)):
            if ratio[i] >= (2 * sigma):
                return i
