import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), './diffusion'))
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

def imagenet_model_and_diffusion_defaults():
    """
    Defaults for imagenet ddpm.
    """
    res = dict(
        image_size=256,
        num_channels=256,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=64,
        attention_resolutions="32,16,8",
        channel_mult="",
        dropout=0.0,
        class_cond=False,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_fp16=False,
        use_new_attention_order=False,
        learn_sigma=True,
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing= None,
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
    )
    return res

def create_argparser(gdm_path):
    defaults = dict(
        clip_denoised=True,
        num_samples=100,
        batch_size=4,
        use_ddim=False,
        model_path=gdm_path,
    )
    defaults.update(imagenet_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser(description='default ddpm')
    add_dict_to_argparser(parser, defaults)
    return parser

class imagenet_gdm(nn.Module):
    def __init__(self, sigma, gdm_path = 'pretrained_models/256x256_diffusion_uncond.pt', device = 'cuda'):
        super(imagenet_gdm,self).__init__()
        self.gdm_path = gdm_path
        self.device = device
        self.args = create_argparser(self.gdm_path).parse_args([])
        self.model, self.diffusion = self.get_model()
        self.sigma = sigma
        self.t = self.get_t(self.sigma)
        self.sqrt_alpha_t = self.diffusion.sqrt_alphas_cumprod[self.t]
        
    @torch.no_grad()
    def forward(self,x):
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
        if self.args.use_fp16:
            model.convert_to_fp16()
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