from archs.cifar_resnet import resnet as resnet_cifar
from datasets import get_normalize_layer, get_input_center_layer, get_vit_normalize_layer
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.nn.functional import interpolate
from torchvision.models.resnet import resnet50
from cifar_ddpm import cifar_ddpm
from imagenet_gdm import imagenet_gdm
from timm.models import create_model

# resnet50 - the classic ResNet-50, sized for ImageNet
# cifar_resnet20 - a 20-layer residual network sized for CIFAR
# cifar_resnet110 - a 110-layer residual network sized for CIFAR
ARCHITECTURES = ["resnet50", "cifar_resnet110", "imagenet32_resnet110", "beit_large_patch16_224"]

def get_architecture(arch: str, dataset: str) -> torch.nn.Module:
    """ Return a neural network (with random weights)

    :param arch: the architecture - should be in the ARCHITECTURES list above
    :param dataset: the dataset - should be in the datasets.DATASETS list
    :return: a Pytorch module
    """
    if arch == "resnet50" and dataset == "imagenet":
        model = torch.nn.DataParallel(resnet50(pretrained=False)).cuda()
        cudnn.benchmark = True
    elif arch == "cifar_resnet20":
        model = resnet_cifar(depth=20, num_classes=10).cuda()
    elif arch == "cifar_resnet110":
        model = resnet_cifar(depth=110, num_classes=10).cuda()
    elif arch == "imagenet32_resnet110":
        model = resnet_cifar(depth=110, num_classes=1000).cuda()
    elif arch == "beit_large_patch16_224" and dataset == "imagenet":
        model =  create_model(
                    'beit_large_patch16_224',
                    pretrained=False,
#                     num_classes=1000,
#                     drop_rate=0.0,
#                     drop_path_rate=0.4,
#                     attn_drop_rate=0.0,
#                     drop_block_rate=None,
#                     use_mean_pooling=True,
#                     init_scale=0.001,
#                     use_rel_pos_bias=True,
#                     use_abs_pos_emb=False,
#                     init_values=0.1,
                ).cuda()
        cudnn.benchmark = True
        normalize_layer = get_vit_normalize_layer()
        return torch.nn.Sequential(normalize_layer, model)  

    # Both layers work fine, We tried both, and they both
    # give very similar results 
    # IF YOU USE ONE OF THESE FOR TRAINING, MAKE SURE
    # TO USE THE SAME WHEN CERTIFYING.
    normalize_layer = get_normalize_layer(dataset)
    # normalize_layer = get_input_center_layer(dataset)
    return torch.nn.Sequential(normalize_layer, model)

def get_diffusion_model(noise_sd: float, dataset: str) -> torch.nn.Module:
    """ Return the diffusion model
    :param noise_sd: the smoothing noise
    :param dataset: the dataset - should be in the datasets.DATASETS list
    :return: a Pytorch module
    """
    if dataset == "cifar10":
        return cifar_ddpm(noise_sd, 'pretrained_models/cifar10_uncond_50M_500K.pt')
    elif dataset == "imagenet":
        return imagenet_gdm(noise_sd, 'pretrained_models/256x256_diffusion_uncond.pt')
    
class DiffSmooth_Classifier(nn.Module):
    """ Return the diffsmooth classifier
    :param dataset: the dataset - should be in the datasets.DATASETS list
    :param classifier: the pretrained model for predict noisy images
    :param diffusion: the diffusion model for purfication
    :param m: the number of local smoothing noises
    :param local_smoothing_sd: the noise_sd for the local smoothing => for the prediction of classifier
    :param fix_local_noise: use the m fixed local smoothing noises or not
    :return: the confidence for each class
    """
    def __init__(self, dataset, classifier, diffusion, m = 5, local_smoothing_sd = 0.25, fix_local_noise = True, seed = 42):
        super(DiffSmooth_Classifier, self).__init__()
        self.dataset = dataset
        self.classifier = classifier
        self.diffusion = diffusion
        self.classifier.eval()
        self.diffusion.eval()
        
        self.local_smoothing_sd = local_smoothing_sd
        self.m = m
        self.fix_local_noise = fix_local_noise
        ## we will fix the local smoothing noise by default, but actually the performance will not differ too much
        if self.fix_local_noise:
            torch.manual_seed(seed)
            if self.dataset == 'cifar10':
                self.local_smoothing_noise = torch.randn(m,3,32,32).cuda() * self.local_smoothing_sd
            elif self.dataset == 'imagenet':
                self.local_smoothing_noise = torch.randn(m,3,224,224).cuda() * self.local_smoothing_sd

    def forward(self, x):
        x = self.diffusion(x)
        confidence = 0
        for i in range(self.m):
            if self.fix_local_noise:
                local_smoothing_noise = self.local_smoothing_noise[i].unsqueeze(0)
            else:
                local_smoothing_noise = torch.randn(1, *x.shape[1:]).cuda() * self.local_smoothing_sd
            output = self.classifier(x + local_smoothing_noise)
            confidence += torch.softmax(output, dim=1)
        confidence /= self.m
        return confidence
        