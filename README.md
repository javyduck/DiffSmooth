# DiffSmooth: Certifiably Robust Learning via Diffusion Models and Local Smoothing

   ## About

   This repository houses the code for the paper titled [**DiffSmooth: Certifiably Robust Learning via Diffusion Models and Local Smoothing**](https://www.usenix.org/system/files/usenixsecurity23-zhang-jiawei.pdf), accepted at the 32nd USENIX Security Symposium, 2023.

   ## Introduction

   DiffSmooth improves the robustness of machine learning models against adversarial attacks. Utilizing a two-fold strategy, it first applies diffusion models for adversarial purification and then enhances robustness via local smoothing. Our SOTA results include a certified accuracy boost from $36.0\%$ to $53.0\%$ under $\ell_2$ radius $1.5$ on ImageNet.

   ## Getting Started

   ### Prerequisites

   To set up your environment:

   ```
git clone git@github.com:javyduck/DiffSmooth.git
cd DiffSmooth
conda create --name diffsmooth python=3.8
conda activate diffsmooth
pip install -r requirements.txt
   ```

### Pretrained Models

Place downloaded pretrained models in the `pretrained_models` directory.

1. **Diffusion Models:**
   - Improved CIFAR10 model: [Download checkpoint](https://openaipublic.blob.core.windows.net/diffusion/march-2021/cifar10_uncond_50M_500K.pt)
   - Guided ImageNet model: [Download checkpoint](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion.pt)
2. **Smoothed Models:**
   - Cohen-trained models: [Cohen GitHub](https://github.com/locuslab/smoothing)
   - SmoothAdv-trained models: [SmoothAdv GitHub](https://github.com/Hadisalman/smoothing-adversarial/tree/master)
   - `beit_large_patch16_224` trained with vanilla Gaussian augmentation (Cohen): [Download](https://uillinoisedu-my.sharepoint.com/:f:/g/personal/jiaweiz7_illinois_edu/EstdhnzHtXJIqhsNlv9q4nUBNcSjVvNp6pcj4ep3BTTVtg?e=iLR8dh)

### Certification:

#### 1. CIFAR10

   ```
# Basic setting
python certify.py cifar10 pretrained_models/cohen/models/cifar10/resnet110/noise_0.25/checkpoint.pth.tar 0.50 certified_results/cifar10/cohen_certification_0.50.txt --local_noise_sd 0.25 --m 5
   ```

   As you'll notice, the base model is trained with a noise level of $\sigma'=0.25$. However, the process of purification tends to blur the images slightly. To compensate for this, we typically employ a `local_noise_sd` value that is slightly larger than $\sigma'$—specifically, $\sigma'+0.01$ or $\sigma'+0.02$—to counteract the blurring effect.

   ```
# Fine-tuned setting to offset blurring effect
python certify.py cifar10 pretrained_models/cohen/models/cifar10/resnet110/noise_0.25/checkpoint.pth.tar 0.50 certified_results/cifar10/cohen_certification_0.50.txt --local_noise_sd 0.26 --m 5
   ```

#### 2. ImageNet

   ```
# Basic setting
python certify.py imagenet pretrained_models/beit_large_patch16_224/noise_sd0.25/checkpoint.pth.tar 0.50 certified_results/imagenet/beit_certification_0.50.txt --local_noise_sd 0.25 --m 5

# Fine-tuned setting to offset blurring effect
python certify.py imagenet pretrained_models/beit_large_patch16_224/noise_sd0.25/checkpoint.pth.tar 0.50 certified_results/imagenet/beit_certification_0.50.txt --local_noise_sd 0.26 --m 5
   ```

## Citation

   If you find our work beneficial, please consider citing our paper:

   ```
@inproceedings {287372,
  author = {Jiawei Zhang and Zhongzhu Chen and Huan Zhang and Chaowei Xiao and Bo Li},
  title = {{DiffSmooth}: Certifiably Robust Learning via Diffusion Models and Local Smoothing},
  booktitle = {32nd USENIX Security Symposium (USENIX Security 23)},
  year = {2023},
  isbn = {978-1-939133-37-3},
  address = {Anaheim, CA},
  pages = {4787--4804},
  url = {https://www.usenix.org/conference/usenixsecurity23/presentation/zhang-jiawei},
  publisher = {USENIX Association},
  month = aug,
}
   ```

## Contact

Thank you for your interest in DiffSmooth!

If you have any questions or encounter any errors while running the code, feel free to contact [jiaweiz@illinois.edu](mailto:jiaweiz7@illinois.edu)!
