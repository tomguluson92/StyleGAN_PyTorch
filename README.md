# A PyTorch Implementation of StyleGAN

![Github](https://img.shields.io/badge/PyTorch-v1.0.1-green.svg?style=for-the-badge&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAMAAABEpIrGAAABWVBMVEUAAAD/AACAgID/VVX/QECZZpn/ZjP/SSTjVTnmTTPqVSvrTjvtWzfuVTPzVTHzXS6eU57xWjChUaHtWS/uVzScU6HvWDTwWDHxVzDxVTTuVTOeUZ7vVTHwVjHwWTTwVzPxWTLxVzHxWDPyVzLuVjLvWDGdUqHvWDLwVzPwVjLwWDLxVzHxWDPxVzLvVzGeU57vVjPvWDLwWDLxVzOeUZ7xVzOfUp/xVjLvWDLwVzLwVzLxVzLxVzLvVjHvVzOeUqDwVzLwVzHwVzPwWDLwVzLxVzLvVzLvWDLwVzHwVzLwVzLwWDLwVjPwVjLxVzPwVzLwVjHwWDLxVzLvVzLwVzLwWDLwVzLwVzLwVzLwVzLvVzLwVzKeUp/wVzLwVzLwVzLvVjLwVzHwVzLwVzLwVzLwVzLwVzLwVzLwVzLwVzLwVzLwVzKeUp/wVzKeUp/wVzLwVzKeUp/wVzJC/9NHAAAAcXRSTlMAAQIDBAUFBwkKDA0ODxUWIiUmKywxMTQ1Njw/P0RFRkhJS0xNTlFRVVZXWFpbXl9fYGZqcX5/f4CEiY+QkZKTmZucnZ6foqOlp6ipq66wtrrAwsXLzM3O0tPV2Nzd3+Tm6O/w8vT19vf4+vv8/P39/ginV/UAAAE/SURBVHgBfc1XUyJREMXxMwsLLLtsVjEHzGLOikFERcSMQRQFFREUgf7+D3ZdnMKewO9l6t/3VA3M+tdQ15ebt7+w0dYMoKFSmYM1LZlyAr9fKzOw1kUUANA6/hXWtonWUYf7gSitwV4fMT/sRYktwJbnkdgFmK8JFgZI+QdgM98CsxgpEaCjTLduGH3LklKaGs7wZxpGQySkPTCIkzQByftE0hGkEaoqlamq8B3CvnoNNzqc/p3qJiDefzzzqTgIZVIt5s1/WNQrRGxPDDb4kvfp9avIeSIGp/JyznkpBkm+xGt5yJkQgwRfjmt5xnkgBlt8yXn1+lnkXBGDHmKzei0TaxcD1x2fCgEooyWOtANCL7GX0B/gf7hMLAiDVVKyOVJiMNKW6JNdF8w6r+lDZkyDFVd35Oo+l4oG3ah5B1uZgMvG1s8xAAAAAElFTkSuQmCC)
![Github](https://img.shields.io/badge/python-3.6%20%7C%203.7-green.svg?style=for-the-badge&logo=python)
![GitHub](https://img.shields.io/github/license/neuralnotworklab/stylegan.svg?style=for-the-badge)
![Github](https://img.shields.io/badge/status-WorkInProgress-blue.svg?style=for-the-badge&logo=fire)


This repository contains a PyTorch implementation of the following paper:
> **A Style-Based Generator Architecture for Generative Adversarial Networks**<br>
> Tero Karras (NVIDIA), Samuli Laine (NVIDIA), Timo Aila (NVIDIA)<br>
> http://stylegan.xyz/paper
>
> **Abstract:** *We propose an alternative generator architecture for generative adversarial networks, borrowing from style transfer literature. The new architecture leads to an automatically learned, unsupervised separation of high-level attributes (e.g., pose and identity when trained on human faces) and stochastic variation in the generated images (e.g., freckles, hair), and it enables intuitive, scale-specific control of the synthesis. The new generator improves the state-of-the-art in terms of traditional distribution quality metrics, leads to demonstrably better interpolation properties, and also better disentangles the latent factors of variation. To quantify interpolation quality and disentanglement, we propose two new, automated methods that are applicable to any generator architecture. Finally, we introduce a new, highly varied and high-quality dataset of human faces.*

## Motivation
To the best of my knowledge, there is still not a similar pytorch 1.0 implementation of styleGAN as NvLabs released(Tensorflow),
therefore, i wanna implement it on pytorch1.0.1 to extend its usage in pytorch community.


## Training

## Related
[1. StyleGAN - Official TensorFlow Implementation](https://github.com/NVlabs/stylegan)

[2. The re-implementation of style-based generator idea](https://github.com/SunnerLi/StyleGAN_demo)

## System Requirements
- Ubuntu18.04
- PyTorch 1.0.1
- Numpy 1.13.3
- torchvision 0.2.1

## Q&A

## Acknowledgements
