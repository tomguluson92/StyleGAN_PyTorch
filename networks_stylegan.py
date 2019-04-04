# coding: UTF-8
"""
    @author: samuel ko
    @date:   2019.04.02
    @notice:
             1) we don't add blur2d mechanism in generator to avoid the generated image from blurry and noisy.

"""
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
import os
from torch.nn.init import kaiming_normal_


class ApplyNoise(nn.Module):
    def __init__(self, bs, res, randomize_noise=True):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(bs, 1, res, res).to("cuda"))

    def forward(self, x, noise):
        return x + self.weight * noise


class FC(nn.Module):
    def __init__(self, in_channels, out_channels, gain=2**(0.5), use_wscale=False, lrmul=1, bias=True):
        """
            The complete conversion of Dense/FC/Linear Layer of original Tensorflow version.
        """
        super(FC, self).__init__()
        he_std = gain * in_channels ** (-0.5)  # He init
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_lrmul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_lrmul = lrmul
        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels) * init_std)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_channels))
            self.b_lrmul = lrmul
        else:
            self.bias = None

    def forward(self, x):
        if self.bias is not None:
            out = F.linear(x, self.weight * self.w_lrmul, self.bias * self.b_lrmul)
        else:
            out = F.linear(x, self.weight * self.w_lrmul)
        out = F.leaky_relu(out, 0.2, inplace=True)
        return out


class Blur2d(nn.Module):
    def __init__(self, f=[1,2,1], normalize=True, flip=False, stride=1):
        """
            depthwise_conv2d:
            https://blog.csdn.net/mao_xiao_feng/article/details/78003476
        """
        super(Blur2d, self).__init__()
        assert isinstance(f, list) or f is None, "kernel f must be an instance of python built_in type list!"

        if f is not None:
            f = torch.tensor(f, dtype=torch.float32)
            f = f[:, None] * f[None, :]
            f = f[None, None]
            if normalize:
                f = f / f.sum()
            if flip:
                f = f[:, :, ::-1, ::-1]
            self.f = f
        else:
            self.f = None
        self.stride = stride

    def forward(self, x):
        if self.f is not None:
            # expand kernel channels
            kernel = self.f.expand(x.size(1), -1, -1, -1).to(x.device)
            x = F.conv2d(
                x,
                kernel,
                stride=self.stride,
                padding=int((self.f.size(2)-1)/2),
                groups=x.size(1)
            )
            return x
        else:
            return x

class PixelNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        """
            @notice: avoid in-place ops.
            https://discuss.pytorch.org/t/encounter-the-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/836/3
        """
        super(PixelNorm, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        tmp  = torch.mul(x, x) # or x ** 2
        tmp1 = torch.rsqrt(torch.mean(tmp, dim=1, keepdim=True) + self.epsilon)

        return x * tmp1


class InstanceNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        """
            @notice: avoid in-place ops.
            https://discuss.pytorch.org/t/encounter-the-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/836/3
        """
        super(InstanceNorm, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        x   = x - torch.mean(x, (2, 3), True)
        tmp = torch.mul(x, x) # or x ** 2
        tmp = torch.rsqrt(torch.mean(tmp, (2, 3), True) + self.epsilon)
        return x * tmp


###
# 初始化策略 2019.3.31
# styleGAN中只初始化weight, 而没管bias.
# https://discuss.pytorch.org/t/how-are-layer-weights-and-biases-initialized-by-default/13073/2
# kaiming_normal_ : https://pytorch.org/docs/master/nn.html?highlight=init#torch-nn-init
###
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        kaiming_normal_(m.weight.data)

#model.apply(weights_init)


# =========================================================================
#   Define sub-network
#   2019.3.31
#   FC
# =========================================================================
class G_mapping(nn.Module):
    def __init__(self,
                 mapping_fmaps=512,
                 dlatent_size=512,
                 resolution=1024,
                 normalize_latents=True,  # Normalize latent vectors (Z) before feeding them to the mapping layers?
                 use_wscale=True,         # Enable equalized learning rate?
                 gain=2**(0.5)            # original gain in tensorflow.
                 ):
        super(G_mapping, self).__init__()
        self.mapping_fmaps = mapping_fmaps
        self.func = nn.Sequential(
            FC(self.mapping_fmaps, dlatent_size, gain, use_wscale),
            FC(dlatent_size, dlatent_size, gain, use_wscale),
            FC(dlatent_size, dlatent_size, gain, use_wscale),
            FC(dlatent_size, dlatent_size, gain, use_wscale),
            FC(dlatent_size, dlatent_size, gain, use_wscale),
            FC(dlatent_size, dlatent_size, gain, use_wscale),
            FC(dlatent_size, dlatent_size, gain, use_wscale),
            FC(dlatent_size, dlatent_size, gain, use_wscale)
        )

        self.normalize_latents = normalize_latents
        self.resolution_log2 = int(np.log2(resolution))
        self.num_layers = self.resolution_log2 * 2 - 2
        self.pixel_norm = PixelNorm()
        # - 2 means we start from feature map with height and width equals 4.
        # as this example, we get num_layers = 18.

    def forward(self, x):
        if self.normalize_latents:
            x = self.pixel_norm(x)
        out = self.func(x)
        return out, self.num_layers


class G_synthesis(nn.Module):
    def __init__(self,
                 dlatent_size,                       # Disentangled latent (W) dimensionality.
                 resolution=1024,                    # Output resolution (1024 x 1024 by default).
                 fmap_base=8192,                     # Overall multiplier for the number of feature maps.
                 num_channels=3,                     # Number of output color channels.
                 structure='fixed',                  # 'fixed' = no progressive growing, 'linear' = human-readable, 'recursive' = efficient, 'auto' = select automatically.
                 fmap_max=512,                       # Maximum number of feature maps in any layer.
                 fmap_decay=1.0,                     # log2 feature map reduction when doubling the resolution.
                 use_pixel_norm      = False,        # Enable pixelwise feature vector normalization?
                 use_instance_norm   = True,        # Enable instance normalization?
                 bs=16):                             # batch size.
        """
            2019.3.31
        :param dlatent_size: 512 Disentangled latent(W) dimensionality.
        :param resolution: 1024 输出图像的分辨率。
        :param fmap_base:
        :param num_channels:
        :param structure: 先实现最基础的fixed模式.
        :param fmap_max:
        :param bs: batch size.
        """
        super(G_synthesis, self).__init__()

        self.nf = lambda stage: min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
        self.structure = structure
        self.resolution_log2 = int(np.log2(resolution))
        # - 2 means we start from feature map with height and width equals 4.
        # as this example, we get num_layers = 18.
        num_layers = self.resolution_log2 * 2 - 2
        self.num_layers = num_layers
        self.use_pixel_norm = use_pixel_norm
        self.use_instance_norm = use_instance_norm
        self.pixel_norm = PixelNorm()
        self.instance_norm = InstanceNorm()


        # Noise inputs.
        self.noise_inputs = []
        for layer_idx in range(num_layers):
            res = layer_idx // 2 + 2
            shape = [1, 1, 2 ** res, 2 ** res]
            self.noise_inputs.append(torch.randn(*shape).to("cuda"))

        self.apply_noise2  = ApplyNoise(bs, 4)
        self.apply_noise3  = ApplyNoise(bs, 8)
        self.apply_noise4  = ApplyNoise(bs, 16)
        self.apply_noise5  = ApplyNoise(bs, 32)
        self.apply_noise6  = ApplyNoise(bs, 64)
        self.apply_noise7  = ApplyNoise(bs, 128)
        self.apply_noise8  = ApplyNoise(bs, 256)
        self.apply_noise9  = ApplyNoise(bs, 512)
        self.apply_noise10 = ApplyNoise(bs, 1024)

        # torgb: fixed mode
        self.torgb = nn.Conv2d(self.nf(self.resolution_log2), num_channels, kernel_size=1)


        self.const_input = nn.Parameter(torch.ones(bs, self.nf(1), 4, 4))
        self.style21  = nn.Linear(dlatent_size, self.nf(1)*2)
        self.style22  = nn.Linear(dlatent_size, self.nf(1)*2)
        self.style3   = nn.Linear(dlatent_size, self.nf(2)*2)
        self.style4   = nn.Linear(dlatent_size, self.nf(3)*2)
        self.style5   = nn.Linear(dlatent_size, self.nf(4)*2)
        self.style6   = nn.Linear(dlatent_size, self.nf(5)*2)
        self.style7   = nn.Linear(dlatent_size, self.nf(6)*2)
        self.style8   = nn.Linear(dlatent_size, self.nf(7)*2)
        self.style9   = nn.Linear(dlatent_size, self.nf(8)*2)
        self.style10  = nn.Linear(dlatent_size, self.nf(9)*2)


        self.up_conv = nn.Upsample(scale_factor=2, mode='nearest')
        self.transpose_conv_64_128    = nn.ConvTranspose2d(self.nf(5), self.nf(5), 4, stride=2, padding=1)
        self.transpose_conv_128_256   = nn.ConvTranspose2d(self.nf(6), self.nf(6), 4, stride=2, padding=1)
        self.transpose_conv_256_512   = nn.ConvTranspose2d(self.nf(7), self.nf(7), 4, stride=2, padding=1)
        self.transpose_conv_512_1024  = nn.ConvTranspose2d(self.nf(8), self.nf(8), 4, stride=2, padding=1)

        # for kernel_size = 3,
        # torch padding=(1, 1) == tensorflow padding='SAME'.
        self.conv2 = nn.Conv2d(in_channels=self.nf(1), out_channels=self.nf(1), kernel_size=3, padding=(1, 1))

        self.conv31 = nn.Conv2d(in_channels=self.nf(1), out_channels=self.nf(2), kernel_size=3, padding=(1, 1))
        self.conv32 = nn.Conv2d(in_channels=self.nf(2), out_channels=self.nf(2), kernel_size=3, padding=(1, 1))

        self.conv41 = nn.Conv2d(in_channels=self.nf(2), out_channels=self.nf(3), kernel_size=3, padding=(1, 1))
        self.conv42 = nn.Conv2d(in_channels=self.nf(3), out_channels=self.nf(3), kernel_size=3, padding=(1, 1))

        self.conv51 = nn.Conv2d(in_channels=self.nf(3), out_channels=self.nf(4), kernel_size=3, padding=(1, 1))
        self.conv52 = nn.Conv2d(in_channels=self.nf(4), out_channels=self.nf(4), kernel_size=3, padding=(1, 1))

        self.conv61 = nn.Conv2d(in_channels=self.nf(4), out_channels=self.nf(5), kernel_size=3, padding=(1, 1))
        self.conv62 = nn.Conv2d(in_channels=self.nf(5), out_channels=self.nf(5), kernel_size=3, padding=(1, 1))

        self.conv71 = nn.Conv2d(in_channels=self.nf(5), out_channels=self.nf(6), kernel_size=3, padding=(1, 1))
        self.conv72 = nn.Conv2d(in_channels=self.nf(6), out_channels=self.nf(6), kernel_size=3, padding=(1, 1))

        self.conv81 = nn.Conv2d(in_channels=self.nf(6), out_channels=self.nf(7), kernel_size=3, padding=(1, 1))
        self.conv82 = nn.Conv2d(in_channels=self.nf(7), out_channels=self.nf(7), kernel_size=3, padding=(1, 1))

        self.conv91 = nn.Conv2d(in_channels=self.nf(7), out_channels=self.nf(8), kernel_size=3, padding=(1, 1))
        self.conv92 = nn.Conv2d(in_channels=self.nf(8), out_channels=self.nf(8), kernel_size=3, padding=(1, 1))

        self.conv101 = nn.Conv2d(in_channels=self.nf(8), out_channels=self.nf(9), kernel_size=3, padding=(1, 1))
        self.conv102 = nn.Conv2d(in_channels=self.nf(9), out_channels=self.nf(9), kernel_size=3, padding=(1, 1))

        self.conv111 = nn.Conv2d(in_channels=self.nf(9), out_channels=self.nf(10), kernel_size=3, padding=(1, 1))

    def forward(self, dlatent):
        """
           dlatent是 Disentangled latents (W), shape为[minibatch, num_layers, dlatent_size].
        :param dlatent:
        :return:
        """
        images_out = None
        # Fixed structure: simple and efficient, but does not support progressive growing.
        if self.structure == 'fixed':
            # initial block 0:
            # A replacement of layer_epilogue in original Tensorflow version.
            x = self.apply_noise2(self.const_input, self.noise_inputs[0])
            x = F.leaky_relu(x, 0.2, inplace=True)
            if self.use_pixel_norm:
                x = self.pixel_norm(x)
            if self.use_instance_norm:
                x = self.instance_norm(x)

            style = self.style21(dlatent[:, 0])
            style = style.reshape(-1, 2, self.nf(1), 1, 1)
            x = x * (style[:, 0] + 1) + style[:, 1]
            x = self.conv2(x)
            x = self.apply_noise2(x, self.noise_inputs[1])
            x = F.leaky_relu(x, 0.2, inplace=True)
            if self.use_pixel_norm:
                x = self.pixel_norm(x)
            if self.use_instance_norm:
                x = self.instance_norm(x)

            style = self.style22(dlatent[:, 1])
            style = style.reshape(-1, 2, self.nf(1), 1, 1)
            x = x * (style[:, 0] + 1) + style[:, 1]

            # block 1:
            # 4 x 4 -> 8 x 8
            res = 3
            """
                notice: 原实现中, 当特征图的高和宽大于等于64的时候, 使用反卷积, 小于等于64的时候, 直接使用最近邻上采样.
            """
            x = self.conv31(self.up_conv(x))
            x = self.apply_noise3(x, self.noise_inputs[res*2-4])
            x = F.leaky_relu(x, 0.2, inplace=True)
            if self.use_pixel_norm:
                x = self.pixel_norm(x)
            if self.use_instance_norm:
                x = self.instance_norm(x)

            style = self.style3(dlatent[:, res*2-4])
            style = style.reshape(-1, 2, self.nf(res-1), 1, 1)
            x = x * (style[:, 0] + 1) + style[:, 1]

            x = self.conv32(x)
            x = self.apply_noise3(x, self.noise_inputs[res*2-3])
            x = F.leaky_relu(x, 0.2, inplace=True)
            if self.use_pixel_norm:
                x = self.pixel_norm(x)
            if self.use_instance_norm:
                x = self.instance_norm(x)

            style = self.style3(dlatent[:, res*2-3])
            style = style.reshape(-1, 2, self.nf(res-1), 1, 1)
            x = x * (style[:, 0] + 1) + style[:, 1]

            # block 2:
            # 8 x 8 -> 16 x 16
            res = 4
            x = self.conv41(self.up_conv(x))
            x = self.apply_noise4(x, self.noise_inputs[res*2-4])
            x = F.leaky_relu(x, 0.2, inplace=True)
            if self.use_pixel_norm:
                x = self.pixel_norm(x)
            if self.use_instance_norm:
                x = self.instance_norm(x)

            style = self.style4(dlatent[:, res*2-4])
            style = style.reshape(-1, 2, self.nf(res-1), 1, 1)
            x = x * (style[:, 0] + 1) + style[:, 1]

            x = self.conv42(x)
            x = self.apply_noise4(x, self.noise_inputs[res*2-3])
            x = F.leaky_relu(x, 0.2, inplace=True)
            if self.use_pixel_norm:
                x = self.pixel_norm(x)
            if self.use_instance_norm:
                x = self.instance_norm(x)

            style = self.style4(dlatent[:, res*2-3])
            style = style.reshape(-1, 2, self.nf(res-1), 1, 1)
            x = x * (style[:, 0] + 1) + style[:, 1]

            # block 3:
            # 16 x 16 -> 32 x 32
            res = 5
            x = self.conv51(self.up_conv(x))
            x = self.apply_noise5(x, self.noise_inputs[res*2-4])
            x = F.leaky_relu(x, 0.2, inplace=True)
            if self.use_pixel_norm:
                x = self.pixel_norm(x)
            if self.use_instance_norm:
                x = self.instance_norm(x)

            style = self.style5(dlatent[:, res*2-4])
            style = style.reshape(-1, 2, self.nf(res-1), 1, 1)
            x = x * (style[:, 0] + 1) + style[:, 1]

            x = self.conv52(x)
            x = self.apply_noise5(x, self.noise_inputs[res*2-3])
            x = F.leaky_relu(x, 0.2, inplace=True)
            if self.use_pixel_norm:
                x = self.pixel_norm(x)
            if self.use_instance_norm:
                x = self.instance_norm(x)

            style = self.style5(dlatent[:, res*2-3])
            style = style.reshape(-1, 2, self.nf(res-1), 1, 1)
            x = x * (style[:, 0] + 1) + style[:, 1]

            # block 4:
            # 32 x 32 -> 64 x 64
            res = 6
            x = self.conv61(self.up_conv(x))
            x = self.apply_noise6(x, self.noise_inputs[res*2-4])
            x = F.leaky_relu(x, 0.2, inplace=True)
            if self.use_pixel_norm:
                x = self.pixel_norm(x)
            if self.use_instance_norm:
                x = self.instance_norm(x)

            style = self.style6(dlatent[:, res*2-4])
            style = style.reshape(-1, 2, self.nf(res-1), 1, 1)
            x = x * (style[:, 0] + 1) + style[:, 1]

            x = self.conv62(x)
            x = self.apply_noise6(x, self.noise_inputs[res*2-3])
            x = F.leaky_relu(x, 0.2, inplace=True)
            if self.use_pixel_norm:
                x = self.pixel_norm(x)
            if self.use_instance_norm:
                x = self.instance_norm(x)

            style = self.style6(dlatent[:, res*2-3])
            style = style.reshape(-1, 2, self.nf(res-1), 1, 1)
            x = x * (style[:, 0] + 1) + style[:, 1]

            # block 5:
            # 64 x 64 -> 128 x 128
            res = 7
            x = self.conv71(self.transpose_conv_64_128(x))
            x = self.apply_noise7(x, self.noise_inputs[res*2-4])
            x = F.leaky_relu(x, 0.2, inplace=True)
            if self.use_pixel_norm:
                x = self.pixel_norm(x)
            if self.use_instance_norm:
                x = self.instance_norm(x)

            style = self.style7(dlatent[:, res*2-4])
            style = style.reshape(-1, 2, self.nf(res-1), 1, 1)
            x = x * (style[:, 0] + 1) + style[:, 1]

            x = self.conv72(x)
            x = self.apply_noise7(x, self.noise_inputs[res*2-3])
            x = F.leaky_relu(x, 0.2, inplace=True)
            if self.use_pixel_norm:
                x = self.pixel_norm(x)
            if self.use_instance_norm:
                x = self.instance_norm(x)

            style = self.style7(dlatent[:, res*2-3])
            style = style.reshape(-1, 2, self.nf(res-1), 1, 1)
            x = x * (style[:, 0] + 1) + style[:, 1]

            # block 6:
            # 128 x 128 -> 256 x 256
            res = 8
            x = self.conv81(self.transpose_conv_128_256(x))
            x = self.apply_noise8(x, self.noise_inputs[res*2-4])
            x = F.leaky_relu(x, 0.2, inplace=True)
            if self.use_pixel_norm:
                x = self.pixel_norm(x)
            if self.use_instance_norm:
                x = self.instance_norm(x)

            style = self.style8(dlatent[:, res*2-4])
            style = style.reshape(-1, 2, self.nf(res-1), 1, 1)
            x = x * (style[:, 0] + 1) + style[:, 1]

            x = self.conv82(x)
            x = self.apply_noise8(x, self.noise_inputs[res*2-3])
            x = F.leaky_relu(x, 0.2, inplace=True)
            if self.use_pixel_norm:
                x = self.pixel_norm(x)
            if self.use_instance_norm:
                x = self.instance_norm(x)

            style = self.style8(dlatent[:, res*2-3])
            style = style.reshape(-1, 2, self.nf(res-1), 1, 1)
            x = x * (style[:, 0] + 1) + style[:, 1]

            # block 7:
            # 256 x 256 -> 512 x 512
            res = 9
            x = self.conv91(self.transpose_conv_256_512(x))
            x = self.apply_noise9(x, self.noise_inputs[res*2-4])
            x = F.leaky_relu(x, 0.2, inplace=True)
            if self.use_pixel_norm:
                x = self.pixel_norm(x)
            if self.use_instance_norm:
                x = self.instance_norm(x)

            style = self.style9(dlatent[:, res*2-4])
            style = style.reshape(-1, 2, self.nf(res-1), 1, 1)
            x = x * (style[:, 0] + 1) + style[:, 1]

            x = self.conv92(x)
            x = self.apply_noise9(x, self.noise_inputs[res*2-3])
            x = F.leaky_relu(x, 0.2, inplace=True)
            if self.use_pixel_norm:
                x = self.pixel_norm(x)
            if self.use_instance_norm:
                x = self.instance_norm(x)

            style = self.style9(dlatent[:, res*2-3])
            style = style.reshape(-1, 2, self.nf(res-1), 1, 1)
            x = x * (style[:, 0] + 1) + style[:, 1]

            # block 8:
            # 512 x 512 -> 1024 x 1024
            res = 10
            x = self.conv101(self.transpose_conv_512_1024(x))
            x = self.apply_noise10(x, self.noise_inputs[res*2-4])
            x = F.leaky_relu(x, 0.2, inplace=True)
            if self.use_pixel_norm:
                x = self.pixel_norm(x)
            if self.use_instance_norm:
                x = self.instance_norm(x)

            style = self.style10(dlatent[:, res*2-4])
            style = style.reshape(-1, 2, self.nf(res-1), 1, 1)
            x = x * (style[:, 0] + 1) + style[:, 1]

            x = self.conv102(x)
            x = self.apply_noise10(x, self.noise_inputs[res*2-3])
            x = F.leaky_relu(x, 0.2, inplace=True)
            if self.use_pixel_norm:
                x = self.pixel_norm(x)
            if self.use_instance_norm:
                x = self.instance_norm(x)

            style = self.style10(dlatent[:, res*2-3])
            style = style.reshape(-1, 2, self.nf(res-1), 1, 1)
            x = x * (style[:, 0] + 1) + style[:, 1]

            x = self.conv111(x)
            images_out = self.torgb(x)
            return images_out


class StyleGenerator(nn.Module):
    def __init__(self,
                 mapping_fmaps=512,
                 bs=16,
                 style_mixing_prob=0.9,       # Probability of mixing styles during training. None = disable.
                 truncation_psi=0.7,          # Style strength multiplier for the truncation trick. None = disable.
                 truncation_cutoff=8          # Number of layers for which to apply the truncation trick. None = disable.
                 ):
        super(StyleGenerator, self).__init__()
        self.mapping_fmaps = mapping_fmaps
        self.style_mixing_prob = style_mixing_prob
        self.truncation_psi = truncation_psi
        self.truncation_cutoff = truncation_cutoff

        self.mapping = G_mapping(self.mapping_fmaps)
        self.synthesis = G_synthesis(self.mapping_fmaps, bs=bs)

    def forward(self, latents1):
        dlatents1, num_layers = self.mapping(latents1)
        # let [N, O] -> [N, num_layers, O]
        # 这里的unsqueeze不能使用inplace操作, 如果这样的话, 反向传播的链条会断掉的.
        dlatents1 = dlatents1.unsqueeze(1)
        dlatents1 = dlatents1.expand(-1, int(num_layers), -1)

        # Add mixing style mechanism.
        # with torch.no_grad():
        #     latents2 = torch.randn(latents1.shape).to(latents1.device)
        #     dlatents2, num_layers = self.mapping(latents2)
        #     dlatents2 = dlatents2.unsqueeze(1)
        #     dlatents2 = dlatents2.expand(-1, int(num_layers), -1)
        #
        #     # TODO: original NvLABs produce a placeholder "lod", this mechanism was not added here.
        #     cur_layers = num_layers
        #     mix_layers = num_layers
        #     if np.random.random() < self.style_mixing_prob:
        #         mix_layers = np.random.randint(1, cur_layers)
        #
        #     # NvLABs: dlatents = tf.where(tf.broadcast_to(layer_idx < mixing_cutoff, tf.shape(dlatents)), dlatents, dlatents2)
        #     for i in range(num_layers):
        #         if i >= mix_layers:
        #             dlatents1[:, i, :] = dlatents2[:, i, :]

        # Apply truncation trick.
        if self.truncation_psi and self.truncation_cutoff:
            coefs = np.ones([1, num_layers, 1], dtype=np.float32)
            for i in range(num_layers):
                if i < self.truncation_cutoff:
                    coefs[:, i, :] *= self.truncation_psi
            """Linear interpolation.
               a + (b - a) * t (a = 0)
               reduce to
               b * t
            """

            dlatents1 = dlatents1 * torch.Tensor(coefs).to(dlatents1.device)

        img = self.synthesis(dlatents1)
        return img


class StyleDiscriminator(nn.Module):
    def __init__(self,
                 resolution=1024,
                 fmap_base=8192,
                 num_channels=3,
                 structure='fixed',  # 'fixed' = no progressive growing, 'linear' = human-readable, 'recursive' = efficient, only support 'fixed' mode now.
                 fmap_max=512,
                 fmap_decay=1.0,
                 f=[1, 2, 1]         # (Huge overload, if you dont have enough resouces, please pass it as `f = None`)Low-pass filter to apply when resampling activations. None = no filtering.
                 ):
        """
            Noitce: we only support input pic with height == width.

            if H or W >= 128, we use avgpooling2d to do feature map shrinkage.
            else: we use ordinary conv2d.
        """
        super().__init__()
        self.resolution_log2 = int(np.log2(resolution))
        assert resolution == 2 ** self.resolution_log2 and resolution >= 4
        self.nf = lambda stage: min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
        # fromrgb: fixed mode
        self.fromrgb = nn.Conv2d(num_channels, self.nf(self.resolution_log2-1), kernel_size=1)
        self.structure = structure

        # blur2d
        self.blur2d = Blur2d(f)

        # down_sample
        self.down1 = nn.AvgPool2d(2)
        self.down21 = nn.Conv2d(self.nf(self.resolution_log2-5), self.nf(self.resolution_log2-5), kernel_size=2, stride=2)
        self.down22 = nn.Conv2d(self.nf(self.resolution_log2-6), self.nf(self.resolution_log2-6), kernel_size=2, stride=2)
        self.down23 = nn.Conv2d(self.nf(self.resolution_log2-7), self.nf(self.resolution_log2-7), kernel_size=2, stride=2)
        self.down24 = nn.Conv2d(self.nf(self.resolution_log2-8), self.nf(self.resolution_log2-8), kernel_size=2, stride=2)

        # conv1: padding=same
        self.conv1 = nn.Conv2d(self.nf(self.resolution_log2-1), self.nf(self.resolution_log2-1), kernel_size=3, padding=(1, 1))
        self.conv2 = nn.Conv2d(self.nf(self.resolution_log2-1), self.nf(self.resolution_log2-2), kernel_size=3, padding=(1, 1))
        self.conv3 = nn.Conv2d(self.nf(self.resolution_log2-2), self.nf(self.resolution_log2-3), kernel_size=3, padding=(1, 1))
        self.conv4 = nn.Conv2d(self.nf(self.resolution_log2-3), self.nf(self.resolution_log2-4), kernel_size=3, padding=(1, 1))
        self.conv5 = nn.Conv2d(self.nf(self.resolution_log2-4), self.nf(self.resolution_log2-5), kernel_size=3, padding=(1, 1))
        self.conv6 = nn.Conv2d(self.nf(self.resolution_log2-5), self.nf(self.resolution_log2-6), kernel_size=3, padding=(1, 1))
        self.conv7 = nn.Conv2d(self.nf(self.resolution_log2-6), self.nf(self.resolution_log2-7), kernel_size=3, padding=(1, 1))
        self.conv8 = nn.Conv2d(self.nf(self.resolution_log2-7), self.nf(self.resolution_log2-8), kernel_size=3, padding=(1, 1))

        # calculate point:
        self.conv_last = nn.Conv2d(self.nf(self.resolution_log2-8), self.nf(1), kernel_size=3, padding=(1, 1))
        self.dense0 = nn.Linear(fmap_base, self.nf(0))
        self.dense1 = nn.Linear(self.nf(0), 1)

    def forward(self, input):
        if self.structure == 'fixed':
            x = F.leaky_relu(self.fromrgb(input), 0.2, inplace=True)
            # 1. 1024 x 1024 x nf(9)(16) -> 512 x 512
            res = self.resolution_log2
            x = F.leaky_relu(self.conv1(x), 0.2, inplace=True)
            x = F.leaky_relu(self.down1(self.blur2d(x)), 0.2, inplace=True)

            # 2. 512 x 512 -> 256 x 256
            res -= 1
            x = F.leaky_relu(self.conv2(x), 0.2, inplace=True)
            x = F.leaky_relu(self.down1(self.blur2d(x)), 0.2, inplace=True)

            # 3. 256 x 256 -> 128 x 128
            res -= 1
            x = F.leaky_relu(self.conv3(x), 0.2, inplace=True)
            x = F.leaky_relu(self.down1(self.blur2d(x)), 0.2, inplace=True)

            # 4. 128 x 128 -> 64 x 64
            res -= 1
            x = F.leaky_relu(self.conv4(x), 0.2, inplace=True)
            x = F.leaky_relu(self.down1(self.blur2d(x)), 0.2, inplace=True)

            # 5. 64 x 64 -> 32 x 32
            res -= 1
            x = F.leaky_relu(self.conv5(x), 0.2, inplace=True)
            x = F.leaky_relu(self.down21(self.blur2d(x)), 0.2, inplace=True)

            # 6. 32 x 32 -> 16 x 16
            res -= 1
            x = F.leaky_relu(self.conv6(x), 0.2, inplace=True)
            x = F.leaky_relu(self.down22(self.blur2d(x)), 0.2, inplace=True)

            # 7. 16 x 16 -> 8 x 8
            res -= 1
            x = F.leaky_relu(self.conv7(x), 0.2, inplace=True)
            x = F.leaky_relu(self.down23(self.blur2d(x)), 0.2, inplace=True)

            # 8. 8 x 8 -> 4 x 4
            res -= 1
            x = F.leaky_relu(self.conv8(x), 0.2, inplace=True)
            x = F.leaky_relu(self.down24(self.blur2d(x)), 0.2, inplace=True)

            # 9. 4 x 4 -> point
            x = F.leaky_relu(self.conv_last(x), 0.2, inplace=True)
            # N x 8192(4 x 4 x nf(1)).
            x = x.view(x.size(0), -1)
            x = F.leaky_relu(self.dense0(x), 0.2, inplace=True)
            # N x 1
            x = F.leaky_relu(self.dense1(x), 0.2, inplace=True)
            return x

