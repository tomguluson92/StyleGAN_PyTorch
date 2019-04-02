# coding: UTF-8
"""
    @author: samuel ko
"""
import torchvision_sunner.transforms as sunnertransforms
import torchvision_sunner.data as sunnerData
import torchvision.transforms as transforms


from networks_stylegan import StyleGenerator, StyleDiscriminator
from networks_gan import Generator, Discriminator
from utils import plotLossCurve
from loss import gradient_penalty
from opts import TrainOptions

from torchvision.utils import save_image
from tqdm import tqdm
from matplotlib import pyplot as plt
import torch.optim as optim
import numpy as np
import torch
import os

# Hyper-parameters
CRITIC_ITER = 5


def main(opts):
    # Create the data loader
    loader = sunnerData.DataLoader(sunnerData.ImageDataset(
        root=[[opts.path]],
        transform=transforms.Compose([
            sunnertransforms.Resize((1024, 1024)),
            sunnertransforms.ToTensor(),
            sunnertransforms.ToFloat(),
            sunnertransforms.Transpose(sunnertransforms.BHWC2BCHW),
            sunnertransforms.Normalize(),
        ])),
        batch_size=opts.batch_size,
        shuffle=True,
    )

    # Create the model
    G = StyleGenerator(bs=opts.batch_size).to(opts.device)
    D = StyleDiscriminator().to(opts.device)

    # G = Generator().to(opts.device)
    # D = Discriminator().to(opts.device)

    # Create the criterion, optimizer and scheduler
    optim_D = optim.Adam(D.parameters(), lr=0.0001, betas=(0.5, 0.999))
    optim_G = optim.Adam(G.parameters(), lr=0.0001, betas=(0.5, 0.999))
    scheduler_D = optim.lr_scheduler.ExponentialLR(optim_D, gamma=0.99)
    scheduler_G = optim.lr_scheduler.ExponentialLR(optim_G, gamma=0.99)

    # Train
    fix_z = torch.randn([opts.batch_size, 512]).to(opts.device)
    Loss_D_list = [0.0]
    Loss_G_list = [0.0]
    for ep in range(opts.epoch):
        bar = tqdm(loader)
        loss_D_list = []
        loss_G_list = []
        for i, (real_img,) in enumerate(bar):
            # =======================================================================================================
            #   Update discriminator
            # =======================================================================================================
            # Compute adversarial loss toward discriminator
            real_img = real_img.to(opts.device)
            real_logit = D(real_img)
            fake_img = G(torch.randn([real_img.size(0), 512]).to(opts.device))
            fake_logit = D(fake_img.detach())
            d_loss = -(real_logit.mean() - fake_logit.mean()) + gradient_penalty(real_img.data, fake_img.data, D) * 10.0
            loss_D_list.append(d_loss.item())

            # Update discriminator
            optim_D.zero_grad()
            d_loss.backward()
            optim_D.step()

            # =======================================================================================================
            #   Update generator
            # =======================================================================================================
            if i % CRITIC_ITER == 0:
                # Compute adversarial loss toward generator
                fake_img = G(torch.randn([opts.batch_size, 512]).to(opts.device))
                fake_logit = D(fake_img)
                g_loss = -fake_logit.mean()
                loss_G_list.append(g_loss.item())

                # Update generator
                D.zero_grad()
                optim_G.zero_grad()
                g_loss.backward()
                optim_G.step()
            bar.set_description(" {} [G]: {} [D]: {}".format(ep, loss_G_list[-1], loss_D_list[-1]))

        # Save the result
        Loss_G_list.append(np.mean(loss_G_list))
        Loss_D_list.append(np.mean(loss_D_list))
        fake_img = G(fix_z)
        save_image(fake_img, os.path.join(opts.det, 'images', str(ep) + '.png'), nrow=4, normalize=True)
        state = {
            'G': G.state_dict(),
            'D': D.state_dict(),
            'Loss_G': Loss_G_list,
            'Loss_D': Loss_D_list,
        }
        torch.save(state, os.path.join(opts.det, 'models', 'latest.pth'))

        scheduler_D.step()
        scheduler_G.step()

    # Plot the total loss curve
    Loss_D_list = Loss_D_list[1:]
    Loss_G_list = Loss_G_list[1:]
    plotLossCurve(opts, Loss_D_list, Loss_G_list)


if __name__ == '__main__':
    opts = TrainOptions().parse()
    main(opts)