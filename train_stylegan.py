# coding: UTF-8
"""
    @author: samuel ko
"""
import torchvision_sunner.transforms as sunnertransforms
import torchvision_sunner.data as sunnerData
import torchvision.transforms as transforms


from networks_stylegan import StyleGenerator, StyleDiscriminator
from networks_gan import Generator, Discriminator
from utils.utils import plotLossCurve
from loss.loss import gradient_penalty, R1Penalty, R2Penalty
from opts.opts import TrainOptions, INFO

from torchvision.utils import save_image
from tqdm import tqdm
from matplotlib import pyplot as plt
from torch import nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import random
import torch
import os

# Set random seem for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

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

    # Create the model (Multi-GPU support)
    start_epoch = 0
    G = StyleGenerator()
    D = StyleDiscriminator()
    if torch.cuda.device_count() > 1:
        INFO("Multiple GPU:" + str(torch.cuda.device_count()) + "GPUs")
        G = nn.DataParallel(G)
        D = nn.DataParallel(D)
    G.to(opts.device)
    D.to(opts.device)

    # Load the pre-trained weight
    if os.path.exists(opts.resume):
        INFO("Load the pre-trained weight!")
        state = torch.load(opts.resume)
        G.load_state_dict(state['G'])
        D.load_state_dict(state['D'])
        start_epoch = state['start_epoch']
    else:
        INFO("Pre-trained weight cannot load successfully, train from scratch!")

    # Create the criterion, optimizer and scheduler
    optim_D = optim.Adam(D.parameters(), lr=0.0001, betas=(0.5, 0.999))
    optim_G = optim.Adam(G.parameters(), lr=0.0001, betas=(0.5, 0.999))
    scheduler_D = optim.lr_scheduler.ExponentialLR(optim_D, gamma=0.99)
    scheduler_G = optim.lr_scheduler.ExponentialLR(optim_G, gamma=0.99)

    # Train
    fix_z = torch.randn([opts.batch_size, 512]).to(opts.device)
    softplus = nn.Softplus()
    Loss_D_list = [0.0]
    Loss_G_list = [0.0]
    for ep in range(start_epoch, opts.epoch):
        bar = tqdm(loader)
        loss_D_list = []
        loss_G_list = []
        for i, (real_img,) in enumerate(bar):
            # =======================================================================================================
            #   (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            # =======================================================================================================
            # Compute adversarial loss toward discriminator
            D.zero_grad()
            real_img = real_img.to(opts.device)
            real_logit = D(real_img)
            fake_img = G(torch.randn([real_img.size(0), 512]).to(opts.device))
            fake_logit = D(fake_img.detach())
            d_loss = softplus(fake_logit).mean()
            d_loss = d_loss + softplus(-real_logit).mean()

            if opts.r1_gamma != 0.0:
                r1_penalty = R1Penalty(real_img.detach(), D)
                d_loss = d_loss + r1_penalty * (opts.r1_gamma * 0.5)

            if opts.r2_gamma != 0.0:
                r2_penalty = R2Penalty(fake_img.detach(), D)
                d_loss = d_loss + r2_penalty * (opts.r2_gamma * 0.5)

            loss_D_list.append(d_loss.item())

            # Update discriminator
            d_loss.backward()
            optim_D.step()

            # =======================================================================================================
            #   (2) Update G network: maximize log(D(G(z)))
            # =======================================================================================================
            G.zero_grad()
            fake_logit = D(fake_img)
            g_loss = softplus(-fake_logit).mean()
            loss_G_list.append(g_loss.item())

            # Update generator
            g_loss.backward()
            optim_G.step()

            # Output training stats
            bar.set_description("Epoch {} [{}, {}] [G]: {} [D]: {}".format(ep, i+1, len(loader), loss_G_list[-1], loss_D_list[-1]))

        # Save the result
        Loss_G_list.append(np.mean(loss_G_list))
        Loss_D_list.append(np.mean(loss_D_list))

        # Check how the generator is doing by saving G's output on fixed_noise
        with torch.no_grad():
            fake_img = G(fix_z).detach().cpu()
            save_image(fake_img, os.path.join(opts.det, 'images', str(ep) + '.png'), nrow=4, normalize=True)

        # Save model
        state = {
            'G': G.state_dict(),
            'D': D.state_dict(),
            'Loss_G': Loss_G_list,
            'Loss_D': Loss_D_list,
            'start_epoch': ep,
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