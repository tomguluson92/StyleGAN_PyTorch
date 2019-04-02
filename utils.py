# coding: UTF-8
"""
    @author: samuel ko
"""

from matplotlib import pyplot as plt
import os

def plotLossCurve(opts, Loss_D_list, Loss_G_list):
    plt.figure()
    plt.plot(Loss_D_list, '-')
    plt.title("Loss curve (Discriminator)")
    plt.savefig(os.path.join(opts.det, 'images', 'loss_curve_discriminator.png'))

    plt.figure()
    plt.plot(Loss_G_list, '-o')
    plt.title("Loss curve (Generator)")
    plt.savefig(os.path.join(opts.det, 'images', 'loss_curve_generator.png'))