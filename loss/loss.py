# coding: UTF-8
"""
    @author: samuel ko
"""

from torch.autograd import Variable
from torch.autograd import grad
import torch.autograd as autograd
import torch.nn as nn
import torch

import numpy as np

def gradient_penalty(x, y, f):
    # interpolation
    shape = [x.size(0)] + [1] * (x.dim() - 1)
    alpha = torch.rand(shape).to(x.device)
    z = x + alpha * (y - x)

    # gradient penalty
    z = Variable(z, requires_grad=True).to(x.device)
    o = f(z)
    g = grad(o, z, grad_outputs=torch.ones(o.size()).to(z.device), create_graph=True)[0].view(z.size(0), -1)
    gp = ((g.norm(p=2, dim=1) - 1)**2).mean()
    return gp


def R1Penalty(real_img, f):
    # gradient penalty
    reals = Variable(real_img, requires_grad=True).to(real_img.device)
    real_logit = f(reals)
    apply_loss_scaling = lambda x: x * torch.exp(x * torch.Tensor([np.float32(np.log(2.0))]).to(real_img.device))
    undo_loss_scaling = lambda x: x * torch.exp(-x * torch.Tensor([np.float32(np.log(2.0))]).to(real_img.device))

    real_logit = apply_loss_scaling(torch.sum(real_logit))
    real_grads = grad(real_logit, reals, grad_outputs=torch.ones(real_logit.size()).to(reals.device), create_graph=True)[0].view(reals.size(0), -1)
    real_grads = undo_loss_scaling(real_grads)
    r1_penalty = torch.sum(torch.mul(real_grads, real_grads))
    return r1_penalty


def R2Penalty(fake_img, f):
    # gradient penalty
    fakes = Variable(fake_img, requires_grad=True).to(fake_img.device)
    fake_logit = f(fakes)
    apply_loss_scaling = lambda x: x * torch.exp(x * torch.Tensor([np.float32(np.log(2.0))]).to(fake_img.device))
    undo_loss_scaling = lambda x: x * torch.exp(-x * torch.Tensor([np.float32(np.log(2.0))]).to(fake_img.device))

    fake_logit = apply_loss_scaling(torch.sum(fake_logit))
    fake_grads = grad(fake_logit, fakes, grad_outputs=torch.ones(fake_logit.size()).to(fakes.device), create_graph=True)[0].view(fakes.size(0), -1)
    fake_grads = undo_loss_scaling(fake_grads)
    r2_penalty = torch.sum(torch.mul(fake_grads, fake_grads))
    return r2_penalty
