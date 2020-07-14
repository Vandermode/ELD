import torch
import torch.nn as nn
import numpy as np
from torch.nn import init
import functools
from torch.optim import lr_scheduler

import util.util as util
from collections import OrderedDict
###############################################################################
# Functions
###############################################################################


def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Sequential):
        return
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.Linear):
        init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    print(classname)
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        init.orthogonal(m.weight.data, gain=1)
    elif isinstance(m, nn.Linear):
        init.orthogonal(m.weight.data, gain=1)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    print('[i] initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    elif init_type == 'edsr':
        pass
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt, last_epoch):
    if opt.lr_policy == 'lambda':
        def lambda_rule(last_epoch):
            lr_l = 1.0 - max(0, last_epoch + 1 - (opt.nEpochs - opt.decay_iter)) / float(opt.decay_iter + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule, last_epoch=last_epoch)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1, last_epoch=last_epoch)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'multistep':
        scheduler = lr_scheduler.MultiStepLR(optimizer, util.parse_args(opt.milestones), last_epoch=last_epoch)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def define_D(opt, in_channels=3):
    # use_sigmoid = opt.gan_type == 'gan'
    use_sigmoid = False # incorporate sigmoid into BCE_stable loss
    
    if opt.which_model_D == 'disc_patch':
        netD = NLayerDiscriminator(in_channels, 64, 3, nn.InstanceNorm2d, use_sigmoid, getIntermFeat=False)
        init_weights(netD, init_type='normal')
    else:
        raise NotImplementedError('%s is not implemented' %opt.which_model_D)

    if len(opt.gpu_ids) > 0:
        assert(torch.cuda.is_available())
        netD.cuda(opt.gpu_ids[0])
    
    return netD


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)
    print('The size of receptive field: %d' % receptive_field(net))


def receptive_field(net):
    def _f(output_size, ksize, stride, dilation):
        return (output_size - 1) * stride + ksize * dilation - dilation + 1    

    stats = []
    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.AvgPool2d, nn.MaxPool2d)):
            stats.append((m.kernel_size, m.stride, m.dilation))
    
    rsize = 1
    for (ksize, stride, dilation) in reversed(stats):
        if type(ksize) == tuple: ksize = ksize[0]
        if type(stride) == tuple: stride = stride[0]
        if type(dilation) == tuple: dilation = dilation[0]
        rsize = _f(rsize, ksize, stride, dilation)
    return rsize


def debug_network(net):
    def _hook(m, i, o):
        print(o.size())
    for m in net.modules():
        m.register_forward_hook(_hook)


##############################################################################
# Classes
##############################################################################

# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, 
    norm_layer=nn.BatchNorm2d, use_sigmoid=False, 
    branch=1, bias=True, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers
        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc*branch, ndf*branch, kernel_size=kw, stride=2, padding=padw, groups=branch, bias=True), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev*branch, nf*branch, groups=branch, kernel_size=kw, stride=2, padding=padw, bias=bias),
                norm_layer(nf*branch), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev*branch, nf*branch, groups=branch, kernel_size=kw, stride=1, padding=padw, bias=bias),
            norm_layer(nf*branch),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf*branch, 1*branch, groups=branch, kernel_size=kw, stride=1, padding=padw, bias=True)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)
