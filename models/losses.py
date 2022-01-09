import torch.nn as nn


class MultipleLoss(nn.Module):
    def __init__(self, losses, weight=None):
        super(MultipleLoss, self).__init__()
        self.losses = nn.ModuleList(losses)
        self.weight = weight or [1/len(self.losses)] * len(self.losses)
    
    def forward(self, predict, target):
        total_loss = 0
        for weight, loss in zip(self.weight, self.losses):
            total_loss += loss(predict, target) * weight
        return total_loss


class ContentLoss():
    def initialize(self, loss):
        self.criterion = loss

    def get_loss(self, fakeIm, realIm):
        return self.criterion(fakeIm, realIm)


def init_loss(opt):
    loss_dic = {}

    print('[i] Pixel Loss: {}'.format(opt.loss))

    pixel_loss = ContentLoss()
    if opt.loss == 'l1':
        pixel_loss.initialize(nn.L1Loss())
    elif opt.loss == 'l2':
        pixel_loss.initialize(nn.MSELoss())

    loss_dic['pixel'] = pixel_loss

    return loss_dic
