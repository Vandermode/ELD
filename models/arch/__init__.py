from .Unet import UNetSeeInDark
import torch.nn as nn



def unet(in_channels, out_channels, **kwargs):
    return UNetSeeInDark(in_channels, out_channels)

