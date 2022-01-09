"""Forward processing of raw data to sRGB images.

Unprocessing Images for Learned Raw Denoising
http://timothybrooks.com/tech/unprocessing
"""

import numpy as np
import torch
from torchinterp1d import Interp1d
from os.path import join

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def apply_gains(bayer_images, wbs):
    """Applies white balance to a batch of Bayer images."""
    N, C, _, _ = bayer_images.shape
    outs = bayer_images * wbs.view(N, C, 1, 1)
    return outs


def apply_ccms(images, ccms):
    """Applies color correction matrices."""
    images = images.permute(
        0, 2, 3, 1)  # Permute the image tensor to BxHxWxC format from BxCxHxW format
    images = images[:, :, :, None, :]
    ccms = ccms[:, None, None, :, :]
    outs = torch.sum(images * ccms, dim=-1)
    # Re-Permute the tensor back to BxCxHxW format
    outs = outs.permute(0, 3, 1, 2)
    return outs


def gamma_compression(images, gamma=2.2):
    """Converts from linear to gamma space."""
    outs = torch.clamp(images, min=1e-8) ** (1 / gamma)
    # outs = (1 + gamma[0]) * np.power(images, 1.0/gamma[1]) - gamma[0] + gamma[2]*images
    outs = torch.clamp((outs*255).int(), min=0, max=255).float() / 255
    return outs


def binning(bayer_images):
    """RGBG -> RGB"""
    lin_rgb = torch.stack([
        bayer_images[:,0,...], 
        torch.mean(bayer_images[:, [1,3], ...], dim=1), 
        bayer_images[:,2,...]], dim=1)

    return lin_rgb


def process(bayer_images, wbs, cam2rgbs, gamma=2.2, CRF=None):
    """Processes a batch of Bayer RGBG images into sRGB images."""
    # White balance.
    bayer_images = apply_gains(bayer_images, wbs)
    # Binning
    bayer_images = torch.clamp(bayer_images, min=0.0, max=1.0)
    images = binning(bayer_images)
    # Color correction.
    images = apply_ccms(images, cam2rgbs)
    # Gamma compression.
    images = torch.clamp(images, min=0.0, max=1.0)
    if CRF is None:
        images = gamma_compression(images, gamma)
    else:
        images = camera_response_function(images, CRF)
    
    return images


def camera_response_function(images, CRF):
    E, fs = CRF # unpack CRF data
    
    outs = torch.zeros_like(images)
    device = images.device

    for i in range(images.shape[0]):
        img = images[i].view(3, -1)
        out = Interp1d()(E.to(device), fs.to(device), img)        
        outs[i, ...] = out.view(3, images.shape[2], images.shape[3])

    outs = torch.clamp((outs*255).int(), min=0, max=255).float() / 255    
    return outs


def raw2rgb(packed_raw, raw, CRF=None, gamma=2.2): 
    """Raw2RGB pipeline (preprocess version)"""
    wb = np.array(raw.camera_whitebalance) 
    wb /= wb[1]
    cam2rgb = raw.rgb_camera_matrix[:3, :3]

    if isinstance(packed_raw, np.ndarray):
        packed_raw = torch.from_numpy(packed_raw).float()

    wb = torch.from_numpy(wb).float().to(packed_raw.device)
    cam2rgb = torch.from_numpy(cam2rgb).float().to(packed_raw.device)

    out = process(packed_raw[None], wbs=wb[None], cam2rgbs=cam2rgb[None], gamma=gamma, CRF=CRF)[0, ...].numpy()
    
    return out


def raw2rgb_v2(packed_raw, wb, ccm, CRF=None, gamma=2.2): # RGBG
    packed_raw = torch.from_numpy(packed_raw).float()
    wb = torch.from_numpy(wb).float()
    cam2rgb = torch.from_numpy(ccm).float()
    out = process(packed_raw[None], wbs=wb[None], cam2rgbs=cam2rgb[None], gamma=gamma, CRF=CRF)[0, ...].numpy()
    return out


def raw2rgb_postprocess(packed_raw, raw, CRF=None):
    """Raw2RGB pipeline (postprocess version)"""
    assert packed_raw.ndimension() == 4 and packed_raw.shape[0] == 1
    wb = np.array(raw.camera_whitebalance) 
    wb /= wb[1]
    cam2rgb = raw.rgb_camera_matrix[:3, :3]

    wb = torch.from_numpy(wb[None]).float().to(packed_raw.device)
    cam2rgb = torch.from_numpy(cam2rgb[None]).float().to(packed_raw.device)
    out = process(packed_raw, wbs=wb, cam2rgbs=cam2rgb, gamma=2.2, CRF=CRF)
    return out


def read_wb_ccm(raw):
    wb = np.array(raw.camera_whitebalance) 
    wb /= wb[1]
    wb = wb.astype(np.float32)
    ccm = raw.rgb_camera_matrix[:3, :3].astype(np.float32)
    return wb, ccm


def read_emor(address):    
    def _read_curve(lst):
        curve = [l.strip() for l in lst]
        curve = ' '.join(curve)        
        curve = np.array(curve.split()).astype(np.float32)
        return curve

    with open(address) as f:
        lines = f.readlines()
        k = 1
        E = _read_curve(lines[k:k+256])        
        k += 257
        f0 = _read_curve(lines[k:k+256])
        hs = []
        for _ in range(25):
            k += 257
            hs.append(_read_curve(lines[k:k+256]))

        hs = np.array(hs)

        return E, f0, hs      


def read_dorf(address):
    with open(address) as f:
        lines = f.readlines()
        curve_names = lines[0::6]
        Es = lines[3::6]
        Bs = lines[5::6]

        Es = [np.array(E.strip().split()).astype(np.float32) for E in Es]
        Bs = [np.array(B.strip().split()).astype(np.float32) for B in Bs]

    return curve_names, Es, Bs


def load_CRF():
    # init CRF function
    fs = np.loadtxt(join('EMoR', 'CRF_SonyA7S2_5.txt'))
    E, _, _ = read_emor(join('EMoR', 'emor.txt'))
    E = torch.from_numpy(E).repeat(3, 1)
    fs = torch.from_numpy(fs)
    CRF = (E, fs)
    return CRF
