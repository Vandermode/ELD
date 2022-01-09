import numpy as np
from os.path import join
import matplotlib.pyplot as plt
import rawpy
import cv2
import torch
from scipy import interpolate
from torchinterp1d import Interp1d
from skimage.measure import compare_psnr


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


if __name__ == '__main__':
    E, f0, hs = read_emor('emor.txt')

    name = 'DSC01034'
    datadir = '/media/kaixuan/DATA/Papers/Code/Data/Raw/ELD-Sony-new/Radiometric Calibration'

    imgpath = join(datadir, '{}.ARW'.format(name))
    # ipdb.set_trace()

    raw = rawpy.imread(imgpath)    
    # print(raw.rgb_camera_matrix)

    # img_raw = raw.postprocess()
    img_raw = raw.postprocess(use_camera_wb=True, gamma=(1, 1), no_auto_bright=True, output_bps=16) / 65535.

    # img_raw = raw.postprocess(use_camera_wb=True, no_auto_bright=True, output_bps=16) / 65535.

    fs = np.loadtxt('CRF_SonyA7S2_5.txt')
    F_r = interpolate.interp1d(E, fs[0, :])
    F_g = interpolate.interp1d(E, fs[1, :])
    F_b = interpolate.interp1d(E, fs[2, :])
    
    img_r = F_r(img_raw[..., 0])
    img_g = F_b(img_raw[..., 1])
    img_b = F_g(img_raw[..., 2])
    img_process = np.stack([img_r, img_g, img_b], axis=-1)

    with torch.no_grad():        
        E_th = torch.from_numpy(E).repeat(3, 1)
        fs_th = torch.from_numpy(fs)
        img_raw_th = torch.from_numpy(img_raw).permute(2, 0, 1).view(3, -1)
        img_th = Interp1d()(E_th, fs_th, img_raw_th)
        img_th = img_th.view(3, img_raw.shape[0], img_raw.shape[1]).permute(1, 2, 0)

    img_th = img_th.cpu().numpy()
    img_th = (np.clip(img_th, 0, 1) * 255).astype(np.uint8)
    img_raw = (np.clip(img_raw, 0, 1) * 255).astype(np.uint8)

    img_process = (np.clip(img_process, 0, 1) * 255).astype(np.uint8)

    print(compare_psnr(img_th, img_process, data_range=255))

    imgpath = join(datadir, '{}.JPG'.format(name))
    img_rgb = cv2.imread(imgpath, cv2.IMREAD_UNCHANGED)[:,:,::-1]

    # Image.fromarray(img_raw).save('img_raw.png')
    # Image.fromarray(img_rgb).save('img_rgb.png')
    # Image.fromarray(img_process).save('img_process_5.png')

    plt.imshow(img_th)
    plt.show()

    plt.imshow(img_process)
    plt.show()

    plt.imshow(img_rgb)
    plt.show()
