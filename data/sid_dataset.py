# See in the Dark (SID) dataset
import torch
import os
import glob
import rawpy
import numpy as np
import random
from os.path import join
import data.torchdata as torchdata
import util.process as process
from util.util import loadmat
import h5py
import exifread
import pickle
import PIL.Image as Image
from scipy.io import loadmat


BaseDataset = torchdata.Dataset


def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def metainfo(rawpath):
    with open(rawpath, 'rb') as f:
        tags = exifread.process_file(f)
        _, suffix = os.path.splitext(os.path.basename(rawpath))

        if suffix == '.dng':
            expo = eval(str(tags['Image ExposureTime']))
            iso = eval(str(tags['Image ISOSpeedRatings']))
        else:
            expo = eval(str(tags['EXIF ExposureTime']))
            iso = eval(str(tags['EXIF ISOSpeedRatings']))

        # print('ISO: {}, ExposureTime: {}'.format(iso, expo))
    return iso, expo


def crop_center(img, cropx, cropy):
    _, y, x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[:, starty:starty+cropy,startx:startx+cropx]

class SIDDataset(BaseDataset):
    def __init__(self, datadir, paired_fns, size=None, flag=None, augment=True, repeat=1, cfa='bayer', memorize=True, stage_in='raw', stage_out='raw', gt_wb=False):
        super(SIDDataset, self).__init__()
        assert cfa == 'bayer' or cfa == 'xtrans'
        self.size = size
        self.datadir = datadir
        self.paired_fns = paired_fns
        self.flag = flag
        self.augment = augment
        self.patch_size = 512
        self.repeat = repeat
        self.cfa = cfa

        self.pack_raw = pack_raw_bayer if cfa == 'bayer' else pack_raw_xtrans

        assert stage_in in ['raw', 'srgb']
        assert stage_out in ['raw', 'srgb']                
        self.stage_in = stage_in
        self.stage_out = stage_out
        self.gt_wb = gt_wb        

        if size is not None:
            self.paired_fns = self.paired_fns[:size]
        
        self.memorize = memorize
        self.target_dict = {}
        self.target_dict_aux = {}
        self.input_dict = {}

    def __getitem__(self, i):
        i = i % len(self.paired_fns)
        input_fn, target_fn = self.paired_fns[i]

        input_path = join(self.datadir, 'short', input_fn)
        target_path = join(self.datadir, 'long', target_fn)

        ratio = compute_expo_ratio(input_fn, target_fn)                

        if self.memorize:
            if target_fn not in self.target_dict:
                with rawpy.imread(target_path) as raw_target:                    
                    target_image = self.pack_raw(raw_target)    
                    wb, ccm = process.read_wb_ccm(raw_target)
                    if self.stage_out == 'srgb':
                        target_image = process.raw2rgb(target_image, raw_target)
                    self.target_dict[target_fn] = target_image
                    self.target_dict_aux[target_fn] = (wb, ccm)

            if input_fn not in self.input_dict:
                with rawpy.imread(input_path) as raw_input:
                    input_image = self.pack_raw(raw_input) * ratio
                    if self.stage_in == 'srgb':
                        if self.gt_wb:
                            wb, ccm = self.target_dict_aux[target_fn]
                            input_image = process.raw2rgb_v2(input_image, wb, ccm)
                        else:
                            input_image = process.raw2rgb(input_image, raw_input)
                    self.input_dict[input_fn] = input_image

            input_image = self.input_dict[input_fn]
            target_image = self.target_dict[target_fn]
            (wb, ccm) = self.target_dict_aux[target_fn]
        else:
            with rawpy.imread(target_path) as raw_target:                    
                target_image = self.pack_raw(raw_target)    
                wb, ccm = process.read_wb_ccm(raw_target)
                if self.stage_out == 'srgb':
                    target_image = process.raw2rgb(target_image, raw_target)

            with rawpy.imread(input_path) as raw_input:
                input_image = self.pack_raw(raw_input) * ratio
                if self.stage_in == 'srgb':
                    if self.gt_wb:
                        input_image = process.raw2rgb_v2(input_image, wb, ccm)
                    else:
                        input_image = process.raw2rgb(input_image, raw_input)  

        if self.augment:
            H = input_image.shape[1]
            W = target_image.shape[2]

            ps = self.patch_size

            xx = np.random.randint(0, W - ps)
            yy = np.random.randint(0, H - ps)

            input = input_image[:, yy:yy + ps, xx:xx + ps]
            target = target_image[:, yy:yy + ps, xx:xx + ps]

            if np.random.randint(2, size=1)[0] == 1:  # random flip
                input = np.flip(input, axis=1) # H
                target = np.flip(target, axis=1)
            if np.random.randint(2, size=1)[0] == 1:
                input = np.flip(input, axis=2) # W
                target = np.flip(target, axis=2)
            if np.random.randint(2, size=1)[0] == 1:  # random transpose
                input = np.transpose(input, (0, 2, 1))
                target = np.transpose(target, (0, 2, 1))
        else:
            input = input_image
            target = target_image

        input = np.maximum(np.minimum(input, 1.0), 0)
        input = np.ascontiguousarray(input)
        target = np.ascontiguousarray(target)

        dic =  {'input': input, 'target': target, 'fn': input_fn, 'cfa': self.cfa, 'rawpath': target_path}

        if self.flag is not None:
            dic.update(self.flag)
        
        return dic

    def __len__(self):
        return len(self.paired_fns) * self.repeat


def compute_expo_ratio(input_fn, target_fn):        
    in_exposure = float(input_fn.split('_')[-1][:-5])
    gt_exposure = float(target_fn.split('_')[-1][:-5])
    ratio = min(gt_exposure / in_exposure, 300)
    return ratio


def pack_raw_bayer(raw):
    #pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    raw_pattern = raw.raw_pattern
    R = np.where(raw_pattern==0)
    G1 = np.where(raw_pattern==1)
    B = np.where(raw_pattern==2)
    G2 = np.where(raw_pattern==3)
    
    white_point = 16383
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.stack((im[R[0][0]:H:2,R[1][0]:W:2], #RGBG
                    im[G1[0][0]:H:2,G1[1][0]:W:2],
                    im[B[0][0]:H:2,B[1][0]:W:2],
                    im[G2[0][0]:H:2,G2[1][0]:W:2]), axis=0).astype(np.float32)

    black_level = np.array(raw.black_level_per_channel)[:,None,None].astype(np.float32)

    # if max(raw.black_level_per_channel) != min(raw.black_level_per_channel):
    #     black_level = 2**round(np.log2(np.max(black_level)))
    # print(black_level)

    out = (out - black_level) / (white_point - black_level)
    out = np.clip(out, 0, 1)
    
    return out


def pack_raw_xtrans(raw):
    # pack X-Trans image to 9 channels
    im = raw.raw_image_visible.astype(np.float32)
    im = (im - 1024) / (16383 - 1024)  # subtract the black level
    im = np.clip(im, 0, 1)

    img_shape = im.shape
    H = (img_shape[0] // 6) * 6
    W = (img_shape[1] // 6) * 6

    out = np.zeros((9, H // 3, W // 3), dtype=np.float32)

    # 0 R
    out[0, 0::2, 0::2] = im[0:H:6, 0:W:6]
    out[0, 0::2, 1::2] = im[0:H:6, 4:W:6]
    out[0, 1::2, 0::2] = im[3:H:6, 1:W:6]
    out[0, 1::2, 1::2] = im[3:H:6, 3:W:6]

    # 1 G
    out[1, 0::2, 0::2] = im[0:H:6, 2:W:6]
    out[1, 0::2, 1::2] = im[0:H:6, 5:W:6]
    out[1, 1::2, 0::2] = im[3:H:6, 2:W:6]
    out[1, 1::2, 1::2] = im[3:H:6, 5:W:6]

    # 1 B
    out[2, 0::2, 0::2] = im[0:H:6, 1:W:6]
    out[2, 0::2, 1::2] = im[0:H:6, 3:W:6]
    out[2, 1::2, 0::2] = im[3:H:6, 0:W:6]
    out[2, 1::2, 1::2] = im[3:H:6, 4:W:6]

    # 4 R
    out[3, 0::2, 0::2] = im[1:H:6, 2:W:6]
    out[3, 0::2, 1::2] = im[2:H:6, 5:W:6]
    out[3, 1::2, 0::2] = im[5:H:6, 2:W:6]
    out[3, 1::2, 1::2] = im[4:H:6, 5:W:6]

    # 5 B
    out[4, 0::2, 0::2] = im[2:H:6, 2:W:6]
    out[4, 0::2, 1::2] = im[1:H:6, 5:W:6]
    out[4, 1::2, 0::2] = im[4:H:6, 2:W:6]
    out[4, 1::2, 1::2] = im[5:H:6, 5:W:6]

    out[5, :, :] = im[1:H:3, 0:W:3]
    out[6, :, :] = im[1:H:3, 1:W:3]
    out[7, :, :] = im[2:H:3, 0:W:3]
    out[8, :, :] = im[2:H:3, 1:W:3]
    return out


def generate_noisy_obs(y):
    def sample_params():
        # log_a = np.random.uniform(low=np.log(1e-4), high=np.log(1.2e-3))
        log_a = np.random.uniform(low=np.log(1e-4), high=np.log(0.012))
        log_b = np.random.standard_normal() * 0.26 + 2.18 * log_a
        log_c = np.random.uniform(low=np.log(1e-9), high=np.log(1e-7))
        log_d = np.random.uniform(low=np.log(1e-8), high=np.log(1e-6))
        a = np.exp(log_a)
        b = np.exp(log_b)
        c = np.exp(log_c)
        d = np.exp(log_d)
        return a, b, c, d
    
    # sig_read = 10. ** np.random.uniform(low=-3., high=-1.5)
    # sig_shot = 10. ** np.random.uniform(low=-2., high=-1.)
    
    ratio = np.random.uniform(low=100, high=300)
    # ratio = 1
    
    a, b, c, d = sample_params()
    chi = 1 / a
    q = 1 / (2**14)
    
    y = y / ratio

    # Burst denoising
    # shot = np.random.randn(*y.shape).astype(np.float32) * np.sqrt(np.maximum(y, 1e-10)) * sig_shot
    # read = np.random.randn(*y.shape).astype(np.float32) * sig_read
    # z = y + shot + read
    
    if a > 1.2e-3:
        z = np.random.poisson(chi*y).astype(np.float32) / chi
    else:
        z = y + np.random.randn(*y.shape).astype(np.float32) * np.sqrt(np.maximum(y*a, 1e-10))
    
    z = z + np.random.randn(*y.shape).astype(np.float32) * np.sqrt(np.maximum(b, 1e-10)) # Gaussian noise
    z = z + np.random.randn(1, y.shape[1], 1).astype(np.float32) * np.sqrt(c) # banding noise
    z = z + np.random.uniform(low=-0.5*q, high=0.5*q) # quantisation error

    z = z * ratio

    return z


class ELDEvalDataset(BaseDataset):
    def __init__(self, basedir, camera_suffix, scenes=None, img_ids=None):
        super(ELDEvalDataset, self).__init__()
        self.basedir = basedir
        self.camera_suffix = camera_suffix # ('Canon', '.CR2')
        self.scenes = scenes
        self.img_ids = img_ids
        # self.input_dict = {}
        # self.target_dict = {}
        
    def __getitem__(self, i):
        camera, suffix = self.camera_suffix
        
        scene_id = i // len(self.img_ids)
        img_id = i % len(self.img_ids)

        scene = 'scene-{}'.format(self.scenes[scene_id])

        datadir = join(self.basedir, camera, scene)

        input_path = join(datadir, 'IMG_{:04d}{}'.format(self.img_ids[img_id], suffix))

        gt_ids = np.array([1, 6, 11, 16])
        ind = np.argmin(np.abs(self.img_ids[img_id] - gt_ids))
        
        target_path = join(datadir, 'IMG_{:04d}{}'.format(gt_ids[ind], suffix))

        iso, expo = metainfo(target_path)
        target_expo = iso * expo
        iso, expo = metainfo(input_path)

        ratio = target_expo / (iso * expo)

        # if input_path not in self.input_dict:
        #     with rawpy.imread(input_path) as raw:
        #         self.input_dict[input_path] = pack_raw(raw) * ratio
        
        with rawpy.imread(input_path) as raw:
            input = pack_raw_bayer(raw) * ratio            

        # if target_path not in self.target_dict:
        with rawpy.imread(target_path) as raw:
            target = pack_raw_bayer(raw)
            # self.target_dict[target_path] = pack_raw_bayer(raw)

        # input = self.input_dict[input_path]
        # target = self.target_dict[target_path]

        input = np.maximum(np.minimum(input, 1.0), 0)
        target = np.maximum(np.minimum(target, 1.0), 0)
        input = np.ascontiguousarray(input)
        target = np.ascontiguousarray(target)        

        data = {'input': input, 'target': target, 'fn':input_path, 'rawpath': target_path}
        
        return data

    def __len__(self):
        return len(self.scenes) * len(self.img_ids)
