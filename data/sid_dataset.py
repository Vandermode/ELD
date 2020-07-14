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


class SynDataset(BaseDataset): # for ELD-D dataset
    def __init__(self, dataset, size=None, flag=None, noise_maker=None, augment=True, repeat=1, cfa='bayer', num_burst=1, darken=False):
        super(SynDataset, self).__init__()        
        self.size = size
        self.dataset = dataset
        self.flag = flag
        self.repeat = repeat
        self.noise_maker = noise_maker
        self.cfa = cfa
        self.num_burst = num_burst
        self.darken = darken
        self.augment = augment
        if darken:
            print('[i] enable darken')
            
    def __getitem__(self, i):
        target = self.dataset[i]        

        if self.darken: # only applicable in raw2raw setting
            dark_factor = np.random.uniform(0.1, 1)
            target = target * dark_factor

        if self.augment:
            if np.random.randint(2, size=1)[0] == 1:  # random flip
                target = np.flip(target, axis=1)
            if np.random.randint(2, size=1)[0] == 1:
                target = np.flip(target, axis=2)
            if np.random.randint(2, size=1)[0] == 1:  # random transpose
                target = np.transpose(target, (0, 2, 1))

        target = np.maximum(np.minimum(target, 1.0), 0)
        target = np.ascontiguousarray(target)

        if self.num_burst > 1:            
            inputs = []
            params = self.noise_maker._sample_params()     
            for k in range(self.num_burst):           
                inputs.append(self.noise_maker(target, params=params))
            input = np.concatenate(inputs, axis=0)
        else:
            input = self.noise_maker(target)
        
        input = np.maximum(np.minimum(input, 1.0), 0)
        input = np.ascontiguousarray(input)

        dic =  {'input': input, 'target': target}        

        if self.flag is not None:
            dic.update(self.flag)

        return dic

    def __len__(self):
        size = self.size or len(self.dataset)
        return size * self.repeat


class SynLMDBDataset(BaseDataset):  # generate noisy image only 
    def __init__(self, dataset, size=None, flag=None, noise_maker=None, repeat=1, cfa='bayer', num_burst=1):
        super(SynLMDBDataset, self).__init__()        
        self.size = size
        self.dataset = dataset
        self.flag = flag
        self.repeat = repeat
        self.noise_maker = noise_maker
        self.cfa = cfa
        self.num_burst = num_burst

    def __getitem__(self, i):
        data = self.dataset[i]        

        if self.num_burst > 1:            
            inputs = []
            params = self.noise_maker._sample_params()     
            for k in range(self.num_burst):           
                # inputs.append(self.noise_maker(data))
                inputs.append(self.noise_maker(data, params=params))
            input = np.concatenate(inputs, axis=0)
        else:
            input = self.noise_maker(data)
        
        input = np.maximum(np.minimum(input, 1.0), 0)
        input = np.ascontiguousarray(input)

        return input
        
    def __len__(self):
        size = self.size or len(self.dataset)
        return size * self.repeat


class SynVideoDataset(BaseDataset):
    def __init__(self, datasets, size=None, flag=None, noise_maker=None, repeat=1, cfa='bayer', augment=True, darken=False, video_output=False):
        super(SynVideoDataset, self).__init__()        
        self.size = size
        self.datasets = datasets
        self.flag = flag
        self.repeat = repeat
        self.noise_maker = noise_maker
        self.cfa = cfa
        self.augment = augment
        self.idxs = np.arange(len(self.datasets))
        self.darken = darken
        self.video_output = video_output
        if darken:
            print('[i] enable darken')

    def __getitem__(self, i):
        idxs = self.idxs        
        target_idx = len(self.datasets) // 2

        if self.darken: # only applicable in raw2raw setting
            dark_factor = np.random.uniform(0.1, 1)
        else:
            dark_factor = 1

        inputs_clean = [self.datasets[ind][i] * dark_factor for ind in idxs]

        if self.video_output:
            target_image = np.concatenate(inputs_clean, axis=0)
        else:
            target_image = inputs_clean[target_idx]

        inputs = []
        params = self.noise_maker._sample_params()     
        for k in range(len(self.datasets)):           
            inputs.append(self.noise_maker(inputs_clean[k], params=params))

        input = np.concatenate(inputs, axis=0)
        target = target_image

        if self.augment:
            W = target_image.shape[2]
            H = target_image.shape[1]

            if np.random.randint(2, size=1)[0] == 1:  # random flip
                target = np.flip(target, axis=1)
                input = np.flip(input, axis=1)
            if np.random.randint(2, size=1)[0] == 1:
                target = np.flip(target, axis=2)
                input = np.flip(input, axis=2)
            if np.random.randint(2, size=1)[0] == 1:  # random transpose
                target = np.transpose(target, (0, 2, 1))
                input = np.transpose(input, (0, 2, 1))        

        input = np.maximum(np.minimum(input, 1.0), 0)
        input = np.ascontiguousarray(input)
        target = np.ascontiguousarray(target)

        dic =  {'input': input, 'target': target}
        return dic
        
    def __len__(self):
        size = self.size or len(self.datasets[0])
        return size * self.repeat


class RealDataset(BaseDataset):
    def __init__(self, datadir, fns=None, size=None, amp_ratio=1, srgb=False):
        super(RealDataset, self).__init__()
        self.size = size
        self.datadir = datadir
        self.fns = fns or [fn for fn in os.listdir(datadir) if os.path.isfile(join(datadir, fn))]
        self.amp_ratio = amp_ratio
        self.srgb = srgb

        if size is not None:
            self.fns = self.fns[:size]
        
    def __getitem__(self, index):
        input_fn = self.fns[index]

        input_path = join(self.datadir, input_fn)
        
        if self.srgb:
            with Image.open(input_path) as srgb:
                input = np.array(srgb, dtype=np.float32).transpose((2, 0, 1)) / 255 * self.amp_ratio
        else:
            with rawpy.imread(input_path) as raw:
                input = pack_raw_bayer(raw) * self.amp_ratio

        input = np.maximum(np.minimum(input, 1.0), 0)
        input = np.ascontiguousarray(input)

        data = {'input': input, 'fn': input_fn, 'rawpath': input_path}
        return data

    def __len__(self):
        if self.size is not None:
            return min(len(self.fns), self.size)
        else:
            return len(self.fns)


class ELDDVideoDataset(BaseDataset):
    def __init__(self, datadir, fns, size=None, start_frame_idx=2, end_frame_idx=4):
        super(ELDDVideoDataset, self).__init__()
        self.size = size
        self.datadir = datadir
        self.fns = fns
        self.cfa = 'bayer'
        self.start_frame_idx = start_frame_idx
        self.end_frame_idx = end_frame_idx
        self.pack_raw = pack_raw_bayer 

        print('[i] video output')

        if size is not None:
            self.fns = self.fns[:size]
        
    def __getitem__(self, i):
        i = i % len(self.fns)
        name = self.fns[i]
        sf_idx = self.start_frame_idx
        ef_idx = self.end_frame_idx        

        target_files = sorted(glob.glob(join(self.datadir, name, '*.ARW')))
        target_files = target_files[sf_idx:ef_idx+1] 

        in_files = sorted(glob.glob(join(self.datadir, 'mat', name, '*.mat')))
        in_files = in_files[sf_idx:ef_idx+1]         
        
        targets = []

        for target_file in target_files:
            with rawpy.imread(target_file) as raw:                    
                target_k = self.pack_raw(raw) 
                targets.append(target_k)

        target = np.concatenate(targets, axis=0).astype(np.float32)
 
        inputs = []
        
        for in_file in in_files:
            input_k = loadmat(in_file)['raw_packed'] / 65535.
            inputs.append(input_k)

        input = (np.concatenate(inputs, axis=0)).astype(np.float32)
        input = np.ascontiguousarray(input)

        input = np.clip(input, 0, 1)
        target = np.clip(target, 0, 1)

        input = np.ascontiguousarray(input)
        target = np.ascontiguousarray(target)

        dic =  {'input': input, 'target': target, 'fn': name, 'cfa': self.cfa, 'rawpath': target_files[0], 'video_input': True}

        if ef_idx - sf_idx > 4:
            dic.update({'video_output': True})
        
        return dic

    def __len__(self):
        return len(self.fns)


class ELDDImageDataset(BaseDataset):
    def __init__(self, datadir, fns, size=None, frame_id=0):
        super(ELDDImageDataset, self).__init__()
        self.size = size
        self.datadir = datadir
        self.fns = fns
        self.cfa = 'bayer'
        self.frame_id = frame_id
        self.pack_raw = pack_raw_bayer 

        if size is not None:
            self.fns = self.fns[:size]
        
    def __getitem__(self, i):
        i = i % len(self.fns)
        name = self.fns[i]
        idx = self.frame_id

        target_files = sorted(glob.glob(join(self.datadir, name, '*.ARW')))
        target_file = target_files[idx] 

        in_files = sorted(glob.glob(join(self.datadir, 'mat', name, '*.mat')))
        in_file = in_files[idx]         
        
        with rawpy.imread(target_file) as raw:                    
            target = self.pack_raw(raw).astype(np.float32)
 
        input = (loadmat(in_file)['raw_packed'] / 65535.).astype(np.float32)

        input = np.clip(input, 0, 1)
        target = np.clip(target, 0, 1)

        # input = crop_center(input, 512, 512)
        # target = crop_center(target, 512, 512)

        input = np.ascontiguousarray(input)
        target = np.ascontiguousarray(target)

        dic =  {'input': input, 'target': target, 'fn': name, 'cfa': self.cfa, 'rawpath': target_file}

        return dic

    def __len__(self):
        return len(self.fns)


class ELDDRealDataset(BaseDataset):
    def __init__(self, datadir, fns=None, size=None, amp_ratio=100, start_frame_idx=2, end_frame_idx=4):
        super(ELDDRealDataset, self).__init__()
        self.size = size
        self.datadir = datadir
        self.fns = fns
        self.amp_ratio = amp_ratio
        self.start_frame_idx = start_frame_idx
        self.end_frame_idx = end_frame_idx
        self.pack_raw = pack_raw_bayer 

        if size is not None:
            self.fns = self.fns[:size]
        
    def __getitem__(self, i):
        i = i % len(self.fns)
        name = self.fns[i]
        sf_idx = self.start_frame_idx
        ef_idx = self.end_frame_idx

        in_files = sorted(glob.glob(join(self.datadir, name, '*.ARW')))
        in_files = in_files[sf_idx:ef_idx+1]  
            
        inputs = []        
        for in_file in in_files:
            with rawpy.imread(in_file) as raw_input:                    
                input_image = self.pack_raw(raw_input) * self.amp_ratio
                inputs.append(input_image)

        if len(inputs) > 1:
            input = (np.concatenate(inputs, axis=0)).astype(np.float32)
        else:
            input = inputs[0]

        input = np.ascontiguousarray(input)
        input = np.maximum(np.minimum(input, 1.0), 0)
        input = np.ascontiguousarray(input)

        dic =  {'input': input, 'target': input, 'fn': name, 'cfa': 'bayer', 'rawpath': in_files[0]}

        if ef_idx > sf_idx:
            dic.update({'video_input': True})
        if ef_idx - sf_idx > 4:
            dic.update({'video_output': True})

        return dic

    def __len__(self):
        if self.size is not None:
            return min(len(self.fns), self.size)
        else:
            return len(self.fns)


class ISPLMDBDataset(BaseDataset):
    def __init__(self, db_path, dtype=np.float32, noise_maker=None, cfa='bayer', meta_info=None):
        super(ISPLMDBDataset, self).__init__()        
        self.dataset = LMDBDataset(db_path=db_path, dtype=dtype)
        self.noise_maker = noise_maker
        self.cfa = cfa

        if meta_info is None:
            self.meta_info = pickle.load(open(join(db_path, 'meta_info.pkl'), 'rb'))
        else:
            self.meta_info = meta_info
        
    def __getitem__(self, i):
        data = self.dataset[i]
        (wb, ccm) = self.meta_info[i]
        
        if self.noise_maker is not None:        
            input = self.noise_maker(data)
        else:
            input = data

        input = np.maximum(np.minimum(input, 1.0), 0)
        input = process.raw2rgb_v2(input, wb, ccm)
        input = np.maximum(np.minimum(input, 1.0), 0)
        input = np.ascontiguousarray(input)

        return input

    def __len__(self):
        return len(self.dataset)


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


class SIDEvalDataset(BaseDataset):  # read .mat file
    def __init__(self, datadir, paired_fns, size=None, amp_ratio=None, cfa='bayer', num_burst=1):
        super(SIDEvalDataset, self).__init__()
        self.size = size
        self.datadir = datadir
        self.paired_fns = paired_fns
        self.amp_ratio = amp_ratio
        self.cfa = cfa
        self.pack_raw = pack_raw_bayer if cfa == 'bayer' else pack_raw_xtrans
        if size is not None:
            self.paired_fns = self.paired_fns[:size]

        self.num_burst = num_burst

    def __getitem__(self, i):
        i = i % len(self.paired_fns)
        input_fn, target_fn = self.paired_fns[i]

        input_fn = input_fn.replace('.ARW', '.mat')
        target_fn = target_fn.replace('.ARW', '.mat')

        target_path = join(self.datadir, 'long', 'mat', target_fn)
        
        mat = loadmat(target_path)
        meta = mat['meta']

        wb = mat['meta'][:, 0].item()
        ccm = mat['meta'][:, 1].item()

        target = (mat['data'] / 65535.).astype(np.float32)
        # target = process.raw2rgb_v2(target, wb, ccm)

            # self.target_dict[target_fn] = target
        # else:
        #     target = self.target_dict[target_fn]        
        # if input_fn not in self.input_dict:

        inputs = []
        for k in range(self.num_burst):
            input_fn_k = input_fn.replace('_00_', '_{:02d}_'.format(k))
            input_path = join(self.datadir, 'short', 'mat', input_fn_k)

            mat = loadmat(input_path)
            inputs.append(mat['data'])

        # input = np.stack(inputs, axis=0) / 65535.        
        # input = np.mean(input, axis=0)
        input = (np.concatenate(inputs, axis=0) / 65535.).astype(np.float32)

        # input = process.raw2rgb_v2(input, wb, ccm)
        input = np.ascontiguousarray(input)

        # self.input_dict[input_fn] = input
        # else:
        #     input = self.input_dict[input_fn]
        
        dic =  {'input': input, 'target': target, 'fn': input_fn, 'cfa': self.cfa}
        
        return dic

    def __len__(self):
        return len(self.paired_fns)


class RealDataset(BaseDataset):
    def __init__(self, datadir, fns=None, size=None, amp_ratio=1, srgb=False):
        super(RealDataset, self).__init__()
        self.size = size
        self.datadir = datadir
        self.fns = fns or [fn for fn in os.listdir(datadir) if os.path.isfile(join(datadir, fn))]
        self.amp_ratio = amp_ratio
        self.srgb = srgb

        if size is not None:
            self.fns = self.fns[:size]
        
    def __getitem__(self, index):
        input_fn = self.fns[index]

        input_path = join(self.datadir, input_fn)
        
        if self.srgb:
            with Image.open(input_path) as srgb:
                input = np.array(srgb, dtype=np.float32).transpose((2, 0, 1)) / 255 * self.amp_ratio
        else:
            with rawpy.imread(input_path) as raw:
                input = pack_raw_bayer(raw) * self.amp_ratio

        input = np.maximum(np.minimum(input, 1.0), 0)
        input = np.ascontiguousarray(input)

        data = {'input': input, 'fn': input_fn, 'rawpath': input_path}
        return data

    def __len__(self):
        if self.size is not None:
            return min(len(self.fns), self.size)
        else:
            return len(self.fns)


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


class ELDTrainDataset(BaseDataset):
    def __init__(self, target_dataset, input_datasets, size=None, flag=None, augment=True, cfa='bayer'):
        super(ELDTrainDataset, self).__init__()
        self.size = size
        self.target_dataset = target_dataset
        self.input_datasets = input_datasets
        self.flag = flag
        self.augment = augment
        self.cfa = cfa

    def __getitem__(self, i):
        N = len(self.input_datasets)
        input_image = self.input_datasets[i%N][i//N]
        target_image = self.target_dataset[i//N]        

        target = target_image 
        input = input_image       
    
        if self.augment:
            W = target_image.shape[2]
            H = target_image.shape[1]

            if np.random.randint(2, size=1)[0] == 1:  # random flip
                target = np.flip(target, axis=1)
                input = np.flip(input, axis=1)
            if np.random.randint(2, size=1)[0] == 1:
                target = np.flip(target, axis=2)
                input = np.flip(input, axis=2)
            if np.random.randint(2, size=1)[0] == 1:  # random transpose
                target = np.transpose(target, (0, 2, 1))
                input = np.transpose(input, (0, 2, 1))        

        input = np.maximum(np.minimum(input, 1.0), 0)
        input = np.ascontiguousarray(input)
        target = np.ascontiguousarray(target)

        dic =  {'input': input, 'target': target}        

        if self.flag is not None:
            dic.update(self.flag)
        
        return dic

    def __len__(self):
        size = self.size or len(self.target_dataset) * len(self.input_datasets)
        return size


class BurstDataset(BaseDataset):
    def __init__(self, target_dataset, input_datasets, fns=None, size=None, flag=None, augment=True, cfa='bayer', num_burst=None):
        super(BurstDataset, self).__init__()
        self.size = size
        self.target_dataset = target_dataset
        self.input_datasets = input_datasets
        self.flag = flag
        self.augment = augment
        self.cfa = cfa
        self.fns = fns
        if num_burst is None:
            self.num_burst = len(input_datasets)
        else:
            self.num_burst = num_burst

        self.reset()

    def reset(self):
        # print('[i] reset dataset...')
        self.idxs = np.arange(len(self.input_datasets))[:self.num_burst]
        print('[i] idx: {}'.format(self.idxs))
        # self.idxs = np.random.choice(np.arange(len(self.input_datasets)), self.num_burst, replace=False)

    def __getitem__(self, i):
        # idxs = np.random.choice(np.arange(len(self.input_datasets)), self.num_burst, replace=False)

        idxs = self.idxs        
        inputs = [self.input_datasets[ind][i] for ind in idxs]
        
        input_image = np.concatenate(inputs, axis=0)

        target_image = self.target_dataset[i]

        target = target_image
        input = input_image
    
        if self.augment:
            W = target_image.shape[2]
            H = target_image.shape[1]

            if np.random.randint(2, size=1)[0] == 1:  # random flip
                target = np.flip(target, axis=1)
                input = np.flip(input, axis=1)
            if np.random.randint(2, size=1)[0] == 1:
                target = np.flip(target, axis=2)
                input = np.flip(input, axis=2)
            if np.random.randint(2, size=1)[0] == 1:  # random transpose
                target = np.transpose(target, (0, 2, 1))
                input = np.transpose(input, (0, 2, 1))

        input = np.maximum(np.minimum(input, 1.0), 0)
        input = np.ascontiguousarray(input)
        target = np.ascontiguousarray(target)

        dic =  {'input': input, 'target': target}        

        if self.fns is not None:
            fn = self.fns[i]
            dic.update({'fn': fn})

        if self.flag is not None:
            dic.update(self.flag)
        
        return dic

    def __len__(self):
        size = self.size or len(self.target_dataset)
        return size


class N2NLMDBDataset(BaseDataset):
    def __init__(self, input_datasets, size=None, flag=None, augment=True, cfa='bayer', meta_info=None):
        super(N2NLMDBDataset, self).__init__()
        self.size = size
        self.input_datasets = input_datasets
        self.flag = flag
        self.augment = augment
        self.cfa = cfa
        self.reset()
        self.meta_info = meta_info

    def reset(self):
        print('[i] reset dataset...')
        self.idxs = np.random.choice(np.arange(len(self.input_datasets)), 2, replace=False)

    def __getitem__(self, i):

        idxs = self.idxs 
        # idxs = np.random.choice(np.arange(len(self.input_datasets)), 2, replace=False)

        input_image = self.input_datasets[idxs[0]][i]
        target_image = self.input_datasets[idxs[1]][i]

        target = target_image
        input = input_image        

        if self.meta_info is not None:
            (wb, ccm) = self.meta_info[i]
            input = process.raw2rgb_v2(input, wb, ccm)
            target = process.raw2rgb_v2(target, wb, ccm)            
    
        if self.augment:
            W = target_image.shape[2]
            H = target_image.shape[1]

            if np.random.randint(2, size=1)[0] == 1:  # random flip
                target = np.flip(target, axis=1)
                input = np.flip(input, axis=1)
            if np.random.randint(2, size=1)[0] == 1:
                target = np.flip(target, axis=2)
                input = np.flip(input, axis=2)
            if np.random.randint(2, size=1)[0] == 1:  # random transpose
                target = np.transpose(target, (0, 2, 1))
                input = np.transpose(input, (0, 2, 1))

        input = np.maximum(np.minimum(input, 1.0), 0)
        target = np.maximum(np.minimum(target, 1.0), 0)
        input = np.ascontiguousarray(input)
        target = np.ascontiguousarray(target)
                
        dic =  {'input': input, 'target': target}        

        if self.flag is not None:
            dic.update(self.flag)
        
        return dic

    def __len__(self):
        size = self.size or len(self.input_datasets[0])
        return size


class DRVDataset(BaseDataset):
    def __init__(self, datadir, fns, size=None, flag=None, repeat=1, cfa='bayer', memorize=True, stage_in='raw', stage_out='raw', gt_wb=False, frame_id=0):
        super(DRVDataset, self).__init__()
        assert cfa == 'bayer' or cfa == 'xtrans'
        self.size = size
        self.datadir = datadir
        self.fns = fns
        self.flag = flag
        self.patch_size = 512
        self.repeat = repeat
        self.cfa = cfa
        self.frame_id = frame_id

        self.pack_raw = pack_raw_bayer if cfa == 'bayer' else pack_raw_xtrans

        assert stage_in in ['raw', 'srgb']
        assert stage_out in ['raw', 'srgb']                
        self.stage_in = stage_in
        self.stage_out = stage_out
        self.gt_wb = gt_wb

        if size is not None:
            self.fns = self.fns[:size]
        
        self.memorize = memorize
        self.target_dict = {}
        self.target_dict_aux = {}
        self.input_dict = {}

    def __getitem__(self, i, k=None):
        if k is None:
            k = self.frame_id            

        i = i % len(self.fns)
        name = self.fns[i]

        in_files = sorted(glob.glob(join(self.datadir, 'short', name, '*.ARW')))
        in_file = in_files[k]  # evaluate the first frame        
        
        gt_file = glob.glob(join(self.datadir, 'long', name, '*.ARW'))[0]

        if self.memorize:
            if name not in self.target_dict:
                _, expo_long = metainfo(gt_file)
                with rawpy.imread(gt_file) as raw_target:                    
                    target_image = self.pack_raw(raw_target)    
                    wb, ccm = process.read_wb_ccm(raw_target)
                    
                    if self.stage_out == 'srgb':
                        target_image = process.raw2rgb(target_image, raw_target)
                    self.target_dict[name] = target_image
                    self.target_dict_aux[name] = (wb, ccm, expo_long)

            if name not in self.input_dict:
                _, expo_short = metainfo(in_file)
                with rawpy.imread(in_file) as raw_input:

                    wb, ccm, expo_long = self.target_dict_aux[name]
                    input_image = self.pack_raw(raw_input) * (expo_long / expo_short)
                    if self.stage_in == 'srgb':
                        if self.gt_wb:
                            input_image = process.raw2rgb_v2(input_image, wb, ccm)
                        else:
                            input_image = process.raw2rgb(input_image, raw_input)
                    self.input_dict[name] = input_image

            input_image = self.input_dict[name]
            target_image = self.target_dict[name]
            (wb, ccm, expo_long) = self.target_dict_aux[name]
        else:
            _, expo_long = metainfo(gt_file)
            with rawpy.imread(gt_file) as raw_target:                    
                target_image = self.pack_raw(raw_target)    
                wb, ccm = process.read_wb_ccm(raw_target)
                
                if self.stage_out == 'srgb':
                    target_image = process.raw2rgb(target_image, raw_target)

            _, expo_short = metainfo(in_file)
            with rawpy.imread(in_file) as raw_input:
                input_image = self.pack_raw(raw_input) * (expo_long / expo_short)
                if self.stage_in == 'srgb':
                    if self.gt_wb:
                        input_image = process.raw2rgb_v2(input_image, wb, ccm)
                    else:
                        input_image = process.raw2rgb(input_image, raw_input)  

        input = input_image
        target = target_image

        input = np.maximum(np.minimum(input, 1.0), 0)
        target = np.maximum(np.minimum(target, 1.0), 0)
        input = np.ascontiguousarray(input)
        target = np.ascontiguousarray(target)

        dic =  {'input': input, 'target': target, 'fn': name, 'cfa': self.cfa, 'rawpath': gt_file, 'aux': (wb, ccm)}

        if self.flag is not None:
            dic.update(self.flag)
        
        return dic

    def __len__(self):
        return len(self.fns) * self.repeat



class DRVEvalDataset(BaseDataset):
    def __init__(self, datadir, fns, size=None, frame_id=4, num_frames=5):
        super(DRVEvalDataset, self).__init__()
        self.size = size
        self.datadir = datadir
        self.fns = fns
        self.cfa = 'bayer'
        self.frame_id = frame_id

        assert num_frames % 2 == 1
        self.num_frames = num_frames

        self.pack_raw = pack_raw_bayer 

        if size is not None:
            self.fns = self.fns[:size]
        
        self.target_dict = {}
        self.target_dict_aux = {}
        self.input_dict = {}

    def __getitem__(self, i, k=None):
        if k is None:
            k = self.frame_id
        nf = self.num_frames

        i = i % len(self.fns)
        name = self.fns[i]

        in_files = sorted(glob.glob(join(self.datadir, 'short', name, '*.ARW')))
        in_files = in_files[k-nf//2:k+nf//2+1]  # evaluate the first frame
        
        gt_file = glob.glob(join(self.datadir, 'long', name, '*.ARW'))[0]
                
        _, expo_long = metainfo(gt_file)
        with rawpy.imread(gt_file) as raw_target:                    
            target_image = self.pack_raw(raw_target)    

        _, expo_short = metainfo(in_files[0])
        inputs = []
        for in_file in in_files:
            with rawpy.imread(in_file) as raw_input:                    
                input_image = self.pack_raw(raw_input) * (expo_long / expo_short)
                inputs.append(input_image)

        input = (np.concatenate(inputs, axis=0)).astype(np.float32)
        input = np.ascontiguousarray(input)            

        target = target_image

        input = np.maximum(np.minimum(input, 1.0), 0)
        target = np.maximum(np.minimum(target, 1.0), 0)
        input = np.ascontiguousarray(input)
        target = np.ascontiguousarray(target)

        dic =  {'input': input, 'target': target, 'fn': name, 'cfa': self.cfa, 'rawpath': gt_file, 'video_input':True}
        
        return dic

    def __len__(self):
        return len(self.fns)


class DRVTrainDataset(BaseDataset):
    def __init__(self, datadir, fns, size=None, flag=None, repeat=1, stage_in='raw', stage_out='raw', gt_wb=False, max_id=0):
        super(DRVTrainDataset, self).__init__()
        self.size = size
        self.datadir = datadir
        self.fns = fns
        self.flag = flag
        self.patch_size = 512
        self.repeat = repeat
        self.max_id = max_id
        self.cfa = 'bayer'

        self.pack_raw = pack_raw_bayer

        assert stage_in in ['raw', 'srgb']
        assert stage_out in ['raw', 'srgb']                
        self.stage_in = stage_in
        self.stage_out = stage_out
        self.gt_wb = gt_wb

        if size is not None:
            self.fns = self.fns[:size]
        
        self.target_dict = {}
        self.target_dict_aux = {}
        self.input_dict = {}

    def __getitem__(self, i):
        k = np.random.randint(0, self.max_id+1)
        # print(k)
        i = i % len(self.fns)
        name = self.fns[i]

        in_files = sorted(glob.glob(join(self.datadir, 'short', name, '*.ARW')))
        in_file = in_files[k]  # evaluate the first frame
        gt_file = glob.glob(join(self.datadir, 'long', name, '*.ARW'))[0]
                
        if name not in self.target_dict:
            _, expo_long = metainfo(gt_file)
            with rawpy.imread(gt_file) as raw_target:                    
                target_image = self.pack_raw(raw_target)    
                wb, ccm = process.read_wb_ccm(raw_target)
                
                if self.stage_out == 'srgb':
                    target_image = process.raw2rgb(target_image, raw_target)
                self.target_dict[name] = target_image
                self.target_dict_aux[name] = (wb, ccm, expo_long)

        if name not in self.input_dict:
            _, expo_short = metainfo(in_file)
            with rawpy.imread(in_file) as raw_input:

                wb, ccm, expo_long = self.target_dict_aux[name]
                input_image = self.pack_raw(raw_input) * (expo_long / expo_short)
                if self.stage_in == 'srgb':
                    if self.gt_wb:
                        input_image = process.raw2rgb_v2(input_image, wb, ccm)
                    else:
                        input_image = process.raw2rgb(input_image, raw_input)
                self.input_dict[name] = {}
                self.input_dict[name][k] = input_image
        elif k not in self.input_dict[name]:
            _, expo_short = metainfo(in_file)
            with rawpy.imread(in_file) as raw_input:

                wb, ccm, expo_long = self.target_dict_aux[name]
                input_image = self.pack_raw(raw_input) * (expo_long / expo_short)
                if self.stage_in == 'srgb':
                    if self.gt_wb:
                        input_image = process.raw2rgb_v2(input_image, wb, ccm)
                    else:
                        input_image = process.raw2rgb(input_image, raw_input)
                self.input_dict[name][k] = input_image                

        input_image = self.input_dict[name][k]
        target_image = self.target_dict[name]
        (wb, ccm, expo_long) = self.target_dict_aux[name]
        
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

        input = np.maximum(np.minimum(input, 1.0), 0)
        target = np.maximum(np.minimum(target, 1.0), 0)
        input = np.ascontiguousarray(input)
        target = np.ascontiguousarray(target)

        dic =  {'input': input, 'target': target, 'fn': name, 'cfa': self.cfa, 'rawpath': gt_file, 'aux': (wb, ccm)}

        if self.flag is not None:
            dic.update(self.flag)

        return dic

    def __len__(self):
        return len(self.fns) * self.repeat


class RawDataset(BaseDataset):
    def __init__(self, datadir, fns, size=None, flag=None, repeat=1, cfa='bayer', frame_id=0, ratios=None, suffix='.ARW'):
        super(RawDataset, self).__init__()
        assert cfa == 'bayer' or cfa == 'xtrans'
        self.size = size
        self.datadir = datadir
        self.fns = fns
        self.flag = flag
        self.repeat = repeat
        self.cfa = cfa
        self.frame_id = frame_id
        self.ratios = ratios
        self.suffix = suffix

        self.pack_raw = pack_raw_bayer if cfa == 'bayer' else pack_raw_xtrans

        if size is not None:
            self.fns = self.fns[:size]

    def __getitem__(self, i):
        k = self.frame_id            

        i = i % len(self.fns)
        name = self.fns[i]

        raw_files = sorted(glob.glob(join(self.datadir, name, '*{}'.format(self.suffix))))
        raw_file = raw_files[k]   # evaluate the first frame
        
        with rawpy.imread(raw_file) as raw:                    
            data = self.pack_raw(raw)
            if self.ratios is not None:
                data = data * self.ratios[i]

        data = np.maximum(np.minimum(data, 1.0), 0)
        data = np.ascontiguousarray(data)
        
        return data

    def __len__(self):
        return len(self.fns) * self.repeat
