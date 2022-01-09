"""Create lmdb dataset"""
import lmdb
import rawpy
import numpy as np 
import os
from os.path import join
from itertools import product
import process
import pickle


# modify to your own path
sourcedir = './data/SID/Sony'
destdir = './data/Train'


def crop_center(img,cropx,cropy):
    _,y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[:, starty:starty+cropy,startx:startx+cropx]


def pack_raw_bayer(raw):
    #pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    raw_pattern = raw.raw_pattern
    R = np.where(raw_pattern==0)
    G1 = np.where(raw_pattern==1)
    B = np.where(raw_pattern==2)
    G2 = np.where(raw_pattern==3)

    white_level = 16383

    img_shape = im.shape
    H = img_shape[0] - img_shape[0] % 2
    W = img_shape[1] - img_shape[1] % 2

    out = np.stack((im[R[0][0]:H:2,R[1][0]:W:2], #RGBG
                    im[G1[0][0]:H:2,G1[1][0]:W:2],
                    im[B[0][0]:H:2,B[1][0]:W:2],
                    im[G2[0][0]:H:2,G2[1][0]:W:2]), axis=0).astype(np.float32)

    black_level = np.array(raw.black_level_per_channel)[:,None,None].astype(np.float32)

    out = (out - black_level) / (white_level - black_level)
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

    out = np.zeros((9, H // 3, W // 3))

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


def read_paired_fns(filename):
    with open(filename) as f:
        fns = f.readlines()
        fns = [tuple(fn.strip().split(' ')) for fn in fns]
    return fns


def Data2Volume(data, ksizes, strides):
    """
    Construct Volumes from Original High Dimensional (D) Data
    """
    dshape = data.shape
    PatNum = lambda l, k, s: (np.floor( (l - k) / s ) + 1)    

    TotalPatNum = 1
    for i in range(len(ksizes)):
        TotalPatNum = TotalPatNum * PatNum(dshape[i], ksizes[i], strides[i])
        
    V = np.zeros([int(TotalPatNum)]+ksizes); # create D+1 dimension volume

    args = [range(kz) for kz in ksizes]
    for s in product(*args):        
        s1 = (slice(None),) + s
        s2 = tuple([slice(key, -ksizes[i]+key+1 or None, strides[i]) for i, key in enumerate(s)])     
        V[s1] = np.reshape(data[s2], (-1,))

    return V


def compute_expo_ratio(input_fn, target_fn):        
    in_exposure = float(input_fn.split('_')[-1][:-5])
    gt_exposure = float(target_fn.split('_')[-1][:-5])
    ratio = min(gt_exposure / in_exposure, 300)
    return ratio


def create_lmdb_train(
    fns, targetdir, 
    ksize, stride, cfa='bayer', transform=None, ratios=None, srgb=False,
    seed=2019, uint16=True, CRF=None):

    def preprocess(data):
        _, h, w = data.shape
        if transform is not None:
            data = transform(data)
        
        cropy = int((h - ksize[1]) / stride[1]) * stride[1] + ksize[1] # c h w 
        cropx = int((w - ksize[2]) / stride[2]) * stride[2] + ksize[2] # w

        data = crop_center(data, cropx, cropy)
        patch_data = Data2Volume(data, list(ksize), list(stride))   
        patch_data = patch_data.astype(np.uint16) if uint16 else patch_data.astype(np.float32)  
        return patch_data

    np.random.seed(seed)    
    
    if cfa == 'bayer':
        pack_raw = pack_raw_bayer
    else:
        pack_raw = pack_raw_xtrans
 
    # calculate the shape of dataset    
    with rawpy.imread(fns[0]) as raw:
        data = pack_raw(raw)        

        if srgb:
            data = process.raw2rgb(data, raw, CRF)

    if uint16:
        data = (data * 65535).astype(np.uint16)

    N = data.shape[0]
    print(data.shape)

    # We need to prepare the database for the size. We'll set it 2 times
    # greater than what we theoretically need.    
    map_size = data.nbytes * len(fns) * 2
    print('map size (GB):', map_size / 1024 / 1024 / 1024)

    # import ipdb; ipdb.set_trace()
    if os.path.exists(targetdir+'.db'):
        raise Exception('database already exist!')
    env = lmdb.open(targetdir+'.db', map_size=map_size, writemap=True)
    meta_info = {}    
    meta_info['shape'] = ksize
    meta_info['dtype'] = np.float32 if not uint16 else np.uint16
    
    with env.begin(write=True) as txn:
        # txn is a Transaction object
        k = 0
        for i, fn in enumerate(fns):
            # try:
            with rawpy.imread(fn) as raw:
                X = pack_raw(raw)
                                
                # record meta info to reproduce ISP
                wb = np.array(raw.camera_whitebalance) 
                wb /= wb[1]
                ccm = raw.rgb_camera_matrix[:3, :3]

                if ratios is not None: 
                    X = X * ratios[i]

                if srgb:
                    X = process.raw2rgb(X, raw, CRF)

                X = np.clip(X, 0, 1)

                if uint16:
                    X = (X * 65535).astype(np.uint16)
            # except:
            #     print('loading', fn, 'fail')
            #     continue

            X = preprocess(X)
            N = X.shape[0]
            print(N)
            for j in range(N):                
                meta_info[k] = (wb, ccm)                
                data = X[j].tobytes()
                str_id = '{:08}'.format(k)
                k += 1
                txn.put(str_id.encode('ascii'), data)
            print('load mat (%d/%d): %s' %(i,len(fns),fn))

        print('done')

    pickle.dump(meta_info, open(join(targetdir+'.db', 'meta_info.pkl'), "wb"))
    print('Finish creating lmdb meta info.')


def create_sony_dataset(num_samples=None): # synthetic data
    print('create sid sony dataset...')
    # sourcedir = '/media/kaixuan/DATA/Papers/Code/Data/Raw/SID/Sony/long'
    sourcedir_ = join(sourcedir, 'long')

    fns = read_paired_fns('./dataset/Sony_train.txt')
    fns = list(set([fn[1] for fn in fns]))
    fns = sorted([join(sourcedir_, fn) for fn in fns])
    if num_samples is not None:
        fns = fns[:num_samples]

    create_lmdb_train(
        fns, join(destdir, 'SID_Sony_Raw'),
        ksize=(4, 512, 512),
        stride=(4, 512, 512),
        srgb=False, 
    )


def create_sony_dataset_paired(num_samples=None): # paired real data
    print('create sid sony dataset...')
    # sourcedir = '/media/kaixuan/DATA/Papers/Code/Data/Raw/SID/Sony/'
    
    fns = sorted(read_paired_fns('./dataset/Sony_train.txt'))
    if num_samples is not None:
        fns = fns[:num_samples]
    ratios = [compute_expo_ratio(fn[0], fn[1]) for fn in fns] 

    create_lmdb_train(
        [join(sourcedir, 'short', fn[0]) for fn in fns], join(destdir, 'SID_Sony_input_Raw'),
        ksize=(4, 512, 512),
        stride=(4, 512, 512), ratios=ratios,
        srgb=False,
    )
    
    create_lmdb_train(
        [join(sourcedir, 'long', fn[1]) for fn in fns], join(destdir, 'SID_Sony_target_Raw'),
        ksize=(4, 512, 512),
        stride=(4, 512, 512),
        srgb=False,
    )


def create_sony_dataset_SRGB(num_samples=None): # synthetic data
    print('create sid sony SRGB dataset...')
    # sourcedir = '/media/kaixuan/DATA/Papers/Code/Data/Raw/SID/Sony/long'
    sourcedir_ = join(sourcedir, 'long')
    
    fns = read_paired_fns('./dataset/Sony_train.txt')
    fns = list(set([fn[1] for fn in fns]))
    fns = sorted([join(sourcedir_, fn) for fn in fns])
    if num_samples is not None:
        fns = fns[:num_samples]

    # use calibrated CRF from SonyA7S2
    CRF = process.load_CRF()
    
    create_lmdb_train(
        fns, join(destdir, 'SID_Sony_SRGB_CRF'),
        ksize=(3, 512, 512),
        stride=(3, 512, 512),
        srgb=True,
        CRF=CRF
    )

    # create_lmdb_train(
    #     fns, join(destdir, 'SID_Sony_SRGB'),
    #     ksize=(3, 512, 512),
    #     stride=(3, 512, 512),
    #     srgb=True,
    #     CRF=None
    # )        


def create_sony_dataset_SRGB_paired(num_samples=None): # paired real data 
    print('create sid sony dataset...')
    # sourcedir = '/media/kaixuan/DATA/Papers/Code/Data/Raw/SID/Sony/'
    
    fns = sorted(read_paired_fns('./dataset/Sony_train.txt'))
    if num_samples is not None:
        fns = fns[:num_samples]
    ratios = [compute_expo_ratio(fn[0], fn[1]) for fn in fns]

    CRF = process.load_CRF()

    create_lmdb_train(
        [join(sourcedir, 'short', fn[0]) for fn in fns], join(destdir, 'SID_Sony_input_SRGB_CRF'),
        ksize=(3, 512, 512),
        stride=(3, 512, 512), ratios=ratios,
        srgb=True, CRF=CRF
    )
    
    create_lmdb_train(
        [join(sourcedir, 'long', fn[1]) for fn in fns], join(destdir, 'SID_Sony_target_SRGB_CRF'),
        ksize=(3, 512, 512),
        stride=(3, 512, 512),
        srgb=True, CRF=CRF
    )


if __name__ == '__main__':        
    create_sony_dataset()
    # create_sony_dataset_paired()
    # create_sony_dataset_SRGB(10)
    # create_sony_dataset_SRGB_paired(10)
