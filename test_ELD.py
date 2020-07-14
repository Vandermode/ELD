from os.path import join
from options.eld.base_options import BaseOptions 
from engine import Engine
import torch
import torch.backends.cudnn as cudnn
import data.sid_dataset as datasets
import util.util as util
import data
import glob
import os
from scipy.io import loadmat, savemat

opt = BaseOptions().parse()

cudnn.benchmark = True

if opt.debug:
    opt.display_freq = 20
    opt.print_freq = 20
    opt.nEpochs = 40
    opt.max_dataset_size = 10
    opt.no_log = False
    opt.nThreads = 0
    opt.decay_iter = 0
    opt.serial_batches = True
    opt.no_flip = True

"""Main Loop"""
engine = Engine(opt)

databasedir = '/media/kaixuan/DATA/Papers/Code/Data/Raw/ELD'
method = opt.name
scenes = list(range(1, 10+1))
cameras = ['CanonEOS5D4', 'CanonEOS70D', 'CanonEOS700D', 'NikonD850', 'SonyA7S2']     
suffixes = ['.CR2', '.CR2', '.CR2', '.nef', '.ARW']


if opt.include is not None:
    cameras = cameras[opt.include:opt.include+1]
    suffixes = suffixes[opt.include:opt.include+1]
else:
    cameras = ['CanonEOS70D', 'CanonEOS700D', 'NikonD850', 'SonyA7S2']     
    suffixes = ['.CR2', '.CR2', '.nef', '.ARW']


img_ids_set = [[4, 9, 14], [5, 10, 15]]
# for scene in scenes:
for i, img_ids in enumerate(img_ids_set):
    print(img_ids)
    # eval_datasets = [datasets.ELDEvalDataset(databasedir, camera_suffix, scenes=[scene], img_ids=img_ids) for camera_suffix in zip(cameras, suffixes)]
    eval_datasets = [datasets.ELDEvalDataset(databasedir, camera_suffix, scenes=scenes, img_ids=img_ids) for camera_suffix in zip(cameras, suffixes)]

    eval_dataloaders = [torch.utils.data.DataLoader(
        eval_dataset, batch_size=1, shuffle=False,
        num_workers=opt.nThreads, pin_memory=True) for eval_dataset in eval_datasets]

    # cameras = ['CanonEOS5D4', 'HuaweiHonor10']
    psnrs = []
    ssims = []
    for camera, dataloader in zip(cameras, eval_dataloaders):
        print('Eval camera {}'.format(camera))
        # res = engine.eval(dataloader, dataset_name='eld_eval_{}'.format(camera), correct=True, crop=False, savedir='res-eld/{}_scene_{}'.format(camera, scene))
        res = engine.eval(dataloader, dataset_name='eld_eval_{}'.format(camera), correct=True, crop=True)
        psnrs.append(res['PSNR'])
        ssims.append(res['SSIM'])
