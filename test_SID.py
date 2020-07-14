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
import socket

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


expo_ratio = [100, 250, 300]

if socket.gethostname() == 'HP':
    datadir = '/media/kaixuan/DATA/Papers/Code/Data/Raw/SID/Sony'
else:
    datadir = '/data/weikaixuan/dark/data/Sony'


read_expo_ratio = lambda x: float(x.split('_')[-1][:-5])

train_fns = data.read_paired_fns('./data/Sony_train.txt')
eval_fns = data.read_paired_fns('./data/Sony_val.txt')
test_fns = data.read_paired_fns('./data/Sony_test.txt')


eval_fns_list = [[fn for fn in eval_fns if min(int(read_expo_ratio(fn[1])/read_expo_ratio(fn[0])), 300)==ratio] for ratio in expo_ratio]
test_fns_list = [[fn for fn in test_fns if min(int(read_expo_ratio(fn[1])/read_expo_ratio(fn[0])), 300)==ratio] for ratio in expo_ratio]
eval_fns_list = [lst_1 + lst_2 for lst_1, lst_2 in zip(eval_fns_list, test_fns_list)]


eval_datasets = [datasets.SIDDataset(datadir, fns, memorize=False, size=None, augment=False, stage_in=opt.stage_in, stage_out=opt.stage_out) for fns in eval_fns_list]


eval_dataloaders = [torch.utils.data.DataLoader(
    eval_dataset, batch_size=1, shuffle=False,
    num_workers=opt.nThreads, pin_memory=True) for eval_dataset in eval_datasets]


"""Main Loop"""
engine = Engine(opt)

for ratio, dataloader in zip(expo_ratio, eval_dataloaders):
    print('Eval ratio {}'.format(ratio))
    # engine.eval(dataloader, dataset_name='sid_eval_{}'.format(ratio), correct=True, crop=False, savedir='res-sid')
    engine.eval(dataloader, dataset_name='sid_eval_{}'.format(ratio), correct=True, crop=True)
