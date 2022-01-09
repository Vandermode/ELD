from os.path import join
from options.eld.train_options import TrainOptions
from engine import Engine
import torch
import torch.backends.cudnn as cudnn
import dataset.sid_dataset as datasets
import dataset.lmdb_dataset as lmdb_dataset
import dataset
import numpy as np
import noise
from util import process
from dataset.sid_dataset import worker_init_fn


opt = TrainOptions().parse()

cudnn.benchmark = True

evaldir = './data/SID/Sony'
traindir = './data/Train'

expo_ratio = [100, 250, 300]
read_expo_ratio = lambda x: float(x.split('_')[-1][:-5])

train_fns = dataset.read_paired_fns('./dataset/Sony_train.txt')
eval_fns = dataset.read_paired_fns('./dataset/Sony_val.txt')
test_fns = dataset.read_paired_fns('./dataset/Sony_test.txt')

eval_fns_list = [[fn for fn in eval_fns if min(int(read_expo_ratio(fn[1])/read_expo_ratio(fn[0])), 300)==ratio] for ratio in expo_ratio]
test_fns_list = [[fn for fn in test_fns if min(int(read_expo_ratio(fn[1])/read_expo_ratio(fn[0])), 300)==ratio] for ratio in expo_ratio]
eval_fns_list = [lst_1 + lst_2 for lst_1, lst_2 in zip(eval_fns_list, test_fns_list)]

# evaluate 15 indoor scenes (but you can also evaluate the performance on the whole dataset)
indoor_ids = dataset.read_paired_fns('./SID_Sony_15_paired.txt')
eval_fns_list = [[(fn[0], fn[1]) for fn in indoor_ids if int(fn[2]) == ratio] for ratio in expo_ratio]

cameras = ['CanonEOS5D4', 'CanonEOS70D', 'CanonEOS700D', 'NikonD850', 'SonyA7S2']
noise_model = noise.NoiseModel(model=opt.noise, include=opt.include)

repeat = 1 if opt.max_dataset_size is None else 1288 / opt.max_dataset_size
print('[i] repeat:', repeat)

CRF = None
if opt.crf:
    print('[i] enable CRF')
    CRF = process.load_CRF()

if opt.stage_out == 'srgb':
    target_data = lmdb_dataset.LMDBDataset(join(traindir, 'SID_Sony_SRGB_CRF.db'))
else:
    target_data = lmdb_dataset.LMDBDataset(
        join(traindir, 'SID_Sony_Raw.db'),
        size=opt.max_dataset_size, repeat=repeat)

if opt.stage_in == 'srgb':
    input_data = datasets.ISPDataset(
        lmdb_dataset.LMDBDataset(join(traindir, 'SID_Sony_Raw.db')),
        noise_maker=noise_model, CRF=CRF)
else:
    ## Synthesizing noise on-the-fly by noise model    
    # input_data = datasets.SynDataset(
    #     lmdb_dataset.LMDBDataset(join(traindir, 'SID_Sony_Raw.db')),
    #     noise_maker=noise_model, num_burst=1,
    #     size=opt.max_dataset_size, repeat=repeat)

    ## Noise generated offline    
    camera = cameras[opt.include]
    input_data = lmdb_dataset.LMDBDataset(
        join(traindir, f'SID_Sony_syn_Raw_{camera}.db'),
        size=opt.max_dataset_size, repeat=repeat)


train_dataset =  datasets.ELDTrainDataset(target_dataset=target_data, input_datasets=[input_data])

eval_datasets = [datasets.SIDDataset(evaldir, fns, size=None, augment=False, memorize=False, stage_in=opt.stage_in, stage_out=opt.stage_out, gt_wb=opt.gt_wb, CRF=CRF) for fns in eval_fns_list]


train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=opt.batchSize, shuffle=True,
    num_workers=opt.nThreads, pin_memory=True, worker_init_fn=worker_init_fn)


eval_dataloaders = [torch.utils.data.DataLoader(
    eval_dataset, batch_size=1, shuffle=False,
    num_workers=0, pin_memory=True) for eval_dataset in eval_datasets]


"""Main Loop"""
engine = Engine(opt)

print('[i] using noise model {}'.format(opt.noise))

# if opt.resume:
# engine.eval(eval_dataloaders[0], dataset_name='sid_eval_100', correct=True)
# engine.eval(eval_dataloaders[2], dataset_name='sid_eval_300', correct=True)

engine.model.opt.save_epoch_freq = 100

engine.set_learning_rate(1e-4)
while engine.epoch < 200:
    np.random.seed()
    if engine.epoch == 100:
        engine.set_learning_rate(5e-5)
    if engine.epoch == 180:
        engine.set_learning_rate(1e-5)
    
    engine.train(train_dataloader)
    if engine.epoch % 20 == 0:
        try:
            engine.eval(eval_dataloaders[0], dataset_name='sid_eval_100', correct=True)
            engine.eval(eval_dataloaders[2], dataset_name='sid_eval_300', correct=True)
        except:
            pass
