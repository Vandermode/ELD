# Visualize your datasets (for debugging)
from os.path import join
from options.eld.train_options import TrainOptions
from dataset.lmdb_dataset import LMDBDataset
import torch
import dataset.sid_dataset as datasets
import dataset
import cv2
import numpy as np
import noise


opt = TrainOptions().parse()

traindir = './data/Train'

expo_ratio = [100, 250, 300]
read_expo_ratio = lambda x: float(x.split('_')[-1][:-5])

train_fns = dataset.read_paired_fns('./dataset/Sony_train.txt')
eval_fns = dataset.read_paired_fns('./dataset/Sony_val.txt')
test_fns = dataset.read_paired_fns('./dataset/Sony_test.txt')

eval_fns_list = [[fn for fn in eval_fns if min(int(read_expo_ratio(fn[1])/read_expo_ratio(fn[0])), 300)==ratio] for ratio in expo_ratio]
eval_fns_list = [lst[-5:] for lst in eval_fns_list]
test_fns_list = [[fn for fn in test_fns if min(int(read_expo_ratio(fn[1])/read_expo_ratio(fn[0])), 300)==ratio and int(fn[0][:5]) not in [10192, 10199, 10203]] for ratio in expo_ratio]
test_fns_list = [lst[-10:] for lst in test_fns_list]
eval_fns_list = [lst_1 + lst_2 for lst_1, lst_2 in zip(eval_fns_list, test_fns_list)]

noise_model = noise.NoiseModel(model=opt.noise, include=opt.include, exclude=None)

repeat = 1 if opt.max_dataset_size is None else 1288 // opt.max_dataset_size


# target_data = LMDBDataset(join(traindir, 'SID_Sony_SRGB_CRF.db'))
# input_data = LMDBDataset(join(traindir, 'SID_Sony_Raw.db'))

# target_data = LMDBDataset(join(traindir, 'SID_Sony_target_Raw.db'))
# input_data = LMDBDataset(join(traindir, 'SID_Sony_input_Raw.db'))

target_data = LMDBDataset(
    join(traindir, 'SID_Sony_Raw.db'), 
    size=opt.max_dataset_size, repeat=repeat)

# input_data = datasets.SynDataset(
#     LMDBDataset(join(traindir, 'SID_Sony_Raw.db')),
#     noise_maker=noise_model,
#     size=opt.max_dataset_size, repeat=repeat
# )

# input_data = datasets.ISPDataset(
#     LMDBDataset(join(traindir, 'SID_Sony_Raw.db')),
#     noise_maker=None,    
# )

camera = 'NikonD850'
input_data = LMDBDataset(
    join(traindir, f'SID_Sony_syn_Raw_{camera}.db'),
    size=opt.max_dataset_size, repeat=repeat)


train_dataset = datasets.ELDTrainDataset(target_dataset=target_data, input_datasets=[input_data], size=10, augment=False)


train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=opt.batchSize, shuffle=True,
    num_workers=0, pin_memory=True, worker_init_fn=datasets.worker_init_fn)


"""Main Loop"""
from models.ELD_model import tensor2im

for dataset in train_dataloader:
    np.random.seed()

    input, target = dataset['input'], dataset['target']

    target_image = tensor2im(target, visualize=True)
    input_image = tensor2im(input, visualize=True)

    display = np.concatenate([input_image[:,:,::-1], target_image[:,:,::-1]], axis=1).astype(np.uint8)
    
    cv2.imshow('display', display)

    cv2.waitKey()
