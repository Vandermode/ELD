import argparse
import models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        # experiment specifics
        self.parser.add_argument('--name', type=str, default=None, help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--model', type=str, default='eld_model', help='chooses which model to use.', choices=model_names)
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
        self.parser.add_argument('--resume_epoch', '-re', type=int, default=None, help='checkpoint to use. (default: latest')
        self.parser.add_argument('--seed', type=int, default=2018, help='random seed to use. Default=2018')

        # for setting input
        self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.parser.add_argument('--nThreads', default=8, type=int, help='# threads for loading data')
        self.parser.add_argument('--chop', action='store_true', help='enable forward_chop')

        # for display
        self.parser.add_argument('--no-log', action='store_true', help='disable tf logger?')
        self.parser.add_argument('--no-verbose', action='store_true', help='disable verbose info?')
        self.parser.add_argument('--debug', action='store_true', help='debugging mode')

        self.initialized = True
