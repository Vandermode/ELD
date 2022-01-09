import os
import torch
import util.util as util


class BaseModel():
    def name(self):
        return self.__class__.__name__.lower()

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self._count = 0

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    # used in test time, no backprop
    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    # update learning rate (called once every epoch)
    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def print_optimizer_param(self):
        # for optimizer in self.optimizers:
        #     print(optimizer)
        print(self.optimizers[-1])

    def save(self, label=None):
        epoch = self.epoch
        iterations = self.iterations

        if label is None:
            # model_name = os.path.join(self.save_dir, self.name() + '_%03d_%08d.pt' % ((epoch), (iterations)))
            model_name = os.path.join(self.save_dir, 'model' + '_%03d_%08d.pt' % ((epoch), (iterations)))
        else:
            # model_name = os.path.join(self.save_dir, self.name() + '_' + label + '.pt')
            model_name = os.path.join(self.save_dir, 'model' + '_' + label + '.pt')
            
        torch.save(self.state_dict(), model_name)

    def _init_optimizer(self, optimizers):
        self.optimizers = optimizers
        # reinitilize schedulers
        self.schedulers = []
        for optimizer in self.optimizers:
            util.set_opt_param(optimizer, 'initial_lr', self.opt.lr)
            util.set_opt_param(optimizer, 'weight_decay', self.opt.wd)
