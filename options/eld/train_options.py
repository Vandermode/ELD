from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--batchSize', '-b', type=int, default=1, help='input batch size')
        self.parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate for adam')  
        self.parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')
        self.parser.add_argument('--wd', type=float, default=0, help='weight decay for adam')          
        self.parser.add_argument('--max_dataset_size', type=int, default=None, help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        
        self.parser.add_argument('--loss', type=str, default='l1', help='pixel loss type')
        self.parser.add_argument('--noise', type=str, default='g', help='noise model to use')
        self.parser.add_argument('--exclude', type=int, default=None, help='camera exclude id')

        self.isTrain = True
