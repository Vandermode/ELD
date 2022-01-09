import torch
from torch import nn
import torch.nn.functional as F

import os
import numpy as np
import fnmatch
from collections import OrderedDict

import util.util as util
import util.index as index
import models.networks as networks
from models import arch, losses

from .base_model import BaseModel
from PIL import Image
from os.path import join

import rawpy
import util.process as process


def tensor2im(image_tensor, visualize=False, video=False):    
    image_tensor = image_tensor.detach()

    if visualize:                
        image_tensor = image_tensor[:, 0:3, ...]

    if not video: 
        image_numpy = image_tensor[0].cpu().float().numpy()
        image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
    else:
        image_numpy = image_tensor.cpu().float().numpy()
        image_numpy = (np.transpose(image_numpy, (0, 2, 3, 1))) * 255.0

    image_numpy = np.clip(image_numpy, 0, 255)

    return image_numpy


def postprocess_bayer(rawpath, img4c):
    img4c = img4c.detach()
    img4c = img4c[0].cpu().float().numpy()
    img4c = np.clip(img4c, 0, 1)

    #unpack 4 channels to Bayer image
    raw = rawpy.imread(rawpath)
    raw_pattern = raw.raw_pattern
    R = np.where(raw_pattern==0)
    G1 = np.where(raw_pattern==1)
    G2 = np.where(raw_pattern==3)
    B = np.where(raw_pattern==2)
    
    black_level = np.array(raw.black_level_per_channel)[:,None,None]

    white_point = 16383

    img4c = img4c * (white_point - black_level) + black_level
    
    img_shape = raw.raw_image_visible.shape
    H = img_shape[0]
    W = img_shape[1]

    raw.raw_image_visible[R[0][0]:H:2, R[1][0]:W:2] = img4c[0, :,:]
    raw.raw_image_visible[G1[0][0]:H:2,G1[1][0]:W:2] = img4c[1, :,:]
    raw.raw_image_visible[B[0][0]:H:2,B[1][0]:W:2] = img4c[2, :,:]
    raw.raw_image_visible[G2[0][0]:H:2,G2[1][0]:W:2] = img4c[3, :,:]
    
    # out = raw.postprocess(use_camera_wb=False, user_wb=[1,1,1,1], half_size=True, no_auto_bright=True, output_bps=8, bright=1, user_black=None, user_sat=None)
    # out = raw.postprocess(use_camera_wb=False, user_wb=[1.96875, 1, 1.444, 1], half_size=True, no_auto_bright=True, output_bps=8, bright=1, user_black=None, user_sat=None)    
    out = raw.postprocess(use_camera_wb=True, half_size=True, no_auto_bright=True, output_bps=8, bright=1, user_black=None, user_sat=None)
    return out


def postprocess_bayer_v2(rawpath, img4c):    
    with rawpy.imread(rawpath) as raw:
        out_srgb = process.raw2rgb_postprocess(img4c.detach(), raw)        
    
    return out_srgb


def postprocess_xtrans(rawpath, img9c):
    img9c = img9c.detach()
    img9c = img9c[0].cpu().float().numpy()
    img9c = np.clip(img9c, 0, 1)

    #unpack 9 channels to xtrans image
    raw = rawpy.imread(rawpath)
    img_shape = raw.raw_image_visible.shape
    H = (img_shape[0] // 6) * 6
    W = (img_shape[1] // 6) * 6
    
    black_level = 1024
    white_point = 16383

    img9c = img9c * (white_point - black_level) + black_level

    # 0 R
    raw.raw_image_visible[0:H:6, 0:W:6] = img9c[0, 0::2, 0::2]
    raw.raw_image_visible[0:H:6, 4:W:6] = img9c[0, 0::2, 1::2]
    raw.raw_image_visible[3:H:6, 1:W:6] = img9c[0, 1::2, 0::2]
    raw.raw_image_visible[3:H:6, 3:W:6] = img9c[0, 1::2, 1::2]

    # 1 G
    raw.raw_image_visible[0:H:6, 2:W:6] = img9c[1, 0::2, 0::2]
    raw.raw_image_visible[0:H:6, 5:W:6] = img9c[1, 0::2, 1::2]
    raw.raw_image_visible[3:H:6, 2:W:6] = img9c[1, 1::2, 0::2]
    raw.raw_image_visible[3:H:6, 5:W:6] = img9c[1, 1::2, 1::2]

    # 1 B
    raw.raw_image_visible[0:H:6, 1:W:6] = img9c[2, 0::2, 0::2]
    raw.raw_image_visible[0:H:6, 3:W:6] = img9c[2, 0::2, 1::2]
    raw.raw_image_visible[3:H:6, 0:W:6] = img9c[2, 1::2, 0::2]
    raw.raw_image_visible[3:H:6, 4:W:6] = img9c[2, 1::2, 1::2]

    # 4 R
    raw.raw_image_visible[1:H:6, 2:W:6] = img9c[3, 0::2, 0::2]
    raw.raw_image_visible[2:H:6, 5:W:6] = img9c[3, 0::2, 1::2] 
    raw.raw_image_visible[5:H:6, 2:W:6] = img9c[3, 1::2, 0::2] 
    raw.raw_image_visible[4:H:6, 5:W:6] = img9c[3, 1::2, 1::2] 

    # 5 B
    raw.raw_image_visible[2:H:6, 2:W:6] = img9c[4, 0::2, 0::2]
    raw.raw_image_visible[1:H:6, 5:W:6] = img9c[4, 0::2, 1::2]
    raw.raw_image_visible[4:H:6, 2:W:6] = img9c[4, 1::2, 0::2]
    raw.raw_image_visible[5:H:6, 5:W:6] = img9c[4, 1::2, 1::2]

    raw.raw_image_visible[1:H:3, 0:W:3] = img9c[5, :, :]
    raw.raw_image_visible[1:H:3, 1:W:3] = img9c[6, :, :]
    raw.raw_image_visible[2:H:3, 0:W:3] = img9c[7, :, :]
    raw.raw_image_visible[2:H:3, 1:W:3] = img9c[8, :, :]
    
    out = raw.postprocess(use_camera_wb=True, half_size=True, no_auto_bright=True, output_bps=8, bright=1, user_black=None, user_sat=None)
    # out = raw.postprocess(use_camera_wb=True, demosaic_algorithm=rawpy.DemosaicAlgorithm.LINEAR, no_auto_bright=True, output_bps=8, bright=1, user_black=None, user_sat=None)
    return out


class IlluminanceCorrect(nn.Module):
    def __init__(self):
        super(IlluminanceCorrect, self).__init__()
    
    # Illuminance Correction
    def forward(self, predict, source):
        if predict.shape[0] != 1:
            output = torch.zeros_like(predict)
            if source.shape[0] != 1:
                for i in range(predict.shape[0]):
                    output[i:i+1, ...] = self.correct(predict[i:i+1, ...], source[i:i+1, ...])               
            else:                                     
                for i in range(predict.shape[0]):
                    output[i:i+1, ...] = self.correct(predict[i:i+1, ...], source)                    
        else:
            output = self.correct(predict, source)
        return output

    def correct(self, predict, source):
        N, C, H, W = predict.shape        
        predict = torch.clamp(predict, 0, 1)
        assert N == 1
        output = torch.zeros_like(predict, device=predict.device)
        pred_c = predict[source != 1]
        source_c = source[source != 1]
        
        num = torch.dot(pred_c, source_c)
        den = torch.dot(pred_c, pred_c)        
        output = num / den * predict
        # print(num / den)

        return output


class ELDModelBase(BaseModel):
    def set_input(self, data, mode='train'):
        target = None
        data_name = None

        mode = mode.lower()
        if mode == 'train':
            input, target = data['input'], data['target']
        elif mode == 'eval':
            input, target, data_name = data['input'], data['target'], data['fn']
        elif mode == 'test':
            input, data_name = data['input'], data['fn']
        else:
            raise NotImplementedError('Mode [%s] is not implemented' % mode)
        
        if len(self.gpu_ids) > 0:  # transfer data into gpu
            input = input.to(device=self.gpu_ids[0])
            if target is not None:
                target = target.to(device=self.gpu_ids[0]) 

        self.input = input
        self.target = target
        self.data_name = data_name

        self.rawpath = data['rawpath'][0] if 'rawpath' in data else None
        self.cfa = data['cfa'][0] if 'cfa' in data else 'bayer'

        # self.issyn = False if 'real' in data else True
        self.aligned = False if 'unaligned' in data else True

            
    def eval(self, data, savedir=None, suffix=None, correct=False, crop=True, frame_id=None):
        # only the 1st input of the whole minibatch would be evaluated
        self._eval()
        self.set_input(data, 'eval')

        # if self.data_name is not None and savedir is not None:
        #     name = os.path.splitext(os.path.basename(self.data_name[0]))[0]
        #     if not os.path.exists(join(savedir, name)):
        #         os.makedirs(join(savedir, name))
            
        #     for fn in os.listdir(join(savedir, name)):                
        #         if fnmatch.fnmatch(fn, '*{}_*'.format(self.opt.name)):
        #             return {}

        with torch.no_grad():
            ### evaluate center region to avoid fixed pattern noise
            cropx = 512; cropy = 512

            if crop:
                self.target = util.crop_center(self.target, cropx, cropy)
                self.input = util.crop_center(self.input, cropx, cropy)

            self.forward()

            if correct:
                self.output = self.corrector(self.output, self.target)
            
            if self.opt.stage_out == 'raw' and self.opt.stage_eval == 'srgb':
                target = postprocess_bayer_v2(self.rawpath, self.target)
                output = postprocess_bayer_v2(self.rawpath, self.output)
                input = postprocess_bayer_v2(self.rawpath, self.input)
            else:
                output = self.output
                target = self.target
                input = self.input

            output = tensor2im(output)
            target = tensor2im(target)   
            input = tensor2im(input)

            if target.shape[0] != output.shape[0]:
                target = np.repeat(target, output.shape[0], axis=0)

            res = index.quality_assess(output, target, data_range=255)
            res_in = index.quality_assess(input, target, data_range=255)  

            if savedir is not None:
                ## raw postprocessing
                if self.rawpath:
                    if self.cfa == 'bayer':
                        output = postprocess_bayer(self.rawpath, self.output)
                        target = postprocess_bayer(self.rawpath, self.target)
                        input = postprocess_bayer(self.rawpath, self.input)

                        # target = tensor2im(postprocess_bayer_v2(self.rawpath, self.target))
                        # output = tensor2im(postprocess_bayer_v2(self.rawpath, self.output))
                        # input = tensor2im(postprocess_bayer_v2(self.rawpath, self.input))

                    elif self.cfa == 'xtrans':
                        output = postprocess_xtrans(self.rawpath, self.output)
                        target = postprocess_xtrans(self.rawpath, self.target)
                        input = postprocess_xtrans(self.rawpath, self.input)
                    else:
                        raise NotImplementedError

                if self.data_name is not None:
                    name = os.path.splitext(os.path.basename(self.data_name[0]))[0]

                    if not os.path.exists(join(savedir, name)):
                        os.makedirs(join(savedir, name))

                    if frame_id is not None:
                        if not os.path.exists(join(savedir, name, self.opt.name)):
                            os.makedirs(join(savedir, name, self.opt.name))

                        if not os.path.exists(join(savedir, name, 'input')):
                            os.makedirs(join(savedir, name, 'input'))                            

                        Image.fromarray(output.astype(np.uint8)).save(join(savedir, name, self.opt.name, '{}_{:.2f}.png'.format(frame_id, res['PSNR'])))

                        if not os.path.exists(join(savedir, name, 'input', '{}_{:.2f}.png'.format(frame_id, res_in['PSNR']))):
                            Image.fromarray(input.astype(np.uint8)).save(join(savedir, name, 'input', '{}_{:.2f}.png'.format(frame_id, res_in['PSNR'])))

                        if not os.path.exists(join(savedir, name, 'label')):
                            os.makedirs(join(savedir, name, 'label'))   

                        if not os.path.exists(join(savedir, name, 'label', '{}.png'.format(frame_id))):
                            Image.fromarray(target.astype(np.uint8)).save(join(savedir, name, 'label', '{}.png'.format(frame_id)))
                    else:
                        if suffix is not None:
                            Image.fromarray(output.astype(np.uint8)).save(join(savedir, name,'{}_{:.1f}_{}.png'.format(self.opt.name, res['PSNR'], suffix)))
                            # Image.fromarray(output.astype(np.uint8)).save(join(savedir, name,'{}_{:.1f}_{}.jpg'.format(self.opt.name, res['PSNR'], suffix)), optimize=True, quality=90)
                            Image.fromarray(input.astype(np.uint8)).save(join(savedir, name, 'm_input_{}.png'.format(suffix)))
                            # Image.fromarray(input.astype(np.uint8)).save(join(savedir, name, 'm_input_{}.jpg'.format(suffix)), optimize=True, quality=90)
                        else:
                            # Image.fromarray(output.astype(np.uint8)).save(join(savedir, name, '{}_{:.1f}.jpg'.format(self.opt.name, res['PSNR'])), optimize=True, quality=90)
                            Image.fromarray(output.astype(np.uint8)).save(join(savedir, name, '{}_{:.2f}.png'.format(self.opt.name, res['PSNR'])))
                            # Image.fromarray(output.astype(np.uint8)).save(join(savedir, name, '{}.png'.format(self.opt.name)))
                            Image.fromarray(input.astype(np.uint8)).save(join(savedir, name, 'm_input_{:.2f}.png'.format(res_in['PSNR'])))
                            # Image.fromarray(input.astype(np.uint8)).save(join(savedir, name, 'm_input.jpg'), optimize=True, quality=90)
                        
                        Image.fromarray(target.astype(np.uint8)).save(join(savedir, name, 't_label.png'))
                        # Image.fromarray(target.astype(np.uint8)).save(join(savedir, name, 't_label.jpg'), optimize=True, quality=90)

            return res

    def test(self, data, savedir=None, video_mode=False):
        # only the 1st input of the whole minibatch would be evaluated
        self._eval()
        self.set_input(data, 'test')

        if self.data_name is not None and savedir is not None:
            name = os.path.splitext(os.path.basename(self.data_name[0]))[0]
            if not video_mode:
                if not os.path.exists(join(savedir, name)):
                    os.makedirs(join(savedir, name))

                # if os.path.exists(join(savedir, name, '{}.png'.format(self.opt.name))):
                for fn in os.listdir(join(savedir, name)):
                    if fnmatch.fnmatch(fn, '*{}_*'.format(self.opt.name)):
                        return
            else:
                if not os.path.exists(join(savedir, self.opt.name)):
                    os.makedirs(join(savedir, self.opt.name))
                
        with torch.no_grad():
            output = self.forward()
            
            # if self.opt.netG == 'fastdvd': # video network  
            #     self.input = self.input[:, 8:12, ...]

            ## raw postprocessing
            if self.rawpath:
                if self.opt.stage_in == 'srgb':
                    output = tensor2im(self.output)
                    input = tensor2im(self.input)
                else:
                    output = postprocess_bayer(self.rawpath, self.output)
                    input = postprocess_bayer(self.rawpath, self.input)                  

                if not video_mode:
                    Image.fromarray(output.astype(np.uint8)).save(join(savedir, name,'{}.jpg'.format(self.opt.name)), optimize=True, quality=90)
                    Image.fromarray(input.astype(np.uint8)).save(join(savedir, name, 'm_input.jpg'), optimize=True, quality=90)
                else:
                    Image.fromarray(output.astype(np.uint8)).save(join(savedir, self.opt.name,'{}.jpg'.format(name)), optimize=True, quality=90)

        return output


class ELDModel(ELDModelBase):
    def __init__(self):
        self.epoch = 0
        self.iterations = 0
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.corrector = IlluminanceCorrect()
        self.CRF = None

    def print_network(self):
        print('--------------------- Model ---------------------')
        networks.print_network(self.netG)

    def _eval(self):
        self.netG.eval()

    def _train(self):
        self.netG.train()

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        # init CRF function
        if opt.crf:
            self.CRF = process.load_CRF()

        if opt.stage_in == 'raw':
            in_channels = opt.channels
        elif opt.stage_in == 'srgb':
            in_channels = 3
        else:
            raise NotImplementedError('Invalid Input Stage: {}'.format(opt.stage_in))
        
        if opt.stage_out == 'raw':
            out_channels = opt.channels
        elif opt.stage_out == 'srgb':
            out_channels = 3
        else:
            raise NotImplementedError('Invalid Output Stage: {}'.format(opt.stage_in))
    
        self.netG = arch.__dict__[self.opt.netG](in_channels, out_channels).to(self.device)
        
        # networks.init_weights(self.netG, init_type=opt.init_type)  # using default initialization as EDSR

        if self.isTrain:
            # define loss functions
            self.loss_dic = losses.init_loss(opt)

            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), 
                lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.wd)

            self._init_optimizer([self.optimizer_G])

        if opt.resume:
            self.load(self, opt.resume_epoch)
        
        if opt.no_verbose is False:
            self.print_network()

    def backward_G(self):        
        self.loss_G = 0
        self.loss_pixel = None
        
        self.loss_pixel = self.loss_dic['pixel'].get_loss(
            self.output, self.target)

        self.loss_G += self.loss_pixel
        
        self.loss_G.backward()

    def forward(self):        
        input_i = self.input

        if self.opt.chop:            
            output = self.forward_chop(input_i)
        else:
            output = self.netG(input_i)
        
        self.output = output        

        return output

    def forward_chop(self, x, base=16):        
        b, c, h, w = x.size()
        h_half, w_half = h // 2, w // 2
        
        shave_h = np.ceil(h_half / base) * base - h_half
        shave_w = np.ceil(w_half / base) * base - w_half

        shave_h = shave_h if shave_h >= 10 else shave_h + base
        shave_w = shave_w if shave_w >= 10 else shave_w + base

        h_size, w_size = int(h_half + shave_h), int(w_half + shave_w)        
        
        inputs = [
            x[:, :, 0:h_size, 0:w_size],
            x[:, :, 0:h_size, (w - w_size):w],
            x[:, :, (h - h_size):h, 0:w_size],
            x[:, :, (h - h_size):h, (w - w_size):w]
        ]

        outputs = [self.netG(input_i) for input_i in inputs]

        c = outputs[0].shape[1]        
        output = x.new(b, c, h, w)

        output[:, :, 0:h_half, 0:w_half] \
            = outputs[0][:, :, 0:h_half, 0:w_half]
        output[:, :, 0:h_half, w_half:w] \
            = outputs[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
        output[:, :, h_half:h, 0:w_half] \
            = outputs[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
        output[:, :, h_half:h, w_half:w] \
            = outputs[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

        return output

    def optimize_parameters(self):
        self._train()
        self.forward()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_errors(self):
        ret_errors = OrderedDict()
        if self.loss_pixel is not None:
            ret_errors['Pixel'] = self.loss_pixel.item()

        return ret_errors

    def get_current_visuals(self):
        ret_visuals = OrderedDict()
        ret_visuals['input'] = tensor2im(self.input, visualize=True).astype(np.uint8)
        ret_visuals['output'] = tensor2im(self.output, visualize=True).astype(np.uint8)
        ret_visuals['target'] = tensor2im(self.target, visualize=True).astype(np.uint8)

        return ret_visuals       

    @staticmethod
    def load(model, resume_epoch=None):
        model_path = model.opt.model_path
        state_dict = None

        if model_path is None:
            model_path = util.get_model_list(model.save_dir, 'model', epoch=resume_epoch)
            state_dict = torch.load(model_path)
            model.epoch = state_dict['epoch']
            model.iterations = state_dict['iterations']            
            model.netG.load_state_dict(state_dict['netG'])
            if model.isTrain:
                model.optimizer_G.load_state_dict(state_dict['opt_g'])
        else:
            state_dict = torch.load(model_path)
            model.netG.load_state_dict(state_dict['netG'])
            model.epoch = state_dict['epoch']
            model.iterations = state_dict['iterations']
            if model.isTrain:
                model.optimizer_G.load_state_dict(state_dict['opt_g'])
            
        print('Resume from epoch %d, iteration %d' % (model.epoch, model.iterations))
        return state_dict

    def state_dict(self):
        state_dict = {
            'netG': self.netG.state_dict(),
            'opt_g': self.optimizer_G.state_dict(), 
            'epoch': self.epoch, 'iterations': self.iterations
        }

        return state_dict
