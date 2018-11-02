import torch
from torch import nn
from torch.optim import lr_scheduler
from torchvision.utils import make_grid

from skimage import io
import numpy as np
import itertools
from collections import OrderedDict
import os
import re

from utils.network import Basic_D, Unet_Dilate, init_weights
from utils.loss import ColorConsistencyLoss, id_loss
# ========== Model config ==========
# ngf = 64
ndf = 64
# nc_G_inp = 9 
nc_G_out = 4 
nc_D_inp = 6 
nc_D_out = 1 
use_instancenorm = True # False: batchnorm
# gamma_i = 0.1
# use_mixup = True

#========== Training config ==========
mixup_alpha = 0.1
# lrD = 2e-4
# lrG = 2e-4
def mkdir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_xi_yi_yj(batch):
    return batch[:,:3,:,:], batch[:,3:6,:,:], batch[:,6:,:,:]

def get_alpha_xij(out_tensor):
    return out_tensor[:, 0:1, :, :], out_tensor[:, 1:, :, :]

def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(current_step):
            lr_l = 1.0 - max(0, current_step + 1 + opt.step - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    # elif opt.lr_policy == 'plateau':
    #     scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=opt.lr*0.0001)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

class CAGAN_Trainer(object):
    def __init__(self, opt, is_train=True, save_dir='train_logs/test/' ,device='cuda:0', is_cyclic=True, use_lsgan=False, img_size=(256, 192)):
        self.is_train = is_train
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.device = device
        self.img_sizes = []
        for i in [4, 2, 1]:
            self.img_sizes.append(tuple([x//i for x in img_size]))
        # model
        self.nets = [Unet_Dilate(opt.nc_G_inp, nc_G_out, opt.ngf, up_type=opt.up_type)]
        if self.is_train:
            for i in range(3):
                self.nets.append(Basic_D(nc_D_inp, ndf, use_sigmoid = not use_lsgan))
            map(init_weights, self.nets)
            self.use_mixup = opt.use_mixup
            self.is_cyclic = is_cyclic
            if self.use_mixup:
                self.beta_dist = torch.distributions.beta.Beta(torch.tensor([mixup_alpha]), torch.tensor([mixup_alpha]))
            # optimizer
            self.opti_G = torch.optim.Adam(self.nets[0].parameters(), lr=opt.lr, betas=(0.5, 0.999))
            if self.is_train:
                D_parameters = itertools.chain(*[netD.parameters() for netD in self.nets[1:]])
                self.opti_D = torch.optim.Adam(D_parameters, lr=opt.lr, betas=(0.5, 0.999))
            
            optimizers = []
            optimizers.append(self.opti_G)
            optimizers.append(self.opti_D)
            self.schedulers = [get_scheduler(optimizer, opt) for optimizer in optimizers]
            # loss function
            if use_lsgan:
                self.loss_fn = nn.MSELoss()
            else:
                self.loss_fn = nn.BCELoss()
            self.loss_fn_L1 = nn.L1Loss()
            self.loss_fn_cc = ColorConsistencyLoss()
            self.gamma_i = opt.gamma_i
            self.loss_names = ['loss_D0', 'loss_D1', 'loss_D2', 'loss_D_sum',
                'loss_G0', 'loss_G1', 'loss_G2', 'loss_G_sum', 
                'loss_color0', 'loss_color1', 'loss_color_sum', 
                'loss_cyc0', 'loss_cyc1', 'loss_cyc2', 'loss_cyc_sum', 
                'loss_id_sum', 'loss_identity', 'loss_updateG_sum']
        else:
            self.nets[0].eval()
        for net in self.nets:
            net.to(self.device)

    def set_input(self, real_input):
        # self.x_i, self.y_i, self.y_j = get_xi_yi_yj(real_input.to(self.device))
        self.real = real_input.to(self.device)

    def resize_tensor(self, x, size=(256, 192)):
        return nn.Upsample(size, mode='bilinear', align_corners=False)(x)

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()

    def netG_forward(self):
        x_i, y_i, y_j = get_xi_yi_yj(self.real)
        fake_output0, fake_output1, fake_output2 = self.nets[0](torch.cat([x_i, y_i, y_j],1))
        xi_s64 = nn.AdaptiveAvgPool2d([x for x in fake_output0.shape[2:]])(x_i)
        xi_s128 = nn.AdaptiveAvgPool2d([x for x in fake_output1.shape[2:]])(x_i)
        
        alpha0, x_i_j0 = get_alpha_xij(fake_output0)
        alpha1, x_i_j1 = get_alpha_xij(fake_output1)  
        alpha2, x_i_j2 = get_alpha_xij(fake_output2) 
        fake_output0 = alpha0*x_i_j0 + (1-alpha0)*xi_s64 
        fake_output0_resize256 = self.resize_tensor(fake_output0)
        fake_output1 = alpha1*x_i_j1 + (1-alpha1)*xi_s128 
        fake_output1_resize256 = self.resize_tensor(fake_output1)
        fake_output2 = alpha2*x_i_j2 + (1-alpha2)*x_i
        # size64: Pass x_i_j0 instead of fake_output0 for masking
        rec_input_size64 = self.get_rec_out(fake_output0_resize256, x_i_j0, y_i, y_j, 0) 
        rec_input_size128 = self.get_rec_out(fake_output1_resize256, fake_output1, y_i, y_j, 1)
        rec_input = self.get_rec_out(fake_output2, fake_output2, y_i, y_j, 2)
        
        identity_input = self.get_rec_out(x_i, x_i, y_i, y_i, 2)
        self.output_dict = {
            'fake_outputs':[fake_output0, fake_output1, fake_output2],
            'rec_inputs': [rec_input_size64, rec_input_size128, rec_input, identity_input],
            'alpha_list':[alpha0, alpha1, alpha2]
        }

    def get_rec_out(self, xi_256, xi_ori, yi, yj, out_idx):
        input_G2 = torch.cat([xi_256, yj, yi], 1) # swap y_i and y_j
        rec_out = self.nets[0](input_G2)[out_idx]
        rec_alpha, rec_xij = get_alpha_xij(rec_out)
        rec_out = rec_alpha*rec_xij + (1-rec_alpha)*xi_ori
        return rec_out

    def loss_D_basic(self, netD, real, fake):
        xi, yi, yj = get_xi_yi_yj(real)
        lam = self.beta_dist.sample().to(self.device)

        if self.use_mixup:
            mixup_x = lam*xi + (1-lam)*fake.detach()
            mixup_y = lam*yi + (1-lam)*yj
            output_real = netD(torch.cat([mixup_x, mixup_y], 1)) # positive sample + negative sample
            output_fake2 = netD(torch.cat([xi, yj], 1)) # negative sample 2 
        else:
            output_real = netD(torch.cat([xi, yi], 1)) # positive sample
            output_fake1 = netD(torch.cat([fake.detach(), yj], 1)) # negative sample
            output_fake2 = netD(torch.cat([xi, yj], 1)) # negative sample 2 

            loss_D_fake1 = self.loss_fn(output_fake1, torch.zeros_like(output_fake1, device=self.device))
        
        loss_D_real = self.loss_fn(output_real, lam*torch.ones_like(output_real, device=self.device)) 
        loss_D_fake2 = self.loss_fn(output_fake2, torch.zeros_like(output_fake2, device=self.device))   
        
        if self.use_mixup:
            loss_D = loss_D_real+ loss_D_fake2
        else:
            loss_D = loss_D_real+ (loss_D_fake1+loss_D_fake2)
        
        return loss_D
    
    def loss_G_basic(self, netD, fake, yj):
        output_fake = netD(torch.cat([fake, yj], 1))
        loss_G = self.loss_fn(output_fake, torch.ones_like(output_fake, device=self.device))
        return loss_G

    def optimize_parameters(self):
        self.netG_forward()
        self.loss_D_sum = 0
        for i in range(3):
            if i !=2:
                real_resize = self.resize_tensor(self.real, self.img_sizes[i])
            else:
                real_resize = self.real
            loss_D_i = self.loss_D_basic(self.nets[i+1], real_resize, self.output_dict['fake_outputs'][i])
            setattr(self, 'loss_D%d'%i, float(loss_D_i))
            self.loss_D_sum += loss_D_i
        self.opti_D.zero_grad()
        self.loss_D_sum.backward()
        self.opti_D.step()

        self.loss_G_sum = 0
        self.loss_cyc_sum = 0
        self.loss_color_sum = 0
        for i in range(3):
            if i !=2:
                real_resize = self.resize_tensor(self.real, self.img_sizes[i])
                loss_color_i = self.loss_fn_cc(self.output_dict['fake_outputs'][i], self.output_dict['fake_outputs'][i+1])
                setattr(self, 'loss_color%d'%i, float(loss_color_i))
                self.loss_color_sum += loss_color_i
            else:
                real_resize = self.real
            xi, yi, yj = get_xi_yi_yj(real_resize)
            loss_G_i = self.loss_G_basic(self.nets[i+1], self.output_dict['fake_outputs'][i], yj)
            setattr(self, 'loss_G%d'%i, float(loss_G_i))
            loss_cyc_i = self.loss_fn_L1(self.output_dict['rec_inputs'][i], xi)
            setattr(self, 'loss_cyc%d'%i, float(loss_cyc_i))
            self.loss_G_sum += loss_G_i
            self.loss_cyc_sum += loss_cyc_i
        # identity loss
        self.loss_identity = self.loss_fn_L1(self.output_dict['rec_inputs'][-1], self.real[:,0:3,:,:])
        # id loss
        self.loss_id_sum = id_loss(*self.output_dict['alpha_list'])
        self.loss_updateG_sum = self.loss_G_sum + self.loss_cyc_sum + self.gamma_i*self.loss_id_sum + self.loss_color_sum + self.loss_identity
        self.opti_G.zero_grad()
        self.loss_updateG_sum.backward()
        self.opti_G.step()

    def get_current_losses(self, *loss_name):
        if len(loss_name) == 0:
            loss_name = self.loss_names
        losses = OrderedDict()
        for name in loss_name:
            if isinstance(name, str):
                losses[name] = float(getattr(self, name))
        return losses

    def get_current_visuals(self):
        vis_batches = list(get_xi_yi_yj(self.real.cpu()))
        for key in self.output_dict.keys():
            for i in range(3):
                vis_batch = self.output_dict[key][i].data.cpu()
                if i != 2:
                    vis_batch = self.resize_tensor(vis_batch)
                if vis_batch.shape[1] == 1:
                    vis_batch = vis_batch.repeat(1,3,1,1)
                vis_batches.append(vis_batch)
        vis_batches = torch.cat(vis_batches, 0)
        return ((vis_batches+1)/2*255).clamp(0, 255).to(dtype=torch.uint8)

    def save_current_visuals(self, step, batchsize=8):
        imt = make_grid(self.get_current_visuals(), nrow=batchsize)
        img = imt.numpy().transpose(1,2,0)
        save_dir = os.path.join(self.save_dir, 'visuals')
        os.makedirs(save_dir, exist_ok=True)
        io.imsave(os.path.join(save_dir, '%d.jpg'%step), img)
    
    def save_networks(self, step):
        save_dir = os.path.join(self.save_dir, 'model_weight')
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.nets[0].state_dict(), os.path.join(save_dir, 'netG_%d.pth'%step))
        if self.is_train:
            for i in range(3):
                torch.save(self.nets[i+1].state_dict(), os.path.join(save_dir, 'netD%d_latest.pth'%i))
    
    def load_networks(self, step=None, load_dir=None, load_netD=False):
        if load_dir == None:
            load_dir = os.path.join(self.save_dir, 'model_weight')
        if step == None:
            names = os.listdir(load_dir)
            names.sort()
            name = names[-1]
            if re.match(r'netG\_(\d+)\.pth$', name):
                step = re.match(r'.+\_(\d+)\.pth$', name).group(1)
            else:
                raise Exception('%s is not a correct model path for match'%name[-1])
        netD_path = os.path.join(load_dir, 'netG_%s.pth'%step)
        params = torch.load(netD_path, map_location=str(self.device))
        self.nets[0].load_state_dict(params)
        if load_netD:
            for i in range(3):
                params = torch.load(os.path.join(load_dir, 'netD%d_latest.pth'%i), map_location=self.device)
                self.nets[i+1].load_state_dict(params)
        print("Load success from %s"%netD_path)
