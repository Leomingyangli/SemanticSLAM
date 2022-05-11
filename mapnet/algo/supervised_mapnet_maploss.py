import pdb
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from einops import rearrange, reduce, asnumpy, parse_shape
from einops.layers.torch import Rearrange, Reduce
from timeit import default_timer as timer
import sys
sys.path.append("../..")
from mapnet.gaussian3d import Gaussian3d

'''
Maintainance History:
Utils for model
----------------------------
ver 1.0 - Oct 26th 2020
    - Add Gausian
ver 1.1 - Nov 10th 2020
    - Modify MapNet
ver 2.0 - March 16 2021
    - Add gaussian3d
    - using 3d gaussian
'''
class SupervisedMapNet:
    """
    Gaussian: 
            size of kernel = 3
            outchan=nangles
    """
    def __init__(self, config):

        # Models
        self.mapnet        = config['mapnet']

        # Optimization
        self.max_grad_norm = config['max_grad_norm']
        self.optimizer      = config['optimizer']

        # Others
        self.angles        = config['angles']
        self.lr_stepsize   = config['lr_stepsize']
        self.gamma         = config['lr_gamma']
        self.gauss_sigma   = config['gauss_sigma']
        self.scale         = config['scale']
        # Define optimizers
        self.lr_schedule     = optim.lr_scheduler.MultiStepLR(self.optimizer, self.lr_stepsize, self.gamma)

        #define gaussian filter
        self.gaussian_filter = Gaussian3d(size=3, outchan=1, sigma=self.gauss_sigma, normalize=True)

    def update(self, inputs,maps=None,scalar=1,cls_scalar=1, debug_flag=False):

        gt_maps = rearrange(inputs['maps'][1:], 't b o h w -> (t b) o h w') #(LB,cls,h,w)
        outputs, maps = self.mapnet(inputs,maps)

        # =================== Self-localization loss ====================
        p = rearrange(outputs['maps_pred'], 't b o h w -> (t b) o h w')  #p is result after softmax   # (LB, cls, map_size, map_size)
        p = F.softmax(p*scalar,dim=1) #softmax on cls channel.
        # p = p[:,None]   # (LB, 1, cls, map_size, map_size)
        p = (p + 1e-12).log()        

        # gt_maps = self.gaussian_filter(gt_maps)
        if cls_scalar!=1:
            mask = torch.ones_like(p,dtype=torch.float32,device=p.device)#lb,cls,h,w
            mask[:,1:,...] = mask[:,1:,...]*cls_scalar
            loss_pose = F.kl_div(p,gt_maps,reduction='none')*mask #lb,cls,h,w
            loss_pose = torch.mean(torch.sum(loss_pose,dim=(1,2,3)))
        else:
            loss_pose = F.kl_div(p,gt_maps,reduction='batchmean') 
        self.groundtruth = gt_maps # (LB, 3) x,y,yaw are index
        self.prediction = p # (LB, 1, nangles, map_size, map_size)

        loss = loss_pose

        # ================ Update =================
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.mapnet.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.lr_schedule.step()

        losses = {} 
        losses['loss'] = loss.item()

        return losses, maps

    def out_fig(self):
        return self.prediction.squeeze(), self.groundtruth

