import csv
import sys
import pdb
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timeit import default_timer as timer
from mapnet.model_utils import (
    localize, 
    rotation_resample, 
    compute_spatial_locs,
    project_to_ground_plane,
)
from mapnet.utils import convert_map2world, convert_world2map,save_trajectory
from mapnet.model_aggregate import GRUAggregate, ConvGRUAggregate
from einops import rearrange, reduce, asnumpy, parse_shape

# torch.set_printoptions(precision=4,threshold=10000,linewidth=200)
'''
Maintainance History:
Use pixel_wise class image form yolo as input
----------------------------
ver 1.0 - Jun 6th 2021
    - 
'''
class MapNet_yolo(nn.Module):
    def __init__(self, config, cnn=None):
        super().__init__()

        # map_shape: (F, height, width)
        self._map_shape       = config['map_shape'] #32x101x101
        self._out_shape       = config['map_shape'] #32x101x101
        self.angles           = config['angles']    
        self.nangles          = self.angles.shape[0] #12
        self._local_map_shape = config['local_map_shape'] #32x51x51
        self.map_scale        = config['map_scale'] #0.1
        self.n_cls            = config['num_cls'] #kinds of env default=11
        self.gru_scalar       = config['gru_scalar']
        self.gru  = config['gru'] 
        # Aggregator
        if self.gru == 'conv':
            self.aggregate        = ConvGRUAggregate(self._map_shape[0],norm_h=False)
            print("----use Convoluational GRU----")
        else:
            self.aggregate        = GRUAggregate(self._map_shape[0])
            print("----use Pytorch GRU----")

        self.zero_angle_idx   = None
        for i in range(self.angles.shape[0]):
            if self.angles[i].item() == 0:
                self.zero_angle_idx = i
                break

        self.fixed = False
        self.maps = None
        self.main = cnn
        self.use_gt_pose = config['use_gt_pose']
        self.train()

    def forward(self, inputs, debug=False, maps=None):
        """
        inputs - dictionary with the following keys
            image_cls   - (L, bs,  H, W) or (L, bs, cls, H, W)
            depth - (L, bs, 1, H, W) --- only for 3D settings
            imu   - ( , bs, 3) -- linear acceleration(x,y) and angular velocity(v)
        """
        x_gp_all = inputs['image_cls'] #(L,bs,cls,local_map_size, local_map_size)
        L, bs, n_cls = inputs['image_cls'].shape[:3]
        device = inputs['image_cls'].device
        map_size = self._map_shape[1]
        local_map_size = self._local_map_shape[1]

        # Compute image features        
        if x_gp_all.dim()==4: x_gp_all = x_gp_all[:,:,None] #(L,bs,1,local_map_size, local_map_size)
        x_gp_rot = rotation_resample(x_gp_all[0], self.angles)
        # Initializations   
        if not maps:
            maps = inputs['maps'][0]
        poses = torch.zeros(bs, self.nangles, map_size, map_size, device=device) # (bs, nangles, mapsize,mapsize)
        poses[:, self.zero_angle_idx, (map_size-1)//2, (map_size-1)//2] = 1.0 #if nangles=12, then idx=6.
        maps_all = []
        poses_all = []
        maps_cls_all = []
        # Create groundtruth pose map
        if self.use_gt_pose == 'True':
            try:
                _gt_poses = rearrange(inputs['gt_poses'],'t b c -> (t b) c ') #(L-1,bs,3) -> ((L-1)*bs,3)
                gt_poses = rearrange(torch.zeros(L-1, bs, self.nangles, map_size, map_size, device=device), 't b c h w -> (t b) c h w')  #((L-1)*bs,nangles, mapsize,mapsize)
                gt_poses[range(int((L-1)*bs)) ,_gt_poses[:, 2], _gt_poses[:, 0], _gt_poses[:, 1]] = 1.0
                gt_poses = rearrange(gt_poses,'(t b) c h w -> t b c h w', t = L-1)
            except:
                print('Get acutual pose Error~')
                pass

        for l in range(L-1):
            # Register current observation at the estimated pose
            maps = self._write_memory(x_gp_rot, maps, poses)

            # Localize next observation
            x_gp = x_gp_all[l+1]
            x_gp_rot = rotation_resample(x_gp, self.angles)
            pred_poses = self._localize(x_gp_rot[:,:,1:,...], F.softmax(maps*self.gru_scalar,dim=1)[:,1:,...]) # localize the observation
            maps_all.append(maps)
            poses_all.append(pred_poses)
            #Use ground truth pose to update the map instead of prediction when self.use_gt_pose is True
            if self.use_gt_pose == 'True':
                try:
                    poses = gt_poses[l]
                except:
                    print('Poses Error~')
                    pass

            else:
                poses = pred_poses

        maps_all = torch.stack(maps_all, dim=0) # (L-1, bs, f, map_size, map_size)
        poses_all = torch.stack(poses_all, dim=0) # (L-1, bs, nangles, map_size, map_size)
        maps_cls_all =  None# (L-1, bs, c, real_size, real_size)
        self._save_trajecotry(poses_all,inputs['pose_changes'])
        outputs = {'x_gp': x_gp_all, 'poses': poses_all, 'maps_pred': maps_all, 'map_cls':maps_cls_all}
        return outputs, maps

    def get_feats(self, inputs):
        x = self.main(inputs)
        return x

    def encode_and_project(self, inputs):
        L, bs = inputs['image_cls'].shape[:2] 
        x = rearrange(inputs['image_cls'], 't b h w -> (t b) h w')
        # x = self.get_feats(rgb) # (L*bs, F, H/K, W/K) K=4 in this cnn
        if self.top_down_inputs:
            x_gp = rearrange(x, '(t b) e h w -> t b e h w', t=L)
            outputs = {'x_gp': x_gp}
        else:
            depth = rearrange(inputs['depth'], 't b c h w -> (t b) c h w')
            spatial_locs, valid_inputs = self._compute_spatial_locs(depth) # (L*bs, 2, H, W)
            x_gp = self._project_to_ground_plane(x[:,None], spatial_locs, valid_inputs) # (L*bs, F, s, s)
            outputs = {
                'x_gp': rearrange(x_gp, '(t b) e h w -> t b e h w', t=L),
                'spatial_locs': rearrange(spatial_locs, '(t b) c h w -> t b c h w', t=L),
                'valid_inputs': rearrange(valid_inputs, '(t b) c h w -> t b c h w', t=L),
            }

        return outputs

    def _project_to_ground_plane(self, img_feats, spatial_locs, valid_inputs, eps=-1e16):
        """
        Inputs:
            img_feats       - (bs, f, H/K, W/K)
            spatial_locs    - (bs, 2, H, W)
                              for each pixel in each batch, the (x, y) ground-plane locations are given.
            valid_inputs    - (bs, 1, H, W) ByteTensor
            eps             - fill_value
        Outputs:
            proj_feats      - (bs, f, s, s)
        """
        proj_feats = project_to_ground_plane(
                         img_feats,
                         spatial_locs,
                         valid_inputs,
                         self._local_map_shape[1:],
                         self.K,
                         eps=eps
                     )
        return proj_feats

    def _compute_spatial_locs(self, depth_inputs):
        """
        Inputs:
            depth_inputs - (bs, 3, imh, imw) depth values per pixel in meters. 
        Outputs:
            spatial_locs - (bs, 2, imh, imw) x,y locations of projection per pixel
            valid_inputs - (bs, 1, imh, imw) ByteTensor (all locations where depth measurements are available)
        """
        local_scale = self.map_scale
        local_shape = self._local_map_shape[1:]
        spatial_locs, valid_inputs = compute_spatial_locs(depth_inputs, local_shape, local_scale)

        return spatial_locs, valid_inputs

    def _write_memory(self, o, m, p):
        """
        Inputs:
            o  - (bs, nangles, f, s, s) Tensor of ground plane projection of current image
            m  - (bs, f, H, W) Tensor of overall map
            p  - (bs, nangles, h, w) probabilities over map(orientations, y, x)
        Outputs:
            m  - (bs, f, H, W) Tensor of overall map after update
        """
        # assume s is odd
        bs, nangles, h, w  = p.shape
        view_range = (o.shape[-1] - 1) // 2

        p_    = rearrange(p, 'b o h w -> () (b o) h w')
        o_    = rearrange(o, 'b o e h w -> (b o) e h w')
        o_reg = F.conv_transpose2d(p_, o_, groups=bs, padding=view_range) # (1, bs*f, h, w)
        o_reg = rearrange(o_reg, '() (b e) h w -> b e h w', b=bs) #(bs, f, h, w)       
        m     = self._update_map(o_reg, m)
        return m

    def _update_map(self, o_, m):
        """
        Inputs:
            o_  - (bs, f, H, W)
            m   - (bs, f, H, W)
        """
        bs, f, H, W = o_.size()
        if 'conv' in self.gru:
            m     = self.aggregate(o_, m) # (bs,f,h,w)
        else:
            # Update feature map
            o_    = rearrange(o_, 'b e h w -> (b h w) e')
            m     = rearrange(m, 'b e h w -> (b h w) e')
            # LSTM/RNN update phase. merge spatial dimensions into batch dim, to treat them independently.
            m     = self.aggregate(o_, m) # (bs*H*W, f)
            m     = rearrange(m, '(b h w) e -> b e h w', b=bs, h=H, w=W)

        return m

    def _localize(self, x_rot, maps):
        """ 
        Inputs: 
            x_rot  - (bs, nangles, f, s, s) ground projection of image 
            maps   - (bs, f, H, W) full map 
        Outputs:
            poses  - (bs, nangles, H, W) softmax over all poses
        """ 
        poses = localize(x_rot, maps)

        return poses

    def _save_trajecotry(self,outputs,labels):
        '''
        outputs: (L-1, bs, nangles, map_size, map_size) （yaw,y,x)
        lables: (L-1,bs,3) (x,y,yaw）float number
        '''
        self.outputs = outputs
        self.labels = labels

    def save_tra2path(self,path,name):
        save_trajectory(self.outputs,self.labels,path,self.angles,name)
        # try:
            # save_trajectory(self.outputs,self.labels,path,self.angles,name)
        # except:
        #     print('Can`t save trajectory before generation')
