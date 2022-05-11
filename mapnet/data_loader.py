import os
import torch
import numpy as np
import sys
sys.path.append("../")
from einops import rearrange, reduce, asnumpy, parse_shape
from einops.layers.torch import Rearrange, Reduce
from mapnet.utils import *
import scipy.integrate as it

class DataLoaderAVD:
    def __init__(
        self,
        data_path,
        batch_size,
        num_steps,
        split, # train / val
        device,
        seed,
        env_name,
        max_steps=None,
        randomize_start_time=False,
        n_cls=11
    ):
        self.data_path = data_path
        self.num_steps = num_steps
        self.split = split
        self.device = device
        self.rng = np.random.RandomState(seed)
        # time.sleep(3) #delay 3 sconds for multi-process
        self.data_file = np.load(self.data_path,allow_pickle=True)
        # self.obs_keys = list(self.data_file[self.split].keys()) # [train所有文件的key] [im,delta,pose]
        self.obs_keys = list(self.data_file.files)
        # self.nsplit = self.data_file[self.split][self.obs_keys[0]].shape[1] # train:im: ndarray.shape[1]      -> number of scenes
        self.nsplit = self.data_file[self.obs_keys[0]].shape[1]
        # print('num_split_file:',self.nsplit)
        '''new function'''
        self.index = list(range(self.nsplit))
        self.n_cls = n_cls
        if max_steps is None:
            # self.max_steps = self.data_file[self.split][self.obs_keys[0]].shape[0] # train:key1: ndarray.shape[0] -> max timesteps
            self.max_steps = self.data_file[self.obs_keys[0]].shape[0]
        else:
            self.max_steps = max_steps
        max_batch = self.data_file[self.obs_keys[0]].shape[1]
        self.batch_size = batch_size if batch_size < max_batch else max_batch
        self.randomize_start_time = randomize_start_time
        self.observation_space = {                      # train:[key1,...]: ndarray.shape[2:] -> observation_space ->{文件名：shape}
            # key: self.data_file[self.split][key].shape[2:] 
            key: self.data_file[key].shape[2:] 
            for key in self.obs_keys
        }
        self.data_idx = 0
        self.env_name = env_name

    def sample_maploss(self):
        """
        Samples a random set of batch_size episodes. 
        Input: 
            image_cls:      (L,B,cls,local_mapsize,local_mapsize)
            delta:          (L,B,3)
            maps:           (L,B,cls,mapsize,mapsize) observabale maps to current step
            map_labl:       (B,mapsize,mapsize)
            map_cls_labl:   (B,cls,mapsize,mapsize)   convert map_label value to onehot tensor
        Outputs:
            episode - generator with each output as 
                      dictionary, values are torch Tensor observations
                      {
                          image_cls :     L,B,cls,C,H,W
                          poses:    L,B,3  (x,y,yaw)
                          delta: == poses
                          maps:     L,B,cls,mapsize,mapsize
                          map_labl: same as input
                          map_cls_labl: same as input
                      }
        """

        if self.data_idx + self.batch_size > self.nsplit:
            self.data_idx = 0
        np.random.shuffle(self.index) #batch number
        episodes = {}
        for key_ in self.obs_keys:
            # if key_=='image' or key_=='depth' or key_=='imu': continue
            # key = 'rgb' if key_ == 'image_cls' else key_   # key='image' or 'delta' or 'depth' or 'imu' or 'image_cls'
            
            # print(key_, np.array(self.data_file[key_]).shape)
            # print(self.data_file['image_cls'][0,0])
            # print((self.data_file['delta']))
            if 'labl' not in key_:
                episodes[key_] = np.array(self.data_file[key_][:, self.index[self.data_idx:(self.data_idx + self.batch_size)], ...]) # (L,b,...)
                episodes[key_] = torch.Tensor(episodes[key_]).float()
            else:
                pass

        self.data_idx += self.batch_size

        poses = torch.zeros(self.max_steps, self.batch_size, 3, device=self.device) #pose:(L,b,3)
        for t in range(0, self.max_steps):
            # poses[t] = poses[t-1] + convert_polar2xyt(episodes['delta'][t]) #convert polar to xyt
            poses[t] = episodes['delta'][t] # x_y_yaw : in world coordinate
        # convert yaw angles to match coordinate system in self.angles
        # poses[:,:,2] = math.pi - poses[:,:,2]
        
        episodes['poses'] = poses

        if self.randomize_start_time:
            starts = np.random.randint(0, self.max_steps-self.num_steps, size=(1, ))
        else:
            starts = np.arange(0, self.max_steps-self.num_steps+1, self.num_steps).astype(np.int32)
        # print(f'starts:{starts}self.max_steps{self.max_steps}self.num_steps{self.num_steps}')
        for t in starts:
            # print('episodes_split')
            episodes_split = {
                key: (val[t:(t+self.num_steps)].to(self.device) if key!='map' else val.to(self.device))
                for key, val in episodes.items() 
            } # (num_steps, batch_size, ...)
            yield episodes_split


    def close(self):
        self.data_file.close()

    def reset(self):
        self.data_file = np.load(self.data_path)
        self.data_idx = 0

def add_imu_to_pose(imu, pre_pose, freq):
    '''
    Calculate pose candidates based on IMU data and Last pose
    input:
        imu     : (t, B, 3)
                    ax,ay,w_yaw -- linear accelaration for x,y; angular velocity for yaw
        pre_pose: (B,3) predicted pose from last step
        freq    : (1) sampling rate of imu data

    Output:
        absolute pose (t,B,3)
    '''
    axy     = imu[:,:,:2] #(t,B,2)
    w_yaw   = imu[:,:,2] #(t,B)
    device = pre_pose.device
    vxy             = it.cumtrapz(axy, dx=1/freq, axis=0, initial=0)
    displacement    = torch.tensor(it.cumtrapz(vxy, dx=1/freq, axis=0, initial=0))[-1,...].to(device) #B,2
    yaw             = torch.tensor(it.cumtrapz(w_yaw, dx=1/freq, axis=0, initial=0))[-1,:,None].to(device) #B,1
    rela_pose       = torch.cat(displacement,yaw, 1)
    abs_pose = pre_pose + rela_pose

    return abs_pose

def make_one_hot(map,cls=11):
    '''
    map: (B,N,N) where N is the map size
    output: 
            (B,C,N,N) where C is class number
    '''
    # print('map: ', map.shape)
    one_hot = torch.zeros(map.shape[0],cls,map.shape[1],map.shape[2], dtype=torch.int64, device=map.device)
    # print('onehot: ', one_hot.shape)
    target = one_hot.scatter(1, map[:,None].long(), 1)
    # print('target:', target.shape)
    return target
