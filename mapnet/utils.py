import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math
import numpy as np
from einops import rearrange, reduce, asnumpy
'''
Maintainance History:
utils
----------------------------
ver 1.0 - Nov 23th 2020
    Chnage parts of following function
    - convert_world2map()
    - convert_map2world()
    - compute_relative_pose()
ver 2.0 
    - process_image()
'''
def flatten_two(x):
    try:
        # print("faltten",x.shape)
        return x.view(-1, *x.shape[2:])
    except:
        print("exp faltten",x.shape)
        return x.contiguous().view(-1, *x.shape[2:])

def unflatten_two(x, sh1, sh2):
    try: 
        # print("unfaltten",x.shape,sh1,sh2)
        return x.view(sh1, sh2, *x.shape[1:])
    except:
        print("exp unfaltten",x.shape,sh1,sh2)
        return x.contiguous().view(sh1, sh2, *x.shape[1:])

def get_camera_parameters(env_name, obs_shape):
    # Note: obs_shape[*]/K done because image features and depth will be downsampled by K
    # These values are obtained from Matterport3D files. 
    if env_name == 'avd':
        K   = 1 # orig image size / feature map size
        fx  = 1070.00 * (obs_shape[2]) / 1920.0
        fy  = 1069.12 * (obs_shape[1]) / 1080.0
        cx  = 927.269 * (obs_shape[2]) / 1920.0
        cy  = 545.760 * (obs_shape[1]) / 1080.0
        fov = math.radians(75)
    else:
        K   = None 
        fx  = None 
        fy  = None 
        cx  = None 
        cy  = None 
        fov = None 

    camera_parameters = {'K': K, 'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy, 'fov': fov}
    return camera_parameters

# Normalize angles between -pi to pi
def norm_angle(x):
    # x - torch Tensor of angles in radians
    return torch.atan2(torch.sin(x), torch.cos(x))

def process_image(img):
    # img - (bs, C, H, W)
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    img_proc = img.float() / 255.0

    # img_proc[:, 0] = (img_proc[:, 0] - mean[0]) / std[0]
    # img_proc[:, 1] = (img_proc[:, 1] - mean[1]) / std[1]
    # img_proc[:, 2] = (img_proc[:, 2] - mean[2]) / std[2]

    return img_proc

def convert_world2map_ori(pose, map_shape, map_scale, angles):
    """
    Inputs:
        pose - (bs, 3) Tensor agent pose in world coordinates (x, y, theta) [0,10] [0,10] [-pi,pi]
        map_shape - (f, h, w) tuple     32x31x31
        map_scale - scalar              1
        angles - (nangles, ) Tensor     [-pi,pi) dim=12
        eps - scalar angular bin-size   

    Conventions:
        world positions - X Upward, Y rightward, origin at center
        map positions - X rightward, Y downward, origin at top-left
    """
    x = pose[:, 0]
    y = pose[:, 1]
    mh, mw = map_shape[1:]
    nangles = float(angles.shape[0])
    # This convention comes from transform_to_map() in model_pose.py [0,100] when mw=101 
    ref_on_map_x = torch.clamp((mw-1)/2 + x/map_scale, 0, mw-1).round().long() 
    ref_on_map_y = torch.clamp((mh-1)/2 - y/map_scale, 0, mh-1).round().long()
    # Mapping heading angles to map locations [6,7,8,9,10,11,0,1,2,3,4,5]
    ref_on_map_dir = ((pose[:, 2]+math.pi) * nangles / (2*math.pi)).round().long() % nangles
    #np.set_printoptions(precision=2,linewidth=250,threshold=100000)
    #print('R, Theta, X, Y, map_x, map_y, phi_head, map_dir')
    #print(torch.stack([r, t, x, y, ref_on_map_x.long().float(), ref_on_map_y.long().float(), normalized_angles*180.0/math.pi, ref_on_map_dir.float()], dim=1)[:5].detach().cpu().numpy())
    #pdb.set_trace()
    # print('convert_world2map','pose:',pose,'ref_on_map:',torch.stack([ref_on_map_x, ref_on_map_y, ref_on_map_dir], dim=1),sep='\n')
    return torch.stack([ref_on_map_x, ref_on_map_y, ref_on_map_dir], dim=1)

def convert_map2world_ori(pose, map_shape, map_scale, angles):
    """
    Inputs:
        pose - (bs, 3) Tensor agent pose in map coordinates (x, y, theta) 
             - Note: theta is discrete angle idx
        map_shape - (f, h, w) tuple
        map_scale - scalar
        angles - (nangles, ) Tensor

    Conventions:
        world positions - X rightward, Y upward, origin at center
        map positions - X rightward, Y downward, origin at top-left
    """
    x = pose[:, 0].float()
    y = pose[:, 1].float()
    angle_idx = pose[:, 2].long()
    mh, mw = map_shape[1:]

    x_world = (x - (mw-1)/2) * map_scale
    y_world = ((mh-1)/2 - y) * map_scale
    theta_world = angles[angle_idx]
    # print('convert_map2world','ref_on_map:',pose,'pose:',torch.stack([x_world, y_world, theta_world], dim=1),sep='\n')
    return torch.stack([x_world, y_world, theta_world], dim=1)

def convert_world2map(pose, map_shape, map_scale, angles):
    """
    Inputs:
        pose - (bs, 3) Tensor agent pose in world coordinates (x, y, theta) [0.5,9.5] [0.5,9.5] [0,2pi]
        map_shape - (f, h, w) tuple     32x31x31
        map_scale - scalar              1
        angles - (nangles, ) Tensor     [0,2pi) dim=12
        eps - scalar angular bin-size   

    Conventions:
        Relative world positions - X downward, Y rightward, origin at center
        map positions - X downward, Y rightward, origin at top-left
    """
    x = pose[:, 0]
    y = pose[:, 1]
    mh, mw = map_shape[1:]
    nangles = float(angles.shape[0])
    # This convention comes from transform_to_map() in model_pose.py [0,100] when mw=101 
    ref_on_map_x = torch.clamp((mw-1)/2 + x/map_scale, 0, mw-1).round().long() 
    ref_on_map_y = torch.clamp((mh-1)/2 + y/map_scale, 0, mh-1).round().long()
    # Mapping heading angles to map locations [6,7,8,9,10,11,0,1,2,3,4,5]
    ref_on_map_dir = (((pose[:, 2]) * nangles / (2*math.pi)).round() % nangles).long()
    return torch.stack([ref_on_map_x, ref_on_map_y, ref_on_map_dir], dim=1)

def convert_map2world(pose, map_shape, map_scale, angles):
    """
    Inputs:
        pose - (bs, 3) Tensor agent pose in map coordinates (x, y, theta) 
             - Note: theta is discrete angle idx
        map_shape - (f, h, w) tuple
        map_scale - scalar
        angles - (nangles, ) Tensor

    Conventions:
        Relative world positions - X rightward, Y upward, origin at center
        map positions - X downward, Y rightward, origin at top-left
    """
    x = pose[:, 0].float()
    y = pose[:, 1].float()
    angle_idx = pose[:, 2].long()
    mh, mw = map_shape[1:]

    x_world = (x - (mw-1)/2) * map_scale
    y_world = (y - (mh-1)/2) * map_scale
    theta_world = angles[angle_idx]
    # print('convert_map2world','ref_on_map:',pose,'pose:',torch.stack([x_world, y_world, theta_world], dim=1),sep='\n')
    return torch.stack([x_world, y_world, theta_world], dim=1)




def convert_polar2xyt(poses):
    """
    poses - (bs, 3) torch Tensor of (r, phi, theta) poses
    converts to (x, y, theta)
    """
    r, phi, theta = poses[:, 0], poses[:, 1], poses[:, 2]
    x = r * torch.cos(phi)
    y = r * torch.sin(phi)
    poses_xyz = torch.stack([x, y, theta], dim=1)
    return poses_xyz

def compute_relative_pose(pose_a, pose_b):
    """
    Compute pose of pose_b in the egocentric coordinate frame of pose_a.
    Inputs:
        pose_a - (bs, 3) --- (x, y, theta)
        pose_b - (bs, 3) --- (x, y, theta)

    Conventions:
        The origin is at the center of the map.
        X is upward with agent's forward direction
        Y is rightward with agent's rightward direction
    """

    x_a, y_a, theta_a = pose_a[:, 0], pose_a[:, 1], pose_a[:, 2]
    x_b, y_b, theta_b = pose_b[:, 0], pose_b[:, 1], pose_b[:, 2]

    r_ab = torch.sqrt((x_a - x_b)**2 + (y_a - y_b)**2) # (bs, )
    phi_ab = torch.atan2(y_b - y_a, x_b - x_a) # (bs, )
    theta_ab = theta_b - theta_a # (bs, )
    #atan2->range(-pi,pi) atan2(y,x)=atan2(rsin_theta,rcos_theta)
    theta_ab = torch.atan2(torch.sin(theta_ab), torch.cos(theta_ab))

    x_ab = torch.stack([
        r_ab * torch.cos(phi_ab),
        r_ab * torch.sin(phi_ab),
        theta_ab,
    ], dim=1) # (bs, 3)
    # print("compute_relative_pose:", "pose_a:",pose_a, "pose_b:",pose_b, "x_ab:",x_ab)
    return x_ab

def process_maze_batch(batch, device):
    for k in batch.keys():
        # Convert (bs, L, ...) -> (L, bs, ...)
        batch[k] = batch[k].transpose(0, 1).contiguous().to(device).float()
    # Rotate image by 90 degrees counter-clockwise --- agent is facing upward
    batch['rgb'] = torch.flip(batch['rgb'].transpose(3, 4), [3]) # (bs, L, 2, H, W)
    # Converting to world coordinates convention
    x = -batch['poses'][..., 1]
    y = batch['poses'][..., 0]
    t = batch['poses'][..., 2]
    batch['poses'] = torch.stack([x, y, t], dim=2)
    return batch

def save_trajectory(inputs,labels,path,angles,name='train'):
    '''
    inputs:L-1,bs,nangles,H,W
    labels:L-1,bs,3
    '''
    L, bs, nangles, H, W = inputs.shape
    inputs = asnumpy(inputs)
    labels = asnumpy(labels)
    angles = angles.cpu()
    pred_pos = np.unravel_index(np.argmax(inputs.reshape(L*bs, -1), axis=1), inputs.shape[1:])
    pred_pos = np.stack(pred_pos, axis=1) # (L*bs, 3) shape with (theta_idx, y, x)
    pred_pos = np.ascontiguousarray(np.flip(pred_pos, axis=1)) # Convert to (x, y, theta_idx)
    pred_pos = torch.Tensor(pred_pos).long() # (L*bs, 3) --> (x, y, dir)
    pred_world_pos = convert_map2world(pred_pos, (nangles, H, W), map_scale=1, angles=angles) # (L*bs, 3) --> (x_world, y_world, theta_world)
    pred_world_pos = rearrange(asnumpy(pred_world_pos), '(t b) n -> t b n',t=L)
    np.savez(os.path.join(path,name),pred=pred_world_pos, gt=labels)
