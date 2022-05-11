import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, asnumpy, parse_shape
from einops.layers.torch import Rearrange, Reduce
from mapnet.utils import (
    flatten_two,
    unflatten_two,
    convert_map2world,
    convert_world2map,
    norm_angle,
    compute_relative_pose,
    process_image,
    process_maze_batch,
)
'''
Maintainance History:
utils
----------------------------
ver 1.0 - Dec 1th 2020
    Chnage evaluate_avd()
    - Using relative position instead of Absolute position
ver 1.1 - 
'''
# np.set_printoptions(precision=4,threshold=10000, edgeitems=100, linewidth=1000, suppress=True)

best={}
best['local'], best['ape'], best['de'] = 100,100,100
def compute_metrics(p, gt_world_pos, map_p_all,map_gt_all, map_shape, map_scale, angles, batch_size,imu_p_all=None,use_mapnet=False):
    """
    Inputs: 
        p - (L, bs, cls, H, W) numpy array of probabilities 
            at each time step for each test element
        gt_world_pos - (L, bs, 3) numpy array of (x, y, theta)
        angles - (nangles, ) torch Tensor

    Outputs:
        metrics     - a dictionary containing the different metrics measured
    """
    metrics = {}

    L, bs, nangles, H, W = p.shape
    f = bs//batch_size #The whole trjecotry are diveided into f segments (Real length = L*f)

    # Compute localization loss
    gt_pos = convert_world2map(
                 rearrange(torch.Tensor(gt_world_pos), 't b n -> (t b) n'),
                 map_shape,
                 map_scale,
                 angles,
             ).long() # (L*bs, 3) ---> (x, y, dir)

    logp = rearrange(np.log(p + 1e-12), 't b o h w -> (t b) o h w')
    logp_at_gtpos = logp[range(gt_pos.shape[0]), gt_pos[:, 2], gt_pos[:, 0], gt_pos[:, 1]] # (L*bs, )
    loss_pose = - logp_at_gtpos.mean().item()

    metrics['loc_loss'] = loss_pose
    p_map = rearrange(map_p_all, 't b o h w -> (t b) o h w')
    p_map = torch.from_numpy(p_map)
    p_map = (p_map + 1e-12).log()
    gt_maps = torch.from_numpy(rearrange(map_gt_all, 't b o h w -> (t b) o h w'))
    metrics['map_loss'] = F.kl_div(p_map,gt_maps,reduction='batchmean') 
    res=[]
    for i in range(L):
        a = rearrange(map_p_all[[i]], 't b o h w -> (t b) o h w')
        a=(torch.from_numpy(a)+ 1e-12).log()
        b=torch.from_numpy(rearrange(map_gt_all[[i]], 't b o h w -> (t b) o h w'))
        res.append((F.kl_div(a,b,reduction='batchmean')).numpy())
    res=np.array(res)
    # metrics['step_map_loss_0_10_98'] = res
    metrics['step_map_loss_avg'] = np.average(res)

    pred_pos = np.unravel_index(np.argmax(logp.reshape(L*bs, -1), axis=1), logp.shape[1:])
    pred_pos = np.stack(pred_pos, axis=1) # (L*bs, 3) shape with (theta_idx, x, y)
    pred_pos = np.ascontiguousarray(pred_pos[...,[1,2,0]]) # Convert to (x, y, theta_idx)
    pred_pos = torch.Tensor(pred_pos).long() # (L*bs, 3) --> (x, y, dir)
    pred_world_pos = convert_map2world(pred_pos, map_shape, map_scale, angles) # (L*bs, 3) --> (x_world, y_world, theta_world)
    pred_world_pos = asnumpy(pred_world_pos)
    gt_world_pos = rearrange(gt_world_pos, 't b n -> (t b) n')

    # Compute APE - average position error
    ape_all = np.linalg.norm(pred_world_pos[:, :2] - gt_world_pos[:, :2], axis=1)
    
    if f == 1:
        ape_all = rearrange(ape_all, '(t b) -> t b', t=L)
    else:
        ape_all = rearrange(ape_all, '(t f b) -> (f t) b',t=L,f=f)


    metrics['median/ape'] = np.median(ape_all, axis=0).mean().item()
    metrics['mean/ape'] = np.mean(ape_all).item()
    # Compute DE - direction error
    pred_angle = torch.Tensor(pred_world_pos[:, 2])
    gt_angle = torch.Tensor(gt_world_pos[:, 2])
    de_all = torch.abs(norm_angle(pred_angle - gt_angle))
    if f == 1:
        de_all = asnumpy(rearrange(de_all, '(t b) -> t b', t=L))
    else:
        de_all = asnumpy(rearrange(de_all, '(t f b) -> (f t) b',t=L,f=f))
    metrics['median/de'] = np.median(de_all, axis=0).mean().item()
    metrics['mean/de'] = np.mean(de_all).item()
    if best['local'] > metrics['loc_loss']: 
        best['local'] = metrics['loc_loss'] 
    if best['ape'] > metrics['mean/ape']: 
        best['ape'] = metrics['mean/ape'] 
    if best['de'] > metrics['mean/de']: 
        best['de'] = metrics['mean/de']

    ape_step = np.mean(ape_all, axis=1)
    de_step = np.mean(de_all, axis=1)
    for step in range(ape_step.shape[0]):
        index = 'step_' + str(step)
        metrics['ape/'+index] = ape_step[step].item()
        metrics['de/'+index] = de_step[step].item()

    print("Evaluation using {} episodes\n=========== metrics =============".format(bs))
    if use_mapnet:
        print('Map_prediction:\n{}\nMap_groundtruth:\n{}'.format(np.argmax(map_p_all,axis=2)[:,0], map_gt_all[0]))
    for k, v in metrics.items():
        try:
            print('{:<20s}: {:^10.3f}'.format(k,v))
        except:
            print('{:<20s}: {}'.format(k,v))
    print('=========== best =============')
    for k, v in best.items():
        print('{:<20s}: {:^10.3f}'.format(k,v))

    # ===================Print updated map if need===========================
    # print("----------- example ----------")
    # if f == 1:
    #     gt_world_pos = rearrange(gt_world_pos, '(t b) n -> t b n',t=L)
    #     pred_world_pos = rearrange(pred_world_pos, '(t b) n -> t b n',t=L)
    #     gt_pos = rearrange(gt_pos, '(t b) n -> t b n',t=L)
    #     pred_pos = rearrange(pred_pos, '(t b) n -> t b n',t=L)
    # else:
    #     gt_world_pos = rearrange(gt_world_pos, '(t f b) n -> (f t) b n',t=L,f=f)
    #     pred_world_pos = rearrange(pred_world_pos, '(t f b) n -> (f t) b n',t=L,f=f)
    #     gt_pos = rearrange(gt_pos, '(t f b) n -> (f t) b n',t=L,f=f)
    #     pred_pos = rearrange(pred_pos, '(t f b) n -> (f t) b n',t=L,f=f)

    # print("World_groundtruth-World_prediction-Relative_groundtruth-Relative_prediction")
    # for i in range(int(L*f)):
    #     try:
    #         print("Step:{:3},\tW_gt: {:50}, R_gt: {}\n\t\t\tW_p:  {:50}, R_p:  {}, imu:  {}\n"
    #             .format(i+1,
    #                     str(gt_world_pos[i,0,:]),
    #                     str(gt_pos[i,0,:]),
    #                     str(pred_world_pos[i,0,:]),
    #                     str(pred_pos[i,0,:]),
    #                     str(imu_p_all[i,0,:])))

    #     except:
    #         print("Step:{:3},\tW_gt: {:50}, R_gt: {}\n\t\t\tW_p:  {:50}, R_p:  {}\n"
    #             .format(i+1,
    #                     str(gt_world_pos[i,0,:]),
    #                     str(gt_pos[i,0,:]),
    #                     str(pred_world_pos[i,0,:]),
    #                     str(pred_pos[i,0,:])))        
    # print()
    return metrics

def evaluate_avd(model, eval_data_loader, config, step, device):

    batch_size = config['batch_size']
    map_shape = config['map_shape']
    map_scale = config['map_scale']
    angles = config['angles']
    env_name = config['env_name']
    max_batches = config['max_batches']
    gru_scalar = config['gru_scalar']
    
    mapnet = model['mapnet']
    eval_data_loader.reset()
    # Set to evaluation mode
    mapnet.eval()

    # Gather evaluation information
    eval_p_all,eval_gt_poses_all = [],[]
    map_gt_all, map_p_all = [], []
    maps_pred_all, maps_all,raw_maps_pred_all=[],[], []
    imu_p_all = []
    if max_batches == -1:
        num_batches = eval_data_loader.nsplit // batch_size
    else:
        num_batches = max_batches
    # print(f'num_batches:{num_batches}')
    for eval_ep in range(0, num_batches):
        episode_iter = eval_data_loader.sample_maploss()
        # Storing evaluation information per process.
        first_step = True
        count = 0
        for episode_batch in episode_iter:
            obs = episode_batch
            L, bs = obs['image_cls'].shape[:2]
            # obs['rgb'] = unflatten_two(process_image(flatten_two(obs['rgb'])), L, bs)
            obs_poses = obs['poses']
            if first_step:
                start_pose = obs_poses[0].clone()
                maps=None
                first_step = False
            # Transform the poses relative to the starting pose
            for l in range(L):
                obs_poses[l] = compute_relative_pose(start_pose, obs_poses[l]) # (x, y, theta)
            obs['pose_changes']=obs_poses[1:] #L-1,bs,3
            gt_poses = convert_world2map(
                 rearrange(obs_poses, 't b n -> (t b) n'),
                 map_shape,
                 map_scale,
                 angles,
             )
            gt_poses = rearrange(gt_poses, '(t b) n -> t b n', t=L) # (x, y, angle_idx)
            obs['gt_poses'] = gt_poses[1:]
            with torch.no_grad():
                # (L-1, bs, nangles, H, W)
                
                output,maps = mapnet(obs,maps)
                p_all = output['poses']
                maps_pred = F.softmax(output['maps_pred']*gru_scalar,dim=2)
                raw_maps_pred = output['maps_pred']
                maps_gt   = obs['maps'][1:]

            # append result to the list
            gt_world_poses = obs_poses[1:]
            eval_p_all.append(asnumpy(p_all))
            eval_gt_poses_all.append(asnumpy(gt_world_poses))
            maps_pred_all.append(asnumpy(maps_pred))
            raw_maps_pred_all.append(asnumpy(raw_maps_pred))
            maps_all.append(asnumpy(maps_gt))
            
            count+=1
            if count>1:
                break

    # concatenate result on the batch dimension
    eval_p_all = np.concatenate(eval_p_all, axis=1) # (L-1, bs, nangles, H, W)
    eval_gt_poses_all = np.concatenate(eval_gt_poses_all, axis=1) # (L-1, bs, 3)
    maps_pred_all = np.concatenate(maps_pred_all, axis=1) #(L-1,bs,cls,H,W)
    raw_maps_pred_all = np.concatenate(raw_maps_pred_all, axis=1) #(L-1,bs,cls,H,W)
    maps_all = np.concatenate(maps_all, axis=1) #(L-1,bs,cls,H,W)

    # print('='*10,'Maps','='*10)
    # for i in range(maps_all.shape[0]):
    #     for j in range(maps_all.shape[2]):
    #         # print(f'step:{i}, cls:{j}\nmaps_pred:{maps_pred_all[i,0,j]}\nmaps_labl:{maps_all[i,0,j]}\n')
    #         print(f'step:{i}, cls:{j}\nsoftmax(gru_scalar*maps_pred)*label:\n{maps_pred_all[i,0,j]*maps_all[i,0,j]}\nraw_maps_pred_all:\n{raw_maps_pred_all[i,0,j]}\n')
    
    eval_data_loader.close()   
    #print the result
    metrics = compute_metrics(eval_p_all, eval_gt_poses_all,maps_pred_all,maps_all, map_shape, map_scale, angles.cpu(), batch_size) 
    return metrics
