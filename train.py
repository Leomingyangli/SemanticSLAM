import os
import time
import numpy as np
import pprint as pp
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from tqdm import tqdm
from collections import deque
from einops import rearrange
import mapnet.algo.supervised_mapnet_maploss as algo
from mapnet.arguments_gaussian import get_args
from mapnet.model import MapNet_yolo
from mapnet.utils import *
from mapnet.data_loader import DataLoaderAVD

args = get_args()

# set random seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# create folder for log
try:
    os.makedirs(args.log_dir)
except OSError:
    pass

def main():
    device = torch.device("cuda:0" if args.cuda else "cpu")
    args.feat_dim = args.num_cls 
    args.map_shape = (args.feat_dim, args.map_size, args.map_size)
    args.local_map_shape = (args.feat_dim, args.local_map_size, args.local_map_size)
    args.angles = torch.Tensor(np.radians(np.linspace(0, 359, 360//args.angles_intvl))).to(device) 
    # dataloader
    train_loader = DataLoaderAVD(
            args.data_path,
            args.batch_size,
            args.num_steps,
            'train',
            device,
            args.seed,
            args.env_name,
            max_steps=None,
            randomize_start_time=False,
            n_cls=args.num_cls
    ) 

    # Load trained model
    save_path = os.path.join(args.save_dir, args.savedate) # ./trained_models/arg.savedate
    try:
        os.makedirs(save_path)
    except OSError:
        pass
    tbwriter = SummaryWriter(log_dir=args.log_dir+'/'+args.savedate+'/')
    pp.pprint(vars(args))

    print('==============> Preparing models')
    #=========================Define model=============================
    mapnet_config = {
        'map_shape': args.map_shape,
        'batch_size': args.batch_size,
        'local_map_shape': args.local_map_shape,
        'angles': args.angles,
        'map_scale': args.map_scale,
        'use_gt_pose': args.use_gt_pose,
        'num_cls': args.num_cls,
        'gru':args.gru,
        'gru_scalar':args.gru_scalar,
    }

    # Define models and optimizer
    mapnet = MapNet_yolo(mapnet_config)
    fltr        = lambda x: [param for param in x if param.requires_grad == True]
    optimizer   = optim.Adam(fltr(mapnet.parameters()), lr=args.lr, amsgrad=True)
    
    j_start = 0
    path = os.path.join(save_path, args.savedate + ".pt")

    #=========================resume from past=============================
    if os.path.isfile(path):
        print("Resuming from old model!")
        checkpoint = torch.load(path)
        mapnet.load_state_dict(checkpoint['model_state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # Resume settings
        j_start = checkpoint['epoch']
        tbwriter = SummaryWriter(log_dir=args.log_dir+'/'+args.savedate+str(j_start)+'/') 
    else:
        print("Train from the scratch!")
    mapnet.to(device)   
    mapnet.train()

    print('==============> Preparing training algorithm')
    #================================================================================
    # =================== Define training algorithm ====================
    algo_config = {
        'mapnet': mapnet,
        'optimizer': optimizer,
        'max_grad_norm': args.max_grad_norm,
        'angles': args.angles,
        'lr_stepsize': args.lr_stepsize,
        'lr_gamma': args.lr_gamma,
        'gauss_sigma': args.gauss_sigma,
        'scale':args.map_scale,
    }
    supervised_agent = algo.SupervisedMapNet(algo_config)

    # =================== Eval metrics ================
    all_losses_deque     = None
    print('==============> Starting training')
    #================================================================================
    # =================== Training ====================
    start = time.time()
    episode_iter = None

    for j in tqdm(range(j_start+1, args.num_updates)):
        # Initialize things
        num_batches  = 0
        # Sample data
        sampled_data = False
        while not sampled_data:
            if episode_iter is None:
                episode_iter = train_loader.sample_maploss()
                first_step = True
            try:
                batch = next(episode_iter)
                sampled_data = True
            except StopIteration:
                episode_iter = None
        
        num_batches += 1
        obs = batch
        L, bs = obs['image_cls'].shape[:2] #L=5 steps;  bs=4*10 ->means 10 scences with 4 different trjactory
        obs_poses = obs['poses'] #L,bs,3
        '''Fix starting position'''
        if first_step == True:
            start_pose = obs_poses[0].clone()
            maps = None
            # first_step = False

        # Transform the poses relative to the starting pose
        for l in range(L):
            obs_poses[l] = compute_relative_pose(start_pose, obs_poses[l]) # (x, y, theta) (L,b,3)
        obs['pose_changes']=obs_poses[1:] #L-1,bs,3
        gt_poses = convert_world2map(
            rearrange(obs_poses, 't b n -> (t b) n'),
            args.map_shape,
            args.map_scale, 
            args.angles,
        )
        gt_poses = rearrange(gt_poses, '(t b) n -> t b n', t=L) # (x, y, angle_idx)
        obs['gt_poses'] = gt_poses[1:] # 5-1=4 poses (L-1,b,3) 

        # Perform update
        all_losses,maps = supervised_agent.update(obs,maps,scalar=args.gru_scalar,cls_scalar=args.cls_scalar)

        if all_losses_deque is None:
            all_losses_deque = {}
        for k, v in all_losses.items():
            if k not in all_losses_deque:
                all_losses_deque[k] = deque(maxlen=10)
            all_losses_deque[k].append(v)
        
        # =================== Save model ====================
        if j % args.save_interval == args.num_updates and args.save_dir != "":
            save_model          = mapnet
            path = os.path.join(save_path, args.savedate + ".pt")
            torch.save({
                'epoch':j,
                'model_state_dict':save_model.state_dict(),
                'optimizer_state_dict':supervised_agent.optimizer.state_dict(),
                }, path)
            np.save(os.path.join(save_path, 'maps' + str(j)),np.asarray(maps.detach().cpu()))

        # =================== Logging data ====================
        total_num_steps = (j + 1 - j_start) * args.batch_size * args.num_steps * num_batches

        if j % args.log_interval == 0:
            end                 = time.time()
            print_string        = '===> Updates {}, #steps {}, FPS {} steps/s '.format(j, total_num_steps, int(total_num_steps / (end - start))) 
            for loss_type, loss_deque in all_losses_deque.items():
                loss = np.mean(loss_deque).item()
                print_string += ', {}: {:.3f}'.format(loss_type, loss)
                tbwriter.add_scalar('train/'+loss_type, loss, j)
            try:
                tbwriter.add_histogram('GRU/ih', mapnet.aggregate.main.weight_ih_l0.grad.clone().cpu().numpy(),j)
                tbwriter.add_histogram('GRU/hh', mapnet.aggregate.main.weight_hh_l0.grad.clone().cpu().numpy(),j)
            except:
                pass
            print(print_string)
            print("%dth learning rate is %f" % (j,supervised_agent.optimizer.param_groups[0]['lr']))

    tbwriter.close()

if __name__ == "__main__":
    main()
