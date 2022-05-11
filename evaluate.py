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
from mapnet.arguments_gaussian import get_args
from mapnet.model import MapNet_yolo
from mapnet.utils import *
from mapnet.eval_utils_maploss import evaluate_avd
from mapnet.data_loader import DataLoaderAVD

args = get_args()

# set random seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.set_printoptions(precision=4,threshold=10000, edgeitems=100, linewidth=1000, sci_mode=False)
np.set_printoptions(precision=4,threshold=10000, edgeitems=100, linewidth=1000, suppress=True)

if args.cuda:
    torch.cuda.manual_seed(args.seed)

def main():
    device = torch.device("cuda:0" if args.cuda else "cpu")
    args.feat_dim = args.num_cls #
    args.map_shape = (args.feat_dim, args.map_size, args.map_size)
    args.local_map_shape = (args.feat_dim, args.local_map_size, args.local_map_size)
    args.angles = torch.Tensor(np.radians(np.linspace(0, 359, 360//args.angles_intvl))).to(device) 

    val_loader = DataLoaderAVD(
            args.data_path_val,
            1,
            args.num_steps,
            'val',
            device,
            args.seed,
            args.env_name,
            max_steps=None,
            randomize_start_time=False,
            n_cls=args.num_cls
    )
    # Load trained model
    save_path = os.path.join(args.save_dir, args.savedate) # ./trained_models/adv_models/arg.savedate
    pp.pprint(vars(args))

    print('==============> Preparing models')
    #=========================Define model=============================
    mapnet_config = {
        'map_shape': args.map_shape,
        'batch_size': args.batch_size,
        'local_map_shape': args.local_map_shape,
        'angles': args.angles,
        'map_scale': args.map_scale,
        'use_gt_pose': False,
        'num_cls': args.num_cls,
        'gru':args.gru,
        'gru_scalar':args.gru_scalar,
    }

    # Define optimizers
    mapnet = MapNet_yolo(mapnet_config)
    j_start = 0
    # path = os.path.join(save_path, args.savedate , ".pt")
    path = os.path.join(save_path, "avd", ".pt")
    #=========================resume from past=============================
    if os.path.isfile(path):
        print("Resuming from old model!")
        checkpoint = torch.load(path)
        mapnet.load_state_dict(checkpoint['model_state_dict'], strict=False)
        # Resume settings
        j_start = checkpoint['epoch']
    else:
        print("Load fail! Check path to saved model!")
    mapnet.to(device)   
    mapnet.eval()

    # =================== Eval metrics ================
    eval_val_metrics_deque = None
    evaluate = evaluate_avd 
    eval_config = {
        'batch_size': 1,
        'split': 'val',
        'seed': args.seed,
        'map_shape': args.map_shape,
        'map_scale': args.map_scale,
        'angles': args.angles,
        'env_name': args.env_name,
        'max_batches': 1,
        'gru_scalar':args.gru_scalar,
    }
    eval_model = {'mapnet': mapnet}
    print('==============> Starting evaluation')

    # =================== Eval ====================

    for j in tqdm(range(10)):
        # Evaluating on val split
        print('============ Evaluating on val split ============')
        val_metrics = evaluate(eval_model, val_loader, eval_config, step=j, device=device)

        if eval_val_metrics_deque is None:
            eval_val_metrics_deque = {}
        for k, v in val_metrics.items():
            if k not in eval_val_metrics_deque:
                eval_val_metrics_deque[k] = deque(maxlen=10)
            eval_val_metrics_deque[k].append(v)
        # for key, value_deque in eval_val_metrics_deque.items():
        #     value = np.mean(value_deque).item()

if __name__ == "__main__":
    main()
