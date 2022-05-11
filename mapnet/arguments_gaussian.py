import argparse

import torch

def str2bool(v):
    if v.lower() in ['y', 'yes', 't', 'true']:
        return True
    return False

def get_args():
    parser = argparse.ArgumentParser(description='RL')

    #training
    parser.add_argument('--num-updates', type=int, default=90001)
    parser.add_argument('--savedate', type=str, default='Testname',
                        help='Name of log file and saved model and tensorboard')
    parser.add_argument('--finetune', type=str, default='False',
                        help='True or False')
    parser.add_argument('--log-interval', type=int, default=5,
                        help='log interval, one log per n updates (default: 100)')
    parser.add_argument('--save-interval', type=int, default=3,
                        help='save interval, one save per n updates (default: 10000)')
    parser.add_argument('--eval-interval', type=int, default=5,
                        help='eval interval, one eval per n updates (default: 100)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    #env related
    parser.add_argument('--data-path', type=str, default='./dataset/train_data.npz',
                        help='Path to file containing source dataset')
    parser.add_argument('--data-path-val', type=str, default='./dataset/test_data.npz',
                        help='Path to file containing source dataset')
    parser.add_argument('--batch-size', type=int, default=36)
    parser.add_argument('--num-steps', type=int, default=100, 
                        help='number of forward steps (max: 100)')
    parser.add_argument('--num-cls', default=11,
                        help='number of labels in the environment')
    parser.add_argument('--angles_intvl', default=3,
                        help='number of labels in the environment')
    parser.add_argument('--env-name', default='avd',
                        help='environment to train on [ avd | maze ]')
    parser.add_argument('--map-size', type=int, default=11,
                        help='dimension of memory')
    parser.add_argument('--local-map-size', type=int, default=11,
                        help='dimension of local ground projection')
    parser.add_argument('--map-scale', type=float, default=1,
                        help='number of pixels per grid length')

    #learning rate related
    parser.add_argument('--lr', type=float, default=2e-2,
                        help='learning rate (default: 2e-5)')
    parser.add_argument('--lr_stepsize', nargs='+',type=int, default=[2e4,8e4],
                        help='learning rate stepsize (default: 2e4)')
    parser.add_argument('--lr_gamma', type=float, default=0.5,
                        help='learning rate gamma(default: 0.1)')     
    parser.add_argument('--gauss_sigma', type=float, default=0.55,
                        help='learning rate gamma(default: 0.45)')    

    #gradient related
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument('--gru_scalar', type=float, default=10,
                        help='weight of gru output for better softmax result(default: 10)')  
    parser.add_argument('--cls_scalar', type=float, default=1,
                        help='weight of object loss(default: 10)')                                
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')


    #log file dir
    parser.add_argument('--log-dir', default='./logs',
                        help='directory to save agent logs (default: ./logs')
    parser.add_argument('--save-dir', default='./trained_models',
                        help='directory to save agent logs (default: ./trained_models/)')
    
    parser.add_argument('--use_gt_pose', type=str, default='True',
                        help='True or False')    
    parser.add_argument('--gru', type=str, default='conv',help='conv, normh, conv_normh, for different RNN cell')

    ####################### Evaluation arguments ##########################
    parser.add_argument('--load-path', type=str, default='model.pt')
    parser.add_argument('--cnn-path', type=str, default='./Resnet_pretrain/trained_models/Resnet_pretrain_Sep30th/20000encoder.pt',
                        help='Path to file containing source dataset')  
    
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.num_refs = 1

    return args
