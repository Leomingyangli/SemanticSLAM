import torch
import torch.nn as nn
import torch.nn.functional as F
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = "cuda:0" if torch.cuda.is_available() else "cpu"
def gaussian_fnc(array_like_hm, mean, sigma):
    """modifyed version normal distribution pdf, vector version"""
    array_like_hm -= mean
    x_term = array_like_hm[:,0] ** 2
    y_term = array_like_hm[:,1] ** 2
    z_term = array_like_hm[:,2] ** 2
    exp_value = - (x_term + y_term + z_term) / 2 / pow(sigma, 2)
    return torch.exp(exp_value)

def gaussian_filter(size=3, sigma=0.75, normalize=True):
    '''Generate one gaussian kernel (size,size)
    '''
    mean = torch.tensor([[1,1,1]]).to(device).float() #the center of kernel
    x_dim = torch.arange(size,dtype=torch.float).to(device)
    y_dim = torch.arange(size,dtype=torch.float).to(device)
    z_dim = torch.arange(size,dtype=torch.float).to(device)
    x,y,z = torch.meshgrid(x_dim, y_dim, z_dim)
    grid = torch.cat((x.reshape(-1,1),y.reshape(-1,1),z.reshape(-1,1)),dim=-1)
    # print(grid,grid.shape)
    gauss = gaussian_fnc(grid, mean, sigma).reshape(size,size,size)
    # print(gauss.shape,gauss)
    return gauss

class Gaussian3d():
    def __init__(self,size=3, outchan=1, sigma=0.75, normalize=True):
        '''
        size:       gauss kernel size (radius of gaussian heatmap)
        outchan:    dim of output channel 
        sigma:      variance of Gaussian distribution
        normalize:  make output sum to 1 for each channel
        '''
        self.size=size
        self.outchan=outchan
        self.sigma=sigma
        self.normalize=normalize
        self.gauss = gaussian_filter(self.size, self.sigma, self.normalize)
    
    def change_sigma(self):
        if self.sigma<=0.3:
            self.sigma = 0.3
        self.gauss = gaussian_filter(self.size, self.sigma, self.normalize)

    def __call__(self,inputs):
        '''
        F.conv2d:
        input – input tensor of shape (minibatch,in_channels,iH,iW)
        filters - filters of shape (out_channels, groups/in_channels,kH,kW)
        '''
        # self.gauss = gaussian_filter(self.size, self.sigma, self.normalize)
        batch,inchan,layersize,insize,_ = inputs.shape
        device = inputs.device
        target = torch.zeros(batch,inchan,layersize+2,insize,insize).to(device)
        target[:,:,1:-1,:,:] = inputs
        # print("target:\n",target,target.shape)
        filters = torch.zeros(self.outchan,inchan,self.size,self.size,self.size).to(device) + self.gauss
        # print('Gaussian kernel: ',filters,filters.shape)
        out = F.conv3d(target, filters, padding=(self.size-1)//2,groups=1)
        if self.normalize:
            # print('sum',out.sum(dim=(-1,-2),keepdim=True).repeat(1,1,insize,insize).shape)
            out = out/(out+1e-12).sum(dim=(-1,-2,-3,-4),keepdim=True).repeat(1,inchan,layersize+2,insize,insize)
        new_out = out[:,:,1:-1,:,:]
        new_out[:,:,0,:,:] += out[:,:,-1,:,:]
        new_out[:,:,-1,:,:] += out[:,:,0,:,:]
        return new_out

