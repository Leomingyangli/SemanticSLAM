import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
'''
Maintainance History:
----------------------------
ver 1.0 - Nov 3th 2020
    -Heat map function:
    -Input file: .npz{R: , P:}
    -R:Bx nangles x size x size, P:Bx3     where B is batchsize
'''

class plt_heatmap():
    def __init__(self,root_dir):
        self.fig = plt.figure(facecolor="snow")
        self.root_dir = root_dir + '/figure'

    def save_fig(self,R,v_q,idx):
        if R.ndim == 4:
            for i in range(R.shape[0]):
                for j in range(R.shape[1]):
                    sns_plot = sns.heatmap(R.squeeze()[i,j], xticklabels=10, yticklabels=10) # cmap='plasma' 颜色主题风格   xticklabels 代表步长
                    sns_plot.tick_params(labelsize=10, direction='in') # heatmap 刻度字体大小   
                    cbar = sns_plot.collections[0].colorbar
                    cax = plt.gcf().axes[-1]
                    cax.tick_params(labelsize=10, right=False) # colorbar 刻度字体大小
                    cbar.set_label('Prb', fontdict={'weight':'normal','size': 13})
                    plt.title('{}-{}'.format(idx,i),fontdict={'weight':'normal','size': 13})
                    plt.xlabel('X={} Angle={}'.format(v_q[i,0], v_q[i,2]), fontdict={'weight': 'normal', 'size': 13})
                    plt.ylabel('Y={}'.format(v_q[i,1]), fontdict={'weight': 'normal', 'size': 13})
                    plt.gca().invert_yaxis()

                    self.fig.savefig(os.path.join(self.root_dir, "{}_{}_{}.png".format(idx,i,j)), transparent=True, dpi=300, pad_inches = 0)
                    plt.clf()
        else:
            print('Dimension is wrong!!!')
if __name__ == '__main__':
    root_dir = './heatmap'
    ida=19000
    a='train'
    #a='test'
    name = '{}_{}'.format(ida,a)
    inpath = os.path.join(root_dir, name)
    # N = 100
    # R = torch.rand(4,1,N,N)
    # V = torch.tensor([[3,4,5],[1,-1,1],[2,-2,2],[6,-6,0]])
    # np.savez(outpath, R=R.numpy(), P=V.numpy())
    npzfile = np.load(inpath+'.npz')
    r= npzfile['R']
    # v= ((npzfile['P']+1)*50).astype(int)
    v = npzfile['P']
    print(r.shape[0],v.shape[0])
    draw_fig = plt_heatmap(root_dir)
    draw_fig.save_fig(r,v,name)