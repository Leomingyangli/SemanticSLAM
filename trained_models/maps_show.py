import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import numpy as np
import os

np.set_printoptions(linewidth=10000,threshold=100000,precision=6,suppress=True)
path = './adv_models'
name1 = './Jan26th_maploss_env1/maps90000.npy'
name2 = './Jan26th_maploss_env3/maps90000.npy'


def main():
    data1 = np.load(name1)[0] #(bs, cls, map_size, map_size)
    data2 = np.load(name2)[0]
    print(f'env1/maps90000:\n{data1}')
    print(f'env3/maps90000:\n{data2}')

if __name__ == '__main__':
    main()