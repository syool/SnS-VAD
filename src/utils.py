import torch

import numpy as np
from glob import glob
import math
import random

import time
import os


def seeds(val):
    ''' generate fixed random values '''
    random.seed(val)
    np.random.seed(val)
    torch.manual_seed(val)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(val)
        torch.cuda.manual_seed_all(val)


def label_encapsule(labels, path, clip_length):
    ''' encapsule labels w.r.t the window size '''
    videos = sorted(glob(os.path.join(path, '*')))
    
    capsules = []
    count = 0
    for vid in videos:
        frames = sorted(glob(os.path.join(vid, '*')))
            
        n_frames = len(frames)
        l = labels[count:count+n_frames]
        
        cap = l[clip_length:]
        capsules = np.append(capsules, cap)
        
        count += n_frames
    
    return capsules


def psnr(mse):
    return 10 * math.log10(1/mse)


def score_norm(arr):
    ''' input must be a numpy array '''
    return (arr - arr.min()) / (arr.max() - arr.min())


def shuffle(x):
    batch = x.size(0)
    
    for i in range(batch):
        ready = x[i].squeeze()
        idx = torch.randperm(ready.shape[0])
        shuffled = ready[idx].view(ready.size())
        x[i] = shuffled
    
    return x


def display_args(args):
    dt = time.strftime('%y-%m-%d %I:%M %p', time.localtime())
    
    print()
    print(f'experiment started at {dt}')
    print()
    print('================= cuda settings ==================')
    print(f'CUDA Device: {args.cuda}    CUDNN Benchmark: {args.cudnn_benchmark}    CUDNN Deterministic: {args.cudnn_deterministic}')
    print(f'============== optimization settings =============')
    print(f'dataset: {args.dataset}    epoch: {args.epoch}    batch: {args.batch}    learning rate: {args.lr}')
    print(f'clip length: {args.clip_length}    init method: {args.init_method}    seed: {args.seed}    workers: {args.num_workers}')
    # print(f'================ pruning settings ================')
    # print(f'pruning: {args.prune}    pruning selection: {args.prune_select}    pruning scope: {args.prune_scope}')
    # print(f'pruning reset: {args.prune_reset}    pruning scale: {args.prune_scale}    pruning percentage: {args.prune_percent}')
    # print(f'pruning iteration: {args.prune_it}    rewinding iteration: {args.rewind_it}    light stat display: {args.light_stats}')
    print()