import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.backends import cudnn

from sort.net import Sortnet
from pred.net import Prednet
from .load import trainloader
from .utils import seeds, display_args, shuffle
from .initialize import Initialize

import time
import os


class Train():
    def __init__(self, args) -> None:
        super(Train, self).__init__()
        self.device = torch.device(f'cuda:{args.cuda}' \
                                   if torch.cuda.is_available() \
                                   else 'cpu')
        cudnn.benchmark = True if args.cudnn_benchmark else False
        cudnn.deterministic = True if args.cudnn_deterministic else False
        seeds(args.seed)
        
        frame_path = f'{args.data_path}/{args.dataset}/training/frames'
        self.loader = trainloader(frame_path=frame_path,
                                  batch=args.batch,
                                  num_workers=args.num_workers,
                                  window=args.clip_length)
        
        self.log_path = f'{args.log_path}/{args.dataset}'
        os.makedirs(self.log_path, exist_ok=True)
        
        self.lamb = 0.98
        self.args = args
        
    def run(self):
        display_args(self.args)
        
        weights_init = Initialize(self.args.init_method)
        
        sortnet = Sortnet().to(self.device)
        sortnet = nn.DataParallel(sortnet)
        sortnet.apply(weights_init)
        # sortnet.load_state_dict(torch.load('/root/shuso/logs/ped2/sort60.pth',
        #                                map_location=f'cuda:{self.args.cuda}'))
        
        # net = Prednet(clip_length=11).to(self.device)
        # net.apply(weights_init)
        
        optimizer = optim.Adam(sortnet.parameters(), lr=self.args.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epoch)
        
        MSE = nn.MSELoss().to(self.device)
        
        sortnet.train()
        # net.train()
        for epoch in range(self.args.epoch):
            for i, frame in enumerate(self.loader):
                frame = Variable(frame).to(self.device)
                
                optimizer.zero_grad()
                
                # shuffled = shuffle(frame)
                # sofr = sortnet(shuffled)
                
                # sofr = sofr.permute(1,0,2,3,4)
                # sofr = sofr.squeeze(dim=0)
                
                # output = net(sofr[:,:-1])
                # loss = MSE(output, sofr[:,-1:])
                
                shuffled = shuffle(frame)
                output = sortnet(shuffled)
                
                loss = MSE(output, shuffled)
                
                loss.backward()
                optimizer.step()
                
                if i % 10 == 9:
                    print(f'{self.args.dataset}:',
                          f'Epoch {epoch+1}/{self.args.epoch}',
                          f'Batch {i+1}/{len(self.loader)}',
                          f'recon.: {loss.item():.6f}')
                
            scheduler.step()
            
            torch.save(sortnet.module.state_dict(), f'{self.log_path}/pred{str(epoch+1).zfill(2)}.pth')
        
        ftime = time.strftime('%m-%d_%I:%M%p', time.localtime())
        picklef = f'final_batch{self.args.batch}_seeds{self.args.seed}_clip{self.args.clip_length}_run{ftime}'
        
        torch.save(sortnet.module.state_dict(), f'{self.log_path}/{picklef}.pth')
        print(f'{self.args.dataset} training done')
    
    def __call__(self):
        self.run()