import torch
from torch.utils import data
from torchvision import transforms

from glob import glob
from PIL import Image
import os


class Cliploader(data.Dataset):
    ''' Load a dataset using the silding window '''
    def __init__(self, input_path, transform, window) -> None:
        super(Cliploader, self).__init__()
        
        self.window = window
        self.transform = transform
        self.clips = self.sliding_window(input_path)
    
    # get windows from the entire frame set
    def sliding_window(self, data_path):
        videos = sorted(glob(os.path.join(data_path, '*')))
        
        entry1 = []
        for vid in videos:
            samples = sorted(glob(os.path.join(vid, '*')))
            entry1.append(samples)
        
        entry2 = []
        for vid in entry1:
            for i in range(len(vid)-(self.window)):
                clip = vid[i:i+self.window]
                entry2.append(clip)
                
        return entry2
    
    # concat a clip
    def clipper(self, input):
        stack = []
        for i in input:
            x = self.transform(Image.open(i)) # [C, H, W]
            # x = torch.squeeze(x) # [H, W]
            stack.append(x)
            
        cat = torch.stack(stack, axis=0) # [window, H, W]
        if len(cat.shape) == 4:
            cat = cat.view(-1, 256, 256)
        
        out = torch.unsqueeze(cat, dim=0) # [1, window, H, W]    
        
        return out

    def __getitem__(self, index):
        frames = self.clips[index]
        frames = self.clipper(frames)
        
        return frames

    def __len__(self):
        return len(self.clips)


class Testloader(Cliploader):
    def __init__(self, input_path, transform, window) -> None:
        super().__init__(input_path, transform, window)
    
    # override -> get windows from one video
    def sliding_window(self, data_path):
        frames = sorted(glob(os.path.join(data_path, '*')))
        
        entry = []
        for i in range(len(frames)-(self.window)):
            clip = frames[i:i+self.window+1]
            entry.append(clip)
            
        return entry


def trainloader(frame_path, batch, num_workers, window):
    trans_frame = [
        transforms.Grayscale(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ]
    dataset = Cliploader(frame_path,
                         transforms.Compose(trans_frame),
                         window)
    
    return data.DataLoader(dataset,
                           batch_size=batch,
                           shuffle=True,
                           num_workers=num_workers,
                           drop_last=True,
                           pin_memory=False)


def testloader(frame_path, num_workers, window):
    trans_frame = [
        transforms.Grayscale(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ]
    dataset = Testloader(frame_path,
                         transforms.Compose(trans_frame),
                         window)
    
    return data.DataLoader(dataset,
                           batch_size=1,
                           shuffle=False,
                           num_workers=num_workers,
                           drop_last=False,
                           pin_memory=False)