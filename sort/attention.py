import torch
from torch import nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, ch_in, ch_out) -> None:
        super(Attention, self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        
        self.theta = nn.Conv3d(ch_in, ch_out, kernel_size=1)
        
        self.phi = nn.Conv3d(ch_in, ch_out, kernel_size=1)
        self.psi = nn.Conv3d(ch_in, ch_out, kernel_size=1)
        
        self.v = nn.Conv3d(ch_out, ch_in, kernel_size=1)
        self.relu = nn.ReLU()
        
    def forward(self, z):
        batch_size = z.size(0)
        
        theta_z = self.theta(z).view(batch_size, self.ch_out, -1)
        theta_z = theta_z.permute(0, 2, 1)
        
        psi_z = self.psi(z).view(batch_size, self.ch_out, -1)
        phi_z = self.phi(z).view(batch_size, self.ch_out, -1)
        phi_z = phi_z.permute(0, 2, 1)
        
        g = torch.matmul(phi_z, psi_z)
        N = g.size(-1)
        norm_g = self.relu(g/N)
        
        tmp_g = torch.matmul(norm_g, theta_z)
        tmp_g = tmp_g.permute(0, 2, 1).contiguous()
        tmp_g = tmp_g.view(batch_size, self.ch_out, *z.size()[2:])
        
        vg = self.v(tmp_g)
        z = vg + z
        
        return z


