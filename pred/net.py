import imp
from torch import nn

from .encoder import Encoder
from .decoder import Decoder


class Prednet(nn.Module):
    def __init__(self, clip_length) -> None:
        super(Prednet, self).__init__()
        self.encoder = Encoder(clip_length)
        self.decoder = Decoder()

    def forward(self, x):
        z, skip = self.encoder(x)
        x = self.decoder(z, skip)

        return x