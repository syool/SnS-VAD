from torch import nn

from .encoder import Encoder
from .decoder import Decoder
from .attention import Attention


class Sortnet(nn.Module):
    def __init__(self) -> None:
        super(Sortnet, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.att = Attention(256, 128)

    def forward(self, x):
        z = self.encoder(x)
        # z = self.att(z)
        x = self.decoder(z)

        return x