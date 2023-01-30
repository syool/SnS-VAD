from torch import nn


class Decoder(nn.Module):
    def __init__(self) -> None:
        super(Decoder, self).__init__()
        def block(ch_in, ch_out):
            return nn.Sequential(
                nn.ConvTranspose3d(ch_in, ch_out, (3,3,3), stride=(2,2,2),
                                   padding=(1,1,1), output_padding=(1,1,1)),
                nn.BatchNorm3d(ch_out),
                nn.ReLU()
            )
        def block_(ch_in, ch_out):
            return nn.Sequential(
                nn.ConvTranspose3d(ch_in, ch_out, (3,3,3), stride=(1,2,2),
                                   padding=(1,1,1), output_padding=(0,1,1)),
                nn.Tanh()
            )
        
        self.ups1 = block(256, 128)
        self.ups2 = block(128, 64)
        self.ups3 = block_(64, 1)

    def forward(self, z):
        up1 = self.ups1(z)
        up2 = self.ups2(up1)
        x = self.ups3(up2)

        return x