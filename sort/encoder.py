from torch import nn


class Encoder(nn.Module):
    def __init__(self) -> None:
        super(Encoder, self).__init__()
        def block_(ch_in, ch_out):
            return nn.Sequential(
                nn.Conv3d(ch_in, ch_out, (3,3,3), stride=(1,1,1), padding=(1,1,1)),
                nn.BatchNorm3d(ch_out),
                nn.ReLU(),
                nn.MaxPool3d(kernel_size=(1,2,2))
            )
        def block(ch_in, ch_out):
            return nn.Sequential(
                nn.Conv3d(ch_in, ch_out, (3,3,3), stride=(1,1,1), padding=(1,1,1)),
                nn.BatchNorm3d(ch_out),
                nn.ReLU(),
                nn.MaxPool3d(kernel_size=(2,2,2))
            )
        
        self.pool1 = block_(1, 64)
        self.pool2 = block(64, 128)
        self.pool3 = block(128, 256)

    def forward(self, x):
        p1 = self.pool1(x)
        p2 = self.pool2(p1)
        z = self.pool3(p2)

        return z # [8, 256, 3, 32, 32]