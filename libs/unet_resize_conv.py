import libs.layers as layers
import torch
import torch.nn as nn
import torchvision.transforms.functional as V


class UNet(layers.BaseModel):
    class Doubleconv(layers.BaseModel):
        def __init__(self, inchans, chans, outchans, padding=True):
            super().__init__()
            self.conv = nn.Sequential(
                # Reduce size by 2 each Doubleconv block
                nn.Conv2d(inchans, chans, 3, padding=1 if padding else 0, padding_mode='reflect', bias=False),
                nn.BatchNorm2d(chans),
                nn.ReLU(inplace=True),
                nn.Conv2d(chans, outchans, 3, padding=1, padding_mode='reflect', bias=False),
                nn.BatchNorm2d(outchans),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.conv(x)

    class Layer(layers.BaseModel):
        def __init__(self, inchans, chans, outchans, inner, padding=True):
            super().__init__()
            self.down = UNet.Doubleconv(inchans, chans, chans, padding=padding)
            self.inner = nn.Sequential(
                nn.MaxPool2d(2),
                inner,
                nn.UpsamplingNearest2d(scale_factor=2)
            )
            self.up = UNet.Doubleconv(2 * chans, chans, outchans, padding=padding)  # Skip connection is concatenated to channels

        def forward(self, x):
            x = self.down(x)
            skip = x
            x = self.inner(x)
            skip = V.center_crop(skip, x.shape[2:])
            x = torch.cat((skip, x), dim=1)
            return self.up(x)

    def __init__(self, inchans=1, outchans=1, features=[64, 128, 256, 512], activation=nn.Tanh(), padding=True):
        super().__init__()
        self.noise = layers.GaussianNoise()
        self.final = nn.Sequential(nn.Conv2d(features[0], outchans, 1 if padding else 3), activation)
        self.inner = UNet.Doubleconv(features[-2], features[-1], features[-2], padding=padding)
        features = list(reversed(features[:-1]))

        for i, ch in enumerate(features):
            upmost = i == len(features)-1

            self.inner = UNet.Layer(
                inchans if upmost else features[i+1],
                ch,
                # Upmost layer has extra 1x1 convolution at the end, therefore outch = ch.
                ch if upmost else features[i+1],
                self.inner,
                padding=padding
            )

    def forward(self, x):
        x = self.inner(x)
        return self.final(x)
