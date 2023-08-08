import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module): #noise 1000
    def __init__(self, nz=1000, ngf=64, nc=3, img_size=32):
        super(Generator, self).__init__()

        self.init_size = img_size // 4

        fc_nodes = [nz , ngf * 2 *self.init_size**2]
        self.l1 = nn.Linear(fc_nodes[0] , fc_nodes[1] , bias=True)

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 2, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf * 2, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, noises):
        out = self.l1(noises)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks2(img)
        return img
