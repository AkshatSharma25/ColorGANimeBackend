import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import os
from torch import autograd
from PIL import Image, ImageEnhance

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 100
BATCH_SIZE = 8
# LR = 2e-4
LR_G=2e-4
LR_D=2e-4
LAMBDA_L1 = 20
curr_epoch=0
lambda_gp = 10  

import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        def down(in_c, out_c, norm=True):
            layers = [nn.Conv2d(in_c, out_c, 4, 2, 1)]
            if norm: layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2))
            return layers

        def up(in_c, out_c, dropout=False):
            layers = [nn.ConvTranspose2d(in_c, out_c, 4, 2, 1),
                      nn.BatchNorm2d(out_c),
                      nn.ReLU()]
            if dropout: layers.append(nn.Dropout(0.5))
            return layers

        def final(in_c,out_c):
            layers=[nn.Conv2d(in_c,out_c,1,1,0),nn.Tanh()]
            return layers
        
        
        # Encoder (downsampling)
        self.down1 = nn.Sequential(*down(1, 64, norm=False))
        self.down2 = nn.Sequential(*down(64, 128))
        self.down3 = nn.Sequential(*down(128, 256))
        self.down4 = nn.Sequential(*down(256, 512))

        #bottleneck:
        self.down5=nn.Sequential(*down(512,512))


        # Decoder (upsampling with skip connections)
        # Only skip with tensors that have the same spatial size
        self.up1 = nn.Sequential(*up(512,512, dropout=True))
        self.up2 = nn.Sequential(*up(768, 256, dropout=True))  # 512 (up1) + 512 (d5)
        self.up3 = nn.Sequential(*up(384, 128, dropout=True))  # 512 (up2) + 512 (d4)
        self.up4 = nn.Sequential(*up(192, 64))                 # No skip here, just upsample
        # self.up5 = nn.Sequential(*up(256, 128))                 # No skip here, just upsample
        # self.up6 = nn.Sequential(*up(128, 64))                  # No skip here, just upsample

        # Final layer: 64 -> 3 (RGB)
        self.output = nn.Sequential(*final(64, 3))

    def forward(self, x):
        # Encoder
        d1 = self.down1(x)      # [B, 64, 64, 64]
        d2 = self.down2(d1)     # [B, 128, 32, 32]
        d3 = self.down3(d2)     # [B, 256, 16, 16]
        d4 = self.down4(d3)    # [B, 512, 2, 2]
        # d5=self.down5(d4)
        # Decoder with skip connections (only skip with matching spatial sizes)
        u1 = self.up1(d4)
        # print(u1.shape,d3.shape)
                                       # [B, 512, 4, 4]
        u2 = self.up2(torch.cat([u1, d3],dim=1))      
        # print(d2.shape,u2.shape)
        u3 = self.up3(torch.cat([u2, d2],dim=1))      
        u4 = self.up4(torch.cat([u3,d1],dim=1))                               # [B, 256, 32, 32]
        # u5 = self.up5(u4)                               # [B, 128, 64, 64]
        # u6 = self.up6(u5)                               # [B, 64, 128, 128]

        # Final output: map to 3 channels (RGB)
        out = self.output(u4)    # [B, 64, 128, 128] -> [B, 3, 128, 128]
        return out


gen = Generator().to(DEVICE)
opt_gen = optim.Adam(gen.parameters(), lr=LR_G, betas=(0.5, 0.999))







