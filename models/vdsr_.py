import torch
import torch.nn as nn
from math import sqrt

class Block(nn.Module):
    def __init__(self):
        super(Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.conv(x))
    
class VDSR(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_blocks=18):
        super(VDSR, self).__init__()
        self.residual_layer = self.make_layer(Block, num_blocks)
        self.input = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.output = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                          
    def make_layer(self, block, num_layers):
        layers=[]
        for _ in range(num_layers):
            layers.append(block())
        return nn.Sequential(*layers)
    
    def forward(self, x):
        residual = x
        out = self.relu(self.input(x))
        out = self.residual_layer(out)
        out = self.output(out)
        out = torch.add(residual, out)
        return out