import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import omni_torch.networks.blocks as block


class Trellis_Structure(nn.Module):
    def __init__(self, depth=4, filters=64, activation=nn.CELU()):
        super().__init__()
        self.depth = depth
        self.total_blocks = sum(range(depth + 1))
        self.trellis_blocks = nn.ModuleList()
        for i in range(depth, 0, -1):
            if i == depth:
                in_channel = filters
                final_inchannel = filters
            else:
                in_channel = filters * 2
                final_inchannel = filters + 1
            sub_module = Trellis_Submodule(in_channel, filters, final_inchannel, activation, depth=i)
            self.trellis_blocks.append(sub_module)
        
    def forward(self, x):
        result = []
        for i, block in enumerate(self.trellis_blocks):
            if i == 0:
                x = block(x, first_line=True)
                result.append(x[-1])
            else:
                x = block(x)
                result.append(x[-1])
        return result
        

class Trellis_Submodule(nn.Module):
    def __init__(self, in_channel, filters, final_inchannel, activation, depth):
        super().__init__()
        self.sub_module = nn.ModuleList([])
        for _ in range(depth):
            if _ == depth - 1:
                filter = 1
                in_channel = final_inchannel
            else:
                filter = filters
            self.sub_module.append(
                block.conv_block(in_channel, filters=filter, kernel_sizes=3, stride=1,
                                 padding=1, activation=activation)
            )
            
    def forward(self, input, first_line=False):
        result = []
        if first_line:
            for module in self.sub_module:
                input = module(input)
                result.append(input)
        else:
            assert len(input) == len(self.sub_module) + 1
            for i, module in enumerate(self.sub_module):
                _input = torch.cat((input[i], input[i + 1]), dim=1)
                result.append(module(_input))
        return result
        

def init_weights(modules):
    pass


class MeanShift(nn.Module):
    def __init__(self, mean_rgb, sub):
        super(MeanShift, self).__init__()
        
        sign = -1 if sub else 1
        r = mean_rgb[0] * sign
        g = mean_rgb[1] * sign
        b = mean_rgb[2] * sign
        
        self.shifter = nn.Conv2d(3, 3, 1, 1, 0)
        self.shifter.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.shifter.bias.data = torch.Tensor([r, g, b])
        
        # Freeze the mean shift layer
        for params in self.shifter.parameters():
            params.requires_grad = False
    
    def forward(self, x):
        x = self.shifter(x)
        return x


class BasicBlock(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 ksize=3, stride=1, pad=1):
        super(BasicBlock, self).__init__()
        
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, pad),
            nn.ReLU(inplace=True)
        )
        
        init_weights(self.modules)
    
    def forward(self, x):
        out = self.body(x)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        )
        
        init_weights(self.modules)
    
    def forward(self, x):
        out = self.body(x)
        out = F.relu(out + x)
        return out


class EResidualBlock(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 group=1):
        super(EResidualBlock, self).__init__()
        
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, groups=group),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=group),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, 1, 0),
        )
        
        init_weights(self.modules)
    
    def forward(self, x):
        out = self.body(x)
        out = F.relu(out + x)
        return out


class UpsampleBlock(nn.Module):
    def __init__(self, n_channels, scale, multi_scale=None, group=1):
        super(UpsampleBlock, self).__init__()
        
        if multi_scale:
            self.up2 = _UpsampleBlock(n_channels, scale=2, group=group)
            self.up3 = _UpsampleBlock(n_channels, scale=3, group=group)
            self.up4 = _UpsampleBlock(n_channels, scale=4, group=group)
        else:
            self.up = _UpsampleBlock(n_channels, scale=scale, group=group)
        
        self.multi_scale = multi_scale
    
    def forward(self, x, scale=3):
        if self.multi_scale:
            if scale == 2:
                return self.up2(x)
            elif scale == 3:
                return self.up3(x)
            elif scale == 4:
                return self.up4(x)
        else:
            return self.up(x)


class _UpsampleBlock(nn.Module):
    def __init__(self,
                 n_channels, scale,
                 group=1):
        super(_UpsampleBlock, self).__init__()
        
        modules = []
        if scale == 2 or scale == 4 or scale == 8:
            for _ in range(int(math.log(scale, 2))):
                modules += [nn.Conv2d(n_channels, 4 * n_channels, 3, 1, 1, groups=group), nn.ReLU(inplace=True)]
                modules += [nn.PixelShuffle(2)]
        elif scale == 3:
            modules += [nn.Conv2d(n_channels, 9 * n_channels, 3, 1, 1, groups=group), nn.ReLU(inplace=True)]
            modules += [nn.PixelShuffle(3)]
        
        self.body = nn.Sequential(*modules)
        init_weights(self.modules)
    
    def forward(self, x):
        out = self.body(x)
        return out