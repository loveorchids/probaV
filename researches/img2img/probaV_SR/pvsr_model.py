import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, vgg16_bn
import omni_torch.networks.blocks as block


class Vgg16BN(nn.Module):
    def __init__(self):
        super(Vgg16BN, self).__init__()
        vgg16 = vgg16_bn(pretrained=True)
        net = list(vgg16.children())[0]
        self.conv_block1 = nn.Sequential(*net[:7])
        self.conv_block1.required_grad = False
        self.conv_block2 = nn.Sequential(*net[7:14])
        self.conv_block2.required_grad = False
        self.conv_block3 = nn.Sequential(*net[14:24])
        self.conv_block3.required_grad = False
        self.conv_block4 = nn.Sequential(*net[24:34])
        self.conv_block4.required_grad = False
        self.conv_block5 = nn.Sequential(*net[34:])
        self.conv_block5.required_grad = False
    
    def forward(self, x):
        def gram_matrix(x):
            nelement = x.size(0) * x.size(2) * x.size(3)
            return torch.mm(x.view(x.size(1), -1), torch.transpose(x.view(x.size(1), -1), 1, 0)) / nelement
        
        # assert len(layers) == len(keys)
        # In this scenario, input x is a grayscale image
        x = x.repeat(1, 3, 1, 1)
        out1 = self.conv_block1(x)
        out2 = self.conv_block2(out1)
        out3 = self.conv_block3(out2)
        # out4 = self.conv_block4(out3)
        # out5 = self.conv_block5(out4)
        return [out1, out2, out3]  # , out4, out5
        # return out1, gram_matrix(out2), gram_matrix(out3)


class ProbaV_basic(nn.Module):
    def __init__(self, inchannel=3, BN=nn.BatchNorm2d, group=1, s_MSE=False):
        super(ProbaV_basic, self).__init__()
        if s_MSE:
            self.evaluator = Vgg16BN()
        else:
            self.evaluator = None
        self.down_conv1 = block.conv_block(inchannel, [48 * group, 128 * group, 128 * group], kernel_sizes=[3, 3, 1],
                                           stride=[1, 1, 1], padding=[1, 1, 0], groups=[group] * 3,
                                           name="down_block1", batch_norm=BN)
        self.down_conv2 = block.conv_block(128 * group, [256 * group, 256 * group, 256 * group], kernel_sizes=[3, 3, 1],
                                           stride=[2, 1, 1], padding=[1, 1, 0], groups=[group] * 3, name="down_block2",
                                           batch_norm=BN)
        self.norm_conv = block.conv_block(256 * group, [512, 512, 512],
                                          kernel_sizes=[3, 3, 1], stride=[1] * 3, padding=[1, 1, 0],
                                          groups=[1] * 3, name="norm_conv", batch_norm=BN)
        self.up_conv1 = block.conv_block(512, [512, 512, 256], kernel_sizes=[5, 3, 3], stride=[3, 1, 1],
                                         padding=[1, 1, 1], groups=[1] * 3, name="up_block1", batch_norm=BN,
                                         transpose=[True, False, False])
        self.up_conv2 = block.conv_block(256, [256, 128, 128], kernel_sizes=[4, 3, 1], stride=[2, 1, 1],
                                         padding=[1, 1, 0], groups=[1] * 3, name="up_block2", batch_norm=BN,
                                         transpose=[True, False, False])
        self.up_conv3 = block.conv_block(128, [128, 48, 24, 1], kernel_sizes=[3, 3, 3, 3], stride=[1, 1, 1, 1],
                                         padding=[1] * 4, groups=[1] * 4, name="up_block3", batch_norm=BN)
    
    def forward(self, x, y=None):
        out = self.down_conv1(x)
        out = self.down_conv2(out)
        #out = self.down_conv3(out)
        # out = self.down_conv4(out)
        # out = self.down_conv5(out)
        out = self.norm_conv(out)
        out = self.up_conv1(out)
        out = self.up_conv2(out)
        # out = self.up_conv3_sig(out)
        out = self.up_conv3(out)
        # out = self.sigmoid(out)
        # out = self.up_conv5(out)
        if self.evaluator:
            s_mse_pred = self.evaluator(out)
            s_mse_label = self.evaluator(y)
            return [out] + s_mse_pred, [y] + s_mse_label
        else:
            return out, y

class ProbaV_SRNTT(nn.Module):
    def __init__(self):
        pass


if __name__ == "__main__":
    x = torch.randn(2, 10, 128, 128)
    net = ProbaV_basic(inchannel=10)
    y = net(x)
    print(y.shape)