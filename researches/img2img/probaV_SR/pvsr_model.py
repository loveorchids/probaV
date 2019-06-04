import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, vgg16_bn
import omni_torch.networks.blocks as block
import researches.img2img.probaV_SR.pvsr_module as module


class Block(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 group=1):
        super(Block, self).__init__()
        
        self.b1 = module.ResidualBlock(64, 64)
        self.b2 = module.ResidualBlock(64, 64)
        self.b3 = module.ResidualBlock(64, 64)
        self.c1 = module.BasicBlock(64 * 2, 64, 1, 1, 0)
        self.c2 = module.BasicBlock(64 * 3, 64, 1, 1, 0)
        self.c3 = module.BasicBlock(64 * 4, 64, 1, 1, 0)
    
    def forward(self, x):
        c0 = o0 = x
        
        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)
        
        b2 = self.b2(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)
        
        b3 = self.b3(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)
        
        return o3


class CARN(nn.Module):
    def __init__(self, inchannel, filters, scale, BN=nn.BatchNorm2d, s_MSE=False):
        super(CARN, self).__init__()
        self.scale = scale
        if s_MSE:
            self.evaluator = Vgg16BN()
        else:
            self.evaluator = None
        
        self.sub_mean = module.MeanShift((0.4488, 0.4371, 0.4040), sub=True)
        self.add_mean = module.MeanShift((0.4488, 0.4371, 0.4040), sub=False)
        
        self.entry = nn.Conv2d(inchannel, filters, 3, 1, 1)
        
        self.b1 = Block(filters, filters)
        self.b2 = Block(filters, filters)
        self.b3 = Block(filters, filters)
        self.c1 = module.BasicBlock(filters * 2, filters, 1, 1, 0)
        self.c2 = module.BasicBlock(filters * 3, filters, 1, 1, 0)
        self.c3 = module.BasicBlock(filters * 4, filters, 1, 1, 0)
        
        self.upsample = module.UpsampleBlock(filters, scale=scale)
        self.exit = nn.Conv2d(filters, 1, 3, 1, 1)
        self.mae = nn.L1Loss()
        self.s_mse_loss = nn.MSELoss()
    
    def forward(self, x, y):
        #x = self.sub_mean(x)
        x = self.entry(x)
        c0 = o0 = x
        
        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)
        
        b2 = self.b2(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)
        
        b3 = self.b3(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)
        
        out = self.upsample(o3)
        
        out = self.exit(out)
        #out = self.add_mean(out)
        mae = self.mae(out, y).unsqueeze_(0)
        if self.evaluator:
            s_mse_pred = self.evaluator(out)
            s_mse_label = self.evaluator(y)
            s_mse_loss = sum([self.s_mse_loss(s_pred, s_mse_label[i]) for i, s_pred in enumerate(s_mse_pred)])
            return out, mae, s_mse_loss.unsqueeze_(0)
        else:
            return out, mae, torch.tensor([0])


class RDN(nn.Module):
    def __init__(self, channel, rdb_number, upscale_factor, BN=nn.BatchNorm2d, s_MSE=False, filters=64):
        super(RDN, self).__init__()
        if s_MSE:
            self.evaluator = Vgg16BN()
        else:
            self.evaluator = None
        self.SFF1 = nn.Conv2d(in_channels=channel, out_channels=filters, kernel_size=3, padding=1, stride=1)
        self.SFF2 = nn.Conv2d(in_channels=filters, out_channels=filters, kernel_size=3, padding=1, stride=1)
        self.RDB1 = RDB(nb_layers=rdb_number, input_dim=filters, growth_rate=filters)
        self.RDB2 = RDB(nb_layers=rdb_number, input_dim=filters, growth_rate=filters)
        self.RDB3 = RDB(nb_layers=rdb_number, input_dim=filters, growth_rate=filters)
        self.GFF1 = nn.Conv2d(in_channels=filters * 3, out_channels=filters, kernel_size=1, padding=0)
        self.GFF2 = nn.Conv2d(in_channels=filters, out_channels=filters, kernel_size=3, padding=1)
        self.upconv = nn.Conv2d(in_channels=filters, out_channels=(filters * upscale_factor * upscale_factor),
                                kernel_size=3, padding=1)
        self.pixelshuffle = nn.PixelShuffle(upscale_factor)
        #self.conv2 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, padding=1)
        self.trellis = module.Trellis_Structure(filters=filters)
        """
        self.norm_conv1 = block.conv_block(128, [128, 128, 64], kernel_sizes=[3, 1, 3], stride=[1, 1, 1],
                                           padding=[1, 0, 1], groups=[1] * 3, name="norm_conv1", batch_norm=BN,
                                           activation=None)
        self.norm_conv2 = block.conv_block(64, [64, 64, 32], kernel_sizes=[3, 1, 3], stride=[1, 1, 1],
                                           padding=[1, 0, 1], groups=[1] * 3, name="norm_conv2", batch_norm=BN,
                                           activation=None)
        self.norm_conv3 = block.conv_block(32, [32, 16, 16, 1], kernel_sizes=[1, 3, 3, 3], stride=[1, 1, 1, 1],
                                           padding=[0, 1, 1, 1], groups=[1] * 4, name="norm_conv3", batch_norm=BN,
                                           activation=None)
                                           """
        self.mae = nn.L1Loss()
        self.s_mse_loss = nn.MSELoss()
        
    
    def forward(self, x, y):
        f_ = self.SFF1(x)
        f_0 = self.SFF2(f_)
        f_1 = self.RDB1(f_0)
        f_2 = self.RDB2(f_1)
        f_3 = self.RDB3(f_2)
        f_D = torch.cat((f_1, f_2, f_3), 1)
        f_1x1 = self.GFF1(f_D)
        f_GF = self.GFF2(f_1x1)
        f_DF = f_GF + f_
        f_upconv = self.upconv(f_DF)
        f_upscale = self.pixelshuffle(f_upconv)
        # f_conv2 = self.conv2(f_upscale)
        results = self.trellis(f_upscale)
        out = results[-1]
        mae = sum([self.mae(result, y) for result in results]).unsqueeze_(0)
        """
        out = self.norm_conv1(f_upscale)
        out = self.norm_conv2(out)
        out = self.norm_conv3(out)
        mae = self.mae(out, y).unsqueeze_(0)
        """
        if self.evaluator:
            s_mse_pred = self.evaluator(out)
            s_mse_label = self.evaluator(y)
            s_mse_loss = sum([self.s_mse_loss(s_pred, s_mse_label[i]) for i, s_pred in enumerate(s_mse_pred)])
            return out, mae, s_mse_loss.unsqueeze_(0)
        else:
            return out, mae, torch.tensor([0])
        

class BasicBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BasicBlock, self).__init__()
        self.ID = input_dim
        self.conv = nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, padding=1, stride=1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        return torch.cat((x, out), 1)


class RDB(nn.Module):
    def __init__(self, nb_layers, input_dim, growth_rate):
        super(RDB, self).__init__()
        self.ID = input_dim
        self.GR = growth_rate
        self.layer = self._make_layer(nb_layers, input_dim, growth_rate)
        self.conv1x1 = nn.Conv2d(in_channels=input_dim + nb_layers * growth_rate, \
                                 out_channels=growth_rate, \
                                 kernel_size=1, \
                                 stride=1, \
                                 padding=0)
    
    def _make_layer(self, nb_layers, input_dim, growth_rate):
        layers = []
        for i in range(nb_layers):
            layers.append(BasicBlock(input_dim + i * growth_rate, growth_rate))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.layer(x)
        out = self.conv1x1(out)
        return out + x


class Vgg16BN(nn.Module):
    def __init__(self):
        super(Vgg16BN, self).__init__()
        vgg16 = vgg16_bn(pretrained=True)
        net = list(vgg16.children())[0]
        self.conv_block1 = nn.Sequential(*net[:7])
        self.conv_block1.required_grad = False
        self.conv_block2 = nn.Sequential(*net[7:14])
        self.conv_block2.required_grad = False
        """
        self.conv_block3 = nn.Sequential(*net[14:24])
        self.conv_block3.required_grad = False
        self.conv_block4 = nn.Sequential(*net[24:34])
        self.conv_block4.required_grad = False
        self.conv_block5 = nn.Sequential(*net[34:])
        self.conv_block5.required_grad = False
        """
    
    def forward(self, x):
        def gram_matrix(x):
            nelement = x.size(0) * x.size(2) * x.size(3)
            return torch.mm(x.view(x.size(1), -1), torch.transpose(x.view(x.size(1), -1), 1, 0)) / nelement
        
        # assert len(layers) == len(keys)
        # In this scenario, input x is a grayscale image
        x = x.repeat(1, 3, 1, 1)
        out1 = self.conv_block1(x)
        out2 = self.conv_block2(out1)
        #out3 = self.conv_block3(out2)
        # out4 = self.conv_block4(out3)
        # out5 = self.conv_block5(out4)
        return [out1, out2]#, out3]  # , out4, out5
        # return out1, gram_matrix(out2), gram_matrix(out3)


class ProbaV_basic(nn.Module):
    def __init__(self, inchannel=3, BN=nn.BatchNorm2d, group=1, s_MSE=False, SA=True):
        super(ProbaV_basic, self).__init__()
        if s_MSE:
            self.evaluator = Vgg16BN()
        else:
            self.evaluator = None
        self.SA = SA
        self.down_conv1 = block.conv_block(inchannel, [48 * group, 128 * group, 128 * group], kernel_sizes=[3, 3, 1],
                                           stride=[2, 1, 1], padding=[1, 1, 0], groups=[group] * 3,
                                           name="down_block1", batch_norm=BN)
        self.down_conv2 = block.conv_block(128 * group, [256 * group, 256 * group, 256 * group], kernel_sizes=[3, 3, 1],
                                           stride=[2, 1, 1], padding=[1, 1, 0], groups=[group] * 3, name="down_block2",
                                           batch_norm=BN)
        self.norm_conv1 = block.conv_block(256 * group, [256, 256, 256],
                                          kernel_sizes=[3, 3, 3], stride=[1] * 3, padding=[2, 1, 1],
                                          groups=[1] * 3, dilation=[2, 1, 1], name="norm_conv1", batch_norm=BN)
        if SA:
            self.self_attn = Self_Attn(256)
        
        self.up_conv1 = block.conv_block(256, [256, 256, 256], kernel_sizes=[5, 3, 3], stride=[3, 1, 1],
                                         padding=[1, 1, 1], groups=[1] * 3, name="up_block1", batch_norm=BN,
                                         transpose=[True, False, False])
        self.norm_conv2 = block.conv_block(256, [256, 256, 256], kernel_sizes=[1, 3, 3], stride=[1, 1, 1],
                                         padding=[0, 1, 1], groups=[1] * 3, name="norm_conv2", batch_norm=BN)
        self.up_conv2 = block.conv_block(256, [256, 256, 256], kernel_sizes=[4, 3, 3], stride=[2, 1, 1],
                                         padding=[1, 1, 1], groups=[1] * 3, name="up_block2", batch_norm=BN,
                                         transpose=[True, False, False])
        self.norm_conv3 = block.conv_block(256, [256, 128, 128], kernel_sizes=[1, 3, 3], stride=[1, 1, 1],
                                           padding=[0, 1, 1], groups=[1] * 3, name="norm_conv3", batch_norm=BN)
        self.up_conv3 = block.conv_block(128, [128, 128, 128], kernel_sizes=[4, 3, 3], stride=[2, 1, 1],
                                         padding=[1, 1, 1], groups=[1] * 3, name="up_block3", batch_norm=BN,
                                         transpose=[True, False, False])
        self.norm_conv4 = block.conv_block(128, [128, 64, 64], kernel_sizes=[1, 3, 3], stride=[1, 1, 1],
                                           padding=[0, 1, 1], groups=[1] * 3, name="norm_conv4", batch_norm=BN,
                                           activation=None)
        self.norm_conv5 = block.conv_block(64, [64, 32, 32], kernel_sizes=[1, 3, 3], stride=[1, 1, 1],
                                           padding=[0, 1, 1], groups=[1] * 3, name="norm_conv5", batch_norm=BN,
                                           activation=None)
        self.norm_conv6 = block.conv_block(32, [32, 16, 16, 1], kernel_sizes=[1, 3, 3, 3], stride=[1, 1, 1, 1],
                                           padding=[0, 1, 1, 1], groups=[1] * 4, name="norm_conv6", batch_norm=BN,
                                           activation=None)

    
    def forward(self, x, y=None):
        out = self.down_conv1(x)
        out = self.down_conv2(out)
        #out = self.down_conv3(out)
        # out = self.down_conv4(out)
        # out = self.down_conv5(out)
        out = self.norm_conv1(out)
        if self.SA:
            out, attn_map = self.self_attn(out)
        out = self.up_conv1(out)
        out = self.norm_conv2(out)
        out = self.up_conv2(out)
        out = self.norm_conv3(out)
        out = self.up_conv3(out)
        out = self.norm_conv4(out)
        out = self.norm_conv5(out)
        out = self.norm_conv6(out)
        #out = out / torch.max(out)
        # out = self.sigmoid(out)
        # out = self.up_conv5(out)
        if self.evaluator:
            s_mse_pred = self.evaluator(out)
            s_mse_label = self.evaluator(y)
            return [out] + s_mse_pred, [y] + s_mse_label
        else:
            return out, y


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super().__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out, attention



if __name__ == "__main__":
    x = torch.randn(3, 10, 128, 128)
    gt = torch.randn(3, 64, 384, 384)
    
    #net = ProbaV_basic(inchannel=10)
    net = module.Trellis_Structure()
    #net = CARN(10, 64, 3)
    y = net(gt)
    print(y.shape)