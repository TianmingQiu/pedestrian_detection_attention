import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


class MNet(nn.Module):
    def __init__(self, init_weights):
        super(MNet, self).__init__()
        self.Conv2d_1_7x7_s2 = BasicConv2d(3, 32, kernel_size=7, stride=2)
        self.Conv2d_2_1x1 = BasicConv2d(32, 32, kernel_size=1)
        self.Conv2d_3_3x3 = BasicConv2d(32, 96, kernel_size=3, padding=1)

        self.incept_block_1 = InceptBlock1()
        self.incept_block_2 = InceptBlock2()
        self.incept_block_3 = InceptBlock3()

        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    import scipy.stats as stats
                    stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                    X = stats.truncnorm(-2, 2, scale=stddev)
                    values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
                    values = values.view(m.weight.size())
                    with torch.no_grad():
                        m.weight.copy_(values)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.Conv2d_1_7x7_s2(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Conv2d_2_1x1(x)
        x = self.Conv2d_3_3x3(x)
        x0 = F.max_pool2d(x, kernel_size=3, stride=2)

        x = self.incept_block_1(x0)
        y = self.incept_block_2(x)
        z = self.incept_block_3(y)
        return x0, x, y, z

    def incepts_weight_save(self):
        torch.save(self.state_dict(), 'data/pretrained_model/hp_mnet.pth')
        torch.save(self.incept_block_1.state_dict(), 'data/pretrained_model/hp_mnet_incept1.pth')
        torch.save(self.incept_block_2.state_dict(), 'data/pretrained_model/hp_mnet_incept2.pth')
        torch.save(self.incept_block_3.state_dict(), 'data/pretrained_model/hp_mnet_incept3.pth')

    def init_w(self):
        pretrained_state_dict = torch.load("data/pretrained_model/hp_mnet.pth")
        self.load_state_dict(pretrained_state_dict)


class InceptionA(nn.Module):
    def __init__(
            self,
            in_channels,
            b_1x1_out,
            b_5x5_1_out,
            b_5x5_2_out,
            b_3x3_1_out,
            b_3x3_2_out,
            b_3x3_3_out,
            b_pool_out
    ):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, b_1x1_out, kernel_size=1)  # H_out = H_in

        self.branch5x5_1 = BasicConv2d(in_channels, b_5x5_1_out, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(b_5x5_1_out, b_5x5_2_out, kernel_size=3, padding=1)  # H_out = H_in

        self.branch3x3dbl_1 = BasicConv2d(in_channels, b_3x3_1_out, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(b_3x3_1_out, b_3x3_2_out, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(b_3x3_2_out, b_3x3_3_out, kernel_size=3, padding=1)

        self.branch_pool = BasicConv2d(in_channels, b_pool_out, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)        
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionB(nn.Module):
    def __init__(
            self,
            in_channels,
            b_1x1_1_out,
            b_1x1_2_out,
            b_3x3_1_out,
            b_3x3_2_out,
            b_3x3_3_out
    ):
        super(InceptionB, self).__init__()
        self.branch1x1_1 = BasicConv2d(in_channels, b_1x1_1_out, kernel_size=1)
        self.branch1x1_2 = BasicConv2d(b_1x1_1_out, b_1x1_2_out, kernel_size=3, stride=2, padding=1)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, b_3x3_1_out, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(b_3x3_1_out, b_3x3_2_out, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(b_3x3_2_out, b_3x3_3_out, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        branch3x3 = self.branch1x1_1(x)
        branch3x3 = self.branch1x1_2(branch3x3)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class InceptBlock1(nn.Module):
    def __init__(self):
        super(InceptBlock1, self).__init__()
        self.module_a = InceptionA(in_channels=96, b_1x1_out=32, b_5x5_1_out=32, b_5x5_2_out=32,
                                   b_3x3_1_out=32, b_3x3_2_out=48, b_3x3_3_out=48, b_pool_out=16)  # C_out =128
        self.module_b = InceptionB(in_channels=128, b_1x1_1_out=32, b_1x1_2_out=80,
                                   b_3x3_1_out=32, b_3x3_2_out=48, b_3x3_3_out=48)

    def forward(self, x):
        x = self.module_a(x)
        x = self.module_b(x)
        return x


class InceptBlock2(nn.Module):
    def __init__(self):
        super(InceptBlock2, self).__init__()
        self.module_a = InceptionA(in_channels=256, b_1x1_out=112, b_5x5_1_out=32, b_5x5_2_out=48,
                                   b_3x3_1_out=48, b_3x3_2_out=64, b_3x3_3_out=64, b_pool_out=64)  # C_out = 288
        self.module_b = InceptionB(in_channels=288, b_1x1_1_out=64, b_1x1_2_out=86,
                                   b_3x3_1_out=96, b_3x3_2_out=128, b_3x3_3_out=128)  # C_out = 502??!

    def forward(self, x):
        x = self.module_a(x)
        x = self.module_b(x)
        return x


class InceptBlock3(nn.Module):
    def __init__(self):
        super(InceptBlock3, self).__init__()
        self.module_a = InceptionA(in_channels=502, b_1x1_out=176, b_5x5_1_out=96, b_5x5_2_out=160,
                                   b_3x3_1_out=80, b_3x3_2_out=112, b_3x3_3_out=112, b_pool_out=64)
        self.module_b = InceptionA(in_channels=512, b_1x1_out=176, b_5x5_1_out=96, b_5x5_2_out=160,
                                   b_3x3_1_out=80, b_3x3_2_out=112, b_3x3_3_out=112, b_pool_out=64)

    def forward(self, x):
        x = self.module_a(x)
        x = self.module_b(x)
        return x