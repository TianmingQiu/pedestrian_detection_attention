import torch
import torch.nn as nn
import torch.nn.functional as F
from .Incep import Inception3, InceptionA, InceptionB, BasicConv2d


class AF1(nn.Module):
    def __init__(self, num_classes=26, aux_logits=False, transform_input=False, ret=False):  # ccc changed here
        super(AF1, self).__init__()
        self.aux_logits = aux_logits
        self.transform_input = transform_input
        self.MNet = Incep.Inception3(ret=True)

        self.Att = BasicConv2d(288, 8, kernel_size=1)
        self.Incep2 = nn.Sequential(
            InceptionB(288),
            InceptionC(768, channels_7x7=128),
            InceptionC(768, channels_7x7=160),
            InceptionC(768, channels_7x7=160),
            InceptionC(768, channels_7x7=192)
        )
  
        if aux_logits:
            self.AuxLogits = InceptionAux(768, num_classes)

        self.Incep3 = nn.Sequential(
            InceptionD(768),
            InceptionE(1280),
            InceptionE(2048)
        )

        self.Incep3_2 = nn.Sequential(
            InceptionD(768),
            InceptionE(1280),
            InceptionE(2048)
        )

        self.fc = nn.Linear(2048 * 24, num_classes)

        self.ret = ret

        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                
                stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                X = stats.truncnorm(-2, 2, scale=stddev)
                values = torch.Tensor(X.rvs(m.weight.data.numel()))
                values = values.view(m.weight.data.size())
                m.weight.data.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        """

    def forward(self, x):
        if self.transform_input:
            x = x.clone()
            x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        # 299 x 299 x 3
        F1, F2, F3 = self.MNet(x)  # three return: x, y, z
        # F1 35 x 35 x 288
        # F2 17 x 17 x 768
        # F3  8 x 8 x 2048

        attentive = self.Att(F1)
        # 35 x 35 x 8
        '''
        temp = torch.sum(attentive,dim=1)
        newimg = misc.imresize(temp.data[0].cpu().numpy(),(299,299))
        plt.imshow(newimg,cmap='jet')
        plt.savefig("af1")  # draw heat map
        '''

        ret = 0
        for i in range(8):
            # print(F1.size())  # N * c * h * w
            temp = attentive[:, i].clone()
            temp = temp.view(-1, 1, 35, 35).expand(-1, 288, 35, 35)
            R1 = F1 * temp
            R1 = self.Incep2(R1)
            # 17 x 17 x 768
            R1 = self.Incep3(R1)
            # 8 x 8 x 2048
            if i == 0:
                ret = R1
            else:
                # print(type(ret),type(R1))
                ret = torch.cat((ret, R1), dim=1)
        # ret 8 x 8 x (2048 x 8)

        attentive2 = F.avg_pool2d(attentive, kernel_size=2, stride=2)

        for i in range(8):
            temp = attentive2[:, i].clone()
            temp = temp.view(-1, 1, 17, 17).expand(-1, 768, 17, 17)
            R2 = F2 * temp
            R2 = self.Incep3_2(R2)
            # 8 x 8 x 2048
            ret = torch.cat((ret, R2), dim=1)
        # ret 8 x 8 x (2048 x 16)

        attentive3 = F.avg_pool2d(attentive, kernel_size=4, stride=4)
        for i in range(8):
            temp = attentive3[:, i].clone()
            temp = temp.view(-1, 1, 8, 8).expand(-1, 2048, 8, 8)
            R3 = F3 * temp
            ret = torch.cat((ret, R3), dim=1)

        if self.ret:
            return ret, attentive
        # ret 8 x 8 x(2048 x 24)
        ret = F.avg_pool2d(ret, kernel_size=8)

        # 1 x 1 x (2048 x 24)
        ret = F.dropout(ret, training=self.training)
        # 1 x 1 x (2048 x 24)
        ret = ret.view(ret.size(0), -1)
        # 2048 x 24

        ret = self.fc(ret)
        # 1000 (num_classes)
        print(ret.size(), attentive.size())

        return ret
