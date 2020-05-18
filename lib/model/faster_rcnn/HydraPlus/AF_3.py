import torch
import torch.nn as nn
import torch.nn.functional as F
# from .Incep import Inception3, InceptionA, InceptionB, BasicConv2d
from .Incep import *
import pdb


class AF3(nn.Module):
    def __init__(self):
        super(AF3, self).__init__()
        self.MNet = Inception3(False)  # todo: flag name should be changed
        for param in self.MNet.parameters():
            param.requires_grad = False

        self.att_channel_L = 1

        self.Att = BasicConv2d(512, self.att_channel_L, kernel_size=1)  # todo:size

        self.att_branch_1 = nn.Sequential(InceptBlock1(), InceptBlock2(), InceptBlock3())
        self.att_branch_2 = nn.Sequential(InceptBlock2(), InceptBlock3())
        self.att_branch_3 = InceptBlock3()

        self.patch = nn.ReflectionPad2d((0, 0, 0, -1))

        self.init_weight()

    def forward(self, x):
        feature_out0, feature_out1, feature_out2, feature_out3 = self.MNet(x)
        # feature_out0: torch.Size([batch_size, 96, 73, 132])
        # feature_out1: torch.Size([batch_size, 256, 37, 66])
        # feature_out2: torch.Size([batch_size, 502, 19, 33])
        # feature_out3: torch.Size([batch_size, 512, 19, 33])

        attentive = self.Att(feature_out3)

        attentive2 = self.patch(F.upsample(attentive, scale_factor=2))  # ToDo: make sure same size as f1
        attentive1 = self.patch(F.upsample(attentive2, scale_factor=2))  # ToDo: make sure same size as f0

        # attention branch 1
        for i in range(self.att_channel_L):
            temp = attentive1[:, i].clone()
            temp = temp.view(-1, 1, 73, 132).expand(-1, 96, 73, 132)
            att_feature_out0 = feature_out0 * temp
            att_feature_out3 = self.att_branch_1(att_feature_out0)
            if i == 0:
                ret = att_feature_out3
            else:
                ret = torch.cat((ret, att_feature_out3), dim=1)

        # attention branch 2
        for i in range(self.att_channel_L):
            temp = attentive2[:, i].clone()
            temp = temp.view(-1, 1, 37, 66).expand(-1, 256, 37, 66)
            att_feature_out1 = feature_out1 * temp
            att_feature_out3 = self.att_branch_2(att_feature_out1)
            # 8 x 8 x 2048
            ret = torch.cat((ret, att_feature_out3), dim=1)

        # attention branch 2
        attentive3 = attentive
        for i in range(self.att_channel_L):
            temp = attentive3[:, i].clone()
            temp = temp.view(-1, 1, 19, 33).expand(-1, 502, 19, 33)
            att_feature_out2 = feature_out2 * temp
            att_feature_out3 = self.att_branch_3(att_feature_out2)
            ret = torch.cat((ret, att_feature_out3), dim=1)

        # final feature size: [batch_size, 512 x 3 x L, 19, 33]
        return attentive, ret

    def init_weight(self):
        incept1_pretrained_state_dict = torch.load('data/pretrained_model/hp_mnet_incept1.pth')
        incept2_pretrained_state_dict = torch.load('data/pretrained_model/hp_mnet_incept2.pth')
        incept3_pretrained_state_dict = torch.load('data/pretrained_model/hp_mnet_incept3.pth')

        # att_brach_1
        att_branch_1_state_dict_1 = {'0.' + k: v for k, v in incept1_pretrained_state_dict.items()}
        att_branch_1_state_dict_2 = {'1.' + k: v for k, v in incept2_pretrained_state_dict.items()}
        att_branch_1_state_dict_3 = {'2.' + k: v for k, v in incept3_pretrained_state_dict.items()}

        att_branch_1_state_dict_1.update(att_branch_1_state_dict_2)
        att_branch_1_state_dict_1.update(att_branch_1_state_dict_3)
        self.att_branch_1.load_state_dict(att_branch_1_state_dict_1)

        # att_branch_2
        att_branch_2_state_dict_2 = {'0.' + k: v for k, v in incept2_pretrained_state_dict.items()}
        att_branch_2_state_dict_3 = {'1.' + k: v for k, v in incept3_pretrained_state_dict.items()}

        att_branch_2_state_dict_2.update(att_branch_2_state_dict_3)
        self.att_branch_2.load_state_dict(att_branch_2_state_dict_2)

        # att_branch_3
        self.att_branch_3.load_state_dict(incept3_pretrained_state_dict)
