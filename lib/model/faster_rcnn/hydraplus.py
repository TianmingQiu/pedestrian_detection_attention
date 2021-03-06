# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torchvision.models as models
from model.faster_rcnn.faster_rcnn import _fasterRCNN

import pdb
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from model.utils.config import cfg
from model.rpn.rpn import _RPN

from model.roi_layers import ROIAlign, ROIPool

# from model.roi_pooling.modules.roi_pool import _RoIPooling
# from model.roi_align.modules.roi_align import RoIAlignAvg

from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
import time
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta


# from .HydraPlus.Hydraplus import HP
from .HydraPlus.MNet import MNet
from .HydraPlus.AF_2 import AF2
from .HydraPlus.AF_3 import AF3


class HydraPlus(_fasterRCNN):
    def __init__(self, classes, class_agnostic=False, stage='MNet', test_flag=False):
        self.stage = stage
        self.dout_base_model = None
        if self.stage == 'MNet':
            self.dout_base_model = 512
        elif self.stage == 'AF2' or self.stage == 'AF3':
            self.dout_base_model = 512 * 3  # AF2: 512 * 3 * L
        self.class_agnostic = class_agnostic
        self.test_flag = test_flag

        _fasterRCNN.__init__(self, classes, class_agnostic)

    def _init_modules(self):
        if self.stage == 'MNet':
            self.RCNN_base = MNet(init_weights=False)  # todo: init weights only for first training
        elif self.stage == 'AF2':
            if self.test_flag:
                self.RCNN_base = AF2(att_out=True)
            else:
                self.RCNN_base = AF2()
        elif self.stage == 'AF3':
            if self.test_flag:
                self.RCNN_base = AF3(att_out=True)
            else:
                self.RCNN_base = AF3()
        # elif self.stage == 'HP':
            # self.RCNN_base = HP()  # different branch output parts of

        self.RCNN_top = nn.Sequential(
            nn.Linear(self.dout_base_model * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096)
        )

        # not using the last maxpool layer
        self.RCNN_cls_score = nn.Linear(4096, self.n_classes)

        if self.class_agnostic:
            self.RCNN_bbox_pred = nn.Linear(4096, 4)
        else:
            self.RCNN_bbox_pred = nn.Linear(4096, 4 * self.n_classes)

    def _head_to_tail(self, pool5):

        pool5_flat = pool5.view(pool5.size(0), -1)  # torch.Size([512, 512, 7, 7])
        fc7 = self.RCNN_top(pool5_flat)  # [512, 4096]

        return fc7

    def forward(self, im_data, im_info, gt_boxes, num_boxes):
        """
        override parent class
        """
        batch_size = im_data.size(0)  # torch.Size([2, 3, 600, 1067])

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        # feed image data to base model to obtain base feature map
        if self.stage == "MNet":
            _, _, _, base_feat = self.RCNN_base(im_data)  # torch.Size([2, 512, 37, 67])
        elif self.stage == "AF2" or self.stage == "AF3":
            if self.test_flag:
                attention, base_feat = self.RCNN_base(im_data)  # torch.Size([2, 512 * 3, 37, 67])
            else:
                base_feat = self.RCNN_base(im_data)

        # feed base feature map tp RPN to obtain rois
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)

        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        rois = Variable(rois)
        # do roi pooling based on predicted rois

        if cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1, 5))  # torch.Size([512, 512, 7, 7])
        # feed pooled features to top model
        pooled_feat = self._head_to_tail(pooled_feat)  # torch.Size([512, 4096])

        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score, 1)

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        if self.training:
            # classification loss
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)

            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)

        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        if self.test_flag:
            return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label, attention
        else:
            return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label
