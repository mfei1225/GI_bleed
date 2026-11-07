"""Faster R-CNN used by the locked GI-bleed detector checkpoints.

Input is 9 channels: 3 slices (center, previous, next) for venous, then
non-contrast, then arterial. Three separate ResNet-50 trunks. The feature
pyramid is the concatenation of venous, non-contrast, venous−non-contrast,
arterial, arterial−non-contrast, and arterial−venous.
"""
from collections import OrderedDict

import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import FeaturePyramidNetwork


class ResNetBody(nn.Module):
    def __init__(self):
        super().__init__()
        m = resnet50(weights=None)
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.layer1 = m.layer1
        self.layer2 = m.layer2
        self.layer3 = m.layer3
        self.layer4 = m.layer4

    def forward(self, x):
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return OrderedDict([("0", c2), ("1", c3), ("2", c4), ("3", c5)])


class MultiStreamFPNBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.out_channels = 256
        self.body_ref = ResNetBody()
        self.bodies = nn.ModuleList([ResNetBody(), ResNetBody()])
        self.diff_gammas = nn.ParameterList([
            nn.Parameter(torch.zeros(1)),
            nn.Parameter(torch.zeros(1)),
        ])
        self.temporal_gamma = nn.Parameter(torch.zeros(1))
        channels = [256, 512, 1024, 2048]
        self.reduce = nn.ModuleList([nn.Conv2d(c * 6, c, kernel_size=1) for c in channels])
        self.fpn = FeaturePyramidNetwork(in_channels_list=channels, out_channels=256, extra_blocks=None)

    def forward(self, x):
        venous, noncon, arterial = torch.split(x, 3, dim=1)
        feats = [
            self.body_ref(venous),
            self.bodies[0](noncon),
            self.bodies[1](arterial),
        ]
        fused = OrderedDict()
        for key in feats[0]:
            f_v = feats[0][key]
            f_nc = feats[1][key]
            f_a = feats[2][key]
            parts = [
                f_v,
                f_nc,
                self.diff_gammas[0] * (f_v - f_nc),
                f_a,
                self.diff_gammas[1] * (f_a - f_nc),
                self.temporal_gamma * (f_a - f_v),
            ]
            fused[key] = self.reduce[int(key)](torch.cat(parts, dim=1))
        return self.fpn(fused)


def build_detector(box_score_thresh: float = 0.1) -> FasterRCNN:
    backbone = MultiStreamFPNBackbone()
    anchors = AnchorGenerator(
        sizes=((8, 16), (32, 48), (64, 96), (128, 256)),
        aspect_ratios=((0.5, 0.75, 1.0, 1.5, 2.0),) * 4,
    )
    return FasterRCNN(
        backbone=backbone,
        num_classes=2,
        rpn_anchor_generator=anchors,
        min_size=512,
        max_size=512,
        image_mean=[0.0] * 9,
        image_std=[1.0] * 9,
        box_score_thresh=box_score_thresh,
        box_nms_thresh=0.3,
        box_detections_per_img=20,
        rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_test=500,
    )
