from typing import List, Callable

import torch
from torch import Tensor
import torch.nn as nn


def channel_shuffle(x: Tensor, groups: int) -> Tensor:

    batch_size, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    # [batch_size, num_channels, height, width] -> [batch_size, groups, channels_per_group, height, width]
    x = x.view(batch_size, groups, channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batch_size, -1, height, width)

    return x


class InvertedResidual(nn.Module):
    def __init__(self, input_c: int, output_c: int, stride: int):
        super(InvertedResidual, self).__init__()

        if stride not in [1, 2]:
            raise ValueError("illegal stride value.")
        self.stride = stride

        assert output_c % 2 == 0
        branch_features = output_c // 2
        # 当stride为1时，input_channel应该是branch_features的两倍
        # python中 '<<' 是位运算，可理解为计算×2的快速方法
        assert (self.stride != 1) or (input_c == branch_features << 1)

        if self.stride == 2:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(input_c, input_c, kernel_s=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(input_c),
                nn.Conv2d(input_c, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True)
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(input_c if self.stride > 1 else branch_features, branch_features, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_s=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True)
        )

    @staticmethod
    def depthwise_conv(input_c: int,
                       output_c: int,
                       kernel_s: int,
                       stride: int = 1,
                       padding: int = 0,
                       bias: bool = False) -> nn.Conv2d:
        return nn.Conv2d(in_channels=input_c, out_channels=output_c, kernel_size=kernel_s,
                         stride=stride, padding=padding, bias=bias, groups=input_c)

    def forward(self, x: Tensor) -> Tensor:
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out


class ShuffleNetV2(nn.Module):
    def __init__(self,
                 stages_repeats: List[int],
                 stages_out_channels: List[int],
                 num_classes: int = 1000,
                 inverted_residual: Callable[..., nn.Module] = InvertedResidual):
        super(ShuffleNetV2, self).__init__()

        if len(stages_repeats) != 3:
            raise ValueError("expected stages_repeats as list of 3 positive ints")
        if len(stages_out_channels) != 5:
            raise ValueError("expected stages_out_channels as list of 5 positive ints")
        self._stage_out_channels = stages_out_channels

        # input RGB image
        input_channels = 3
        output_channels = self._stage_out_channels[0]

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Static annotations for mypy
        self.stage2: nn.Sequential
        self.stage3: nn.Sequential
        self.stage4: nn.Sequential

        stage_names = ["stage{}".format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(stage_names, stages_repeats,
                                                  self._stage_out_channels[1:]):
###修改
            downsamping_block = inverted_residual(input_channels, output_channels, 2)
            setattr(self, name + '_down', downsamping_block)
            seq = []
            internel_channels = int(output_channels / 2)
            for i in range(repeats - 1):
                seq.append(inverted_residual(internel_channels, internel_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            #csp_conv1 = nn.Sequential(nn.Conv2d(internel_channels, internel_channels, 1, bias=False),
            #                          nn.BatchNorm2d(internel_channels),
            #                          nn.ReLU(inplace=True))
            csp_conv2 = nn.Sequential(nn.Conv2d(internel_channels, internel_channels, 1, bias=False),
                                      nn.BatchNorm2d(internel_channels),
                                      nn.ReLU(inplace=True))
            #csp_conv3 = nn.Sequential(nn.Conv2d(output_channels, output_channels, 1, bias=False),
            #                          nn.BatchNorm2d(output_channels),
            #                          nn.ReLU(inplace=True))
            #setattr(self, name + '_csp_conv1', csp_conv1)
            setattr(self, name + '_csp_conv2', csp_conv2)
            #setattr(self, name + '_csp_conv3', csp_conv3)
            input_channels = output_channels

        output_channels = self._stage_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

        self.fc = nn.Linear(output_channels, num_classes)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        # x = self.conv1(x)
        # x = self.maxpool(x)
        # x = self.stage2(x)
        # x = self.stage3(x)
        # x = self.stage4(x)
        # x = self.conv5(x)
        # x = x.mean([2, 3])  # global pool
        # x = self.fc(x)
        x = self.conv1(x)
        x = self.maxpool(x)

        s2_donw = self.stage2_down(x)
        s2_ori = self.stage2(s2_donw[:, :int(s2_donw.shape[1] / 2), :, :])
        # s2_ori = self.stage2_csp_conv1(s2_ori)
        s2_csp = self.stage2_csp_conv2(s2_donw[:, int(s2_donw.shape[1] / 2):, :, :])
        s2_cat = torch.cat((s2_ori, s2_csp), dim=1)
        # s2 = self.stage2_csp_conv3(s2_cat)
        s2 = channel_shuffle(s2_cat, 2)

        s3_donw = self.stage3_down(s2)
        s3_ori = self.stage3(s3_donw[:, :int(s3_donw.shape[1] / 2), :, :])
        # s3_ori = self.stage3_csp_conv1(s3_ori)
        s3_csp = self.stage3_csp_conv2(s3_donw[:, int(s3_donw.shape[1] / 2):, :, :])
        s3_cat = torch.cat((s3_ori, s3_csp), dim=1)
        # s3 = self.stage3_csp_conv3(s3_cat)
        s3 = channel_shuffle(s3_cat, 2)

        s4_donw = self.stage4_down(s3)
        s4_ori = self.stage4(s4_donw[:, :int(s4_donw.shape[1] / 2), :, :])
        # s4_ori = self.stage4_csp_conv1(s4_ori)
        s4_csp = self.stage4_csp_conv2(s4_donw[:, int(s4_donw.shape[1] / 2):, :, :])
        s4_cat = torch.cat((s4_ori, s4_csp), dim=1)
        # s4 = self.stage4_csp_conv3(s4_cat)
        s4 = channel_shuffle(s4_cat, 2)

        x = self.conv5(s4)
        x = x.mean([2, 3])  # globalpool
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def shufflenet_v2_x1_0(num_classes=1000):
    """
    Constructs a ShuffleNetV2 with 1.0x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`.
    weight: https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth

    :param num_classes:
    :return:
    """
    model = ShuffleNetV2(stages_repeats=[4, 8, 4],
                         stages_out_channels=[24, 116, 232, 464, 1024],
                         num_classes=num_classes)

    return model


def shufflenet_v2_x0_5(num_classes=1000):
    """
    Constructs a ShuffleNetV2 with 0.5x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`.
    weight: https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth

    :param num_classes:
    :return:
    """
    model = ShuffleNetV2(stages_repeats=[4, 8, 4],
                         stages_out_channels=[24, 48, 96, 192, 1024],
                         num_classes=num_classes)

    return model
