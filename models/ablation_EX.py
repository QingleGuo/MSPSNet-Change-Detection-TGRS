"""
Ablation study of the MSPSNet
"""
import torch
import torch.nn as nn

from models.MSPSNet import Conv_CAM_Layer, FEC, CAM_Module


class MSPSNET_WPCS(nn.Module):
    """without PCS"""

    def __init__(self, ou_ch=2):
        super(MSPSNET_WPCS, self).__init__()
        torch.nn.Module.dump_patches = True
        n1 = 40  # the initial number of channels of feature map
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.conv0_0 = nn.Conv2d(3, n1, kernel_size=5, padding=2, stride=1)
        self.conv0 = FEC(filters[0], filters[0], filters[0])
        self.conv2 = FEC(filters[0], filters[1], filters[1])
        self.conv4 = FEC(filters[1], filters[2], filters[2])
        self.conv5 = FEC(filters[2], filters[3], filters[3])
        self.conv6 = nn.Conv2d(filters[0], filters[0], kernel_size=1, stride=1)
        self.conv7 = nn.Conv2d(filters[0], ou_ch, kernel_size=3, padding=1, bias=False)

        self.conv1_1 = nn.Conv2d(filters[0] * 2 + filters[1], filters[0], kernel_size=1, stride=1)
        self.conv2_1 = nn.Conv2d(filters[1] * 2 + filters[2], filters[1], kernel_size=1, stride=1)
        self.conv3_1 = nn.Conv2d(filters[2] * 2 + filters[3], filters[2], kernel_size=1, stride=1)
        self.conv4_1 = nn.Conv2d(filters[3] * 2, filters[3], kernel_size=1, stride=1)

        self.cam_attention_1 = Conv_CAM_Layer(filters[0], filters[0], False)
        self.cam_attention_2 = Conv_CAM_Layer(filters[1], filters[1], False)
        self.cam_attention_3 = Conv_CAM_Layer(filters[2], filters[2], False)
        self.cam_attention_4 = Conv_CAM_Layer(filters[3], filters[3], False)

        self.c4_conv = nn.Conv2d(filters[3], filters[1], kernel_size=3, padding=1)
        self.c3_conv = nn.Conv2d(filters[2], filters[1], kernel_size=3, padding=1)
        self.c2_conv = nn.Conv2d(filters[1], filters[1], kernel_size=3, padding=1)
        self.c1_conv = nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1)

        self.pool1 = nn.AdaptiveAvgPool2d(128)
        self.pool2 = nn.AdaptiveAvgPool2d(64)
        self.pool3 = nn.AdaptiveAvgPool2d(32)

        self.Up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.Up2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.Up3 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x1, x2):

        # The first branch
        x1 = self.conv0(self.conv0_0(x1))
        x3 = self.conv2(self.pool1(x1))
        x4 = self.conv4(self.pool2(x3))
        A_F4 = self.conv5(self.pool3(x4))

        # the second branch
        x2 = self.conv0(self.conv0_0(x2))
        x5 = self.conv2(self.pool1(x2))
        x6 = self.conv4(self.pool2(x5))
        A_F8 = self.conv5(self.pool3(x6))

        c4 = self.cam_attention_4(self.conv4_1(torch.cat([A_F4, A_F8], 1)))
        c3 = self.cam_attention_3(self.conv3_1(torch.cat([torch.cat([x4, x6], 1), self.Up1(c4)], 1)))
        c2 = self.cam_attention_2(self.conv2_1(torch.cat([torch.cat([x3, x5], 1), self.Up1(c3)], 1)))
        c1 = self.cam_attention_1(self.conv1_1(torch.cat([torch.cat([x1, x2], 1), self.Up1(c2)], 1)))

        c1 = self.conv6(c1)
        out1 = self.conv7(c1)

        return (out1,)


class MSPSNET_WSA(nn.Module):
    """without SA"""

    def __init__(self, ou_ch=2):
        super(MSPSNET_WSA, self).__init__()
        torch.nn.Module.dump_patches = True
        n1 = 40  # the initial number of channels of feature map
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.conv0_0 = nn.Conv2d(3, n1, kernel_size=5, padding=2, stride=1)
        self.conv0 = FEC(filters[0], filters[0], filters[0])
        self.conv2 = FEC(filters[0], filters[1], filters[1])
        self.conv4 = FEC(filters[1], filters[2], filters[2])
        self.conv5 = FEC(filters[2], filters[3], filters[3])
        self.conv6 = nn.Conv2d(600, filters[1], kernel_size=1, stride=1)
        self.conv7 = nn.Conv2d(filters[1], ou_ch, kernel_size=3, padding=1, bias=False)

        self.conv6_1_1 = nn.Conv2d(filters[0] * 2, filters[0], padding=1, kernel_size=3, groups=filters[0] // 2,
                                   dilation=1)
        self.conv6_1_2 = nn.Conv2d(filters[0] * 2, filters[0], padding=2, kernel_size=3, groups=filters[0] // 2,
                                   dilation=2)
        self.conv6_1_3 = nn.Conv2d(filters[0] * 2, filters[0], padding=3, kernel_size=3, groups=filters[0] // 2,
                                   dilation=3)
        self.conv6_1_4 = nn.Conv2d(filters[0] * 2, filters[0], padding=4, kernel_size=3, groups=filters[0] // 2,
                                   dilation=4)
        self.conv1_1 = nn.Conv2d(filters[0] * 4, filters[0], kernel_size=1, stride=1)

        self.conv6_2_1 = nn.Conv2d(filters[1] * 2, filters[1], padding=1, kernel_size=3, groups=filters[1] // 2,
                                   dilation=1)
        self.conv6_2_2 = nn.Conv2d(filters[1] * 2, filters[1], padding=2, kernel_size=3, groups=filters[1] // 2,
                                   dilation=2)
        self.conv6_2_3 = nn.Conv2d(filters[1] * 2, filters[1], padding=3, kernel_size=3, groups=filters[1] // 2,
                                   dilation=3)
        self.conv6_2_4 = nn.Conv2d(filters[1] * 2, filters[1], padding=4, kernel_size=3, groups=filters[1] // 2,
                                   dilation=4)
        self.conv2_1 = nn.Conv2d(filters[1] * 4, filters[1], kernel_size=1, stride=1)

        self.conv6_3_1 = nn.Conv2d(filters[2] * 2, filters[2], padding=1, kernel_size=3, groups=filters[2] // 2,
                                   dilation=1)
        self.conv6_3_2 = nn.Conv2d(filters[2] * 2, filters[2], padding=2, kernel_size=3, groups=filters[2] // 2,
                                   dilation=2)
        self.conv6_3_3 = nn.Conv2d(filters[2] * 2, filters[2], padding=3, kernel_size=3, groups=filters[2] // 2,
                                   dilation=3)
        self.conv6_3_4 = nn.Conv2d(filters[2] * 2, filters[2], padding=4, kernel_size=3, groups=filters[2] // 2,
                                   dilation=4)
        self.conv3_1 = nn.Conv2d(filters[2] * 4, filters[2], kernel_size=1, stride=1)

        self.conv6_4_1 = nn.Conv2d(filters[3] * 2, filters[3], padding=1, kernel_size=3, groups=filters[3] // 2,
                                   dilation=1)
        self.conv6_4_2 = nn.Conv2d(filters[3] * 2, filters[3], padding=2, kernel_size=3, groups=filters[3] // 2,
                                   dilation=2)
        self.conv6_4_3 = nn.Conv2d(filters[3] * 2, filters[3], padding=3, kernel_size=3, groups=filters[3] // 2,
                                   dilation=3)
        self.conv6_4_4 = nn.Conv2d(filters[3] * 2, filters[3], padding=4, kernel_size=3, groups=filters[3] // 2,
                                   dilation=4)
        self.conv4_1 = nn.Conv2d(filters[3] * 4, filters[3], kernel_size=1, stride=1)

        self.c4_conv = nn.Conv2d(filters[3], filters[1], kernel_size=3, padding=1)
        self.c3_conv = nn.Conv2d(filters[2], filters[1], kernel_size=3, padding=1)
        self.c2_conv = nn.Conv2d(filters[1], filters[1], kernel_size=3, padding=1)
        self.c1_conv = nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1)

        self.pool1 = nn.AdaptiveAvgPool2d(128)
        self.pool2 = nn.AdaptiveAvgPool2d(64)
        self.pool3 = nn.AdaptiveAvgPool2d(32)

        self.Up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.Up2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.Up3 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x1, x2):

        # The first branch

        x1 = self.conv0(self.conv0_0(x1))
        x3 = self.conv2(self.pool1(x1))
        x4 = self.conv4(self.pool2(x3))
        A_F4 = self.conv5(self.pool3(x4))

        # The second branch

        x2 = self.conv0(self.conv0_0(x2))
        x5 = self.conv2(self.pool1(x2))
        x6 = self.conv4(self.pool2(x5))
        A_F8 = self.conv5(self.pool3(x6))

        c4 = ((self.conv4_1(torch.cat(
            [self.conv6_4_1(torch.cat([A_F4, A_F8], 1)), self.conv6_4_2(torch.cat([A_F4, A_F8], 1)),
             self.conv6_4_3(torch.cat([A_F4, A_F8], 1)), self.conv6_4_4(torch.cat([A_F4, A_F8], 1))], 1))))

        c3 = torch.cat([((self.conv3_1(torch.cat(
            [self.conv6_3_1(torch.cat([x4, x6], 1)), self.conv6_3_2(torch.cat([x4, x6], 1)),
             self.conv6_3_3(torch.cat([x4, x6], 1)), self.conv6_3_4(torch.cat([x4, x6], 1))], 1)))), self.Up1(c4)], 1)
        c2 = torch.cat([((self.conv2_1(torch.cat(
            [self.conv6_2_1(torch.cat([x3, x5], 1)), self.conv6_2_2(torch.cat([x3, x5], 1)),
             self.conv6_2_3(torch.cat([x3, x5], 1)), self.conv6_2_4(torch.cat([x3, x5], 1))], 1)))), self.Up1(c3)], 1)
        c1 = torch.cat([((self.conv1_1(torch.cat(
            [self.conv6_1_1(torch.cat([x1, x2], 1)), self.conv6_1_2(torch.cat([x1, x2], 1)),
             self.conv6_1_3(torch.cat([x1, x2], 1)), self.conv6_1_4(torch.cat([x1, x2], 1))], 1)))), self.Up1(c2)], 1)
        c1 = self.conv6(c1)
        out1 = self.conv7(c1)

        return (out1,)
