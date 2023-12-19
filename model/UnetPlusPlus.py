import torch
import torch.nn as nn
import torch.nn.functional as F
from .model_parts import avgMixAdd


class ContinusParalleConv(nn.Module):
    # 一个连续的卷积模块，包含BatchNorm 在前 和 在后 两种模式
    def __init__(self, in_channels, out_channels, pre_Batch_Norm=True):
        super(ContinusParalleConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if pre_Batch_Norm:
            self.Conv_forward = nn.Sequential(
                nn.BatchNorm2d(self.in_channels),
                nn.ReLU(),
                nn.Conv2d(self.in_channels, self.out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1))

        else:
            self.Conv_forward = nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1),
                nn.BatchNorm2d(self.out_channels),
                nn.ReLU())

    def forward(self, x):
        x = self.Conv_forward(x)
        return x


# class avgMixAdd(nn.Module):
#     def __init__(self, Lin_ch, Hin_ch, out_ch):
#         super(avgMixAdd, self).__init__()
#         self.HH_ch = int(Hin_ch / 2)
#         self.HL_ch = Hin_ch - self.HH_ch
#         self.LL_ch = int(Lin_ch / 2)
#         self.LH_ch = Lin_ch - self.LL_ch
#         self.LL_HH_out = int(out_ch / 2)
#         self.LH_HL_out = out_ch - self.LL_HH_out
#
#         self.LL = nn.Sequential(
#             nn.Conv2d(self.LL_ch, self.LL_HH_out, kernel_size=3, padding=1),
#             nn.BatchNorm2d(self.LL_HH_out),
#             nn.GELU()
#         )
#         self.LH = nn.Sequential(
#             nn.Conv2d(self.LH_ch, self.LH_HL_out, kernel_size=3, padding=1),
#             nn.BatchNorm2d(self.LH_HL_out),
#             nn.GELU(),
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         )
#         self.HH = nn.Sequential(
#             nn.Conv2d(self.HH_ch, self.LL_HH_out, kernel_size=3, padding=1),
#             nn.BatchNorm2d(self.LL_HH_out),
#             nn.GELU()
#         )
#         self.HL = nn.Sequential(
#             nn.Conv2d(self.HL_ch, self.LH_HL_out, kernel_size=3, padding=1),
#             nn.BatchNorm2d(self.LH_HL_out),
#             nn.GELU(),
#             nn.AvgPool2d(2)
#         )
#         self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         self.avgpool = nn.AdaptiveAvgPool2d(1)
#
#     def forward(self, Lx, *Hxs):
#         LHx = self.LH(Lx[:, self.LL_ch:, :, :])
#         LLx = self.LL(Lx[:, :self.LL_ch, :, :])
#         HHx, HLx, _num = [], [], 1
#         for i in Hxs:
#             HHx.append(self.HH(i[:, :self.HH_ch, :, :]))
#             HLx.append(self.HL(i[:, self.HH_ch:, :, :]))
#             _num += 1
#
#         HHx_ = []
#         for i in HHx:
#             temp = self.avgpool(i)
#             b_, c_, _, _ = temp.size()
#             HHx_.append(temp.view(b_, c_, 1))
#         LHx_ = self.avgpool(LHx)
#         b_, c_, _, _ = LHx_.size()
#         LHx_ = LHx_.view(b_, c_, 1)
#         H_ = torch.cat([i for i in HHx_], dim=2)
#         H_ = torch.cat([H_, LHx_], dim=2)
#         H_ = F.softmax(H_, dim=2)
#         out_H = torch.split(H_, H_.size(2) // _num, dim=2)
#         for i in range(len(out_H)):
#             out_H[i] = out_H[i].unsqueeze(dim=2)
#
#         HLx_ = []
#         for i in HLx:
#             temp = self.avgpool(i)
#             b_, c_, _, _ = temp.size()
#             HLx_.append(temp.view(b_, c_, 1))
#         LLx_ = self.avgpool(LLx)
#         b_, c_, _, _ = LLx_.size()
#         LLx_ = LLx_.view(b_, c_, 1)
#         L_ = torch.cat([i for i in HLx_], dim=2)
#         L_ = torch.cat([LLx_, L_], dim=2)
#         L_ = F.softmax(L_, dim=2)
#         out_L = torch.split(L_, L_.size(2) // _num, dim=2)
#         for i in range(len(out_L)):
#             out_L[i] = out_L[i].unsqueeze(dim=2)
#
#         x1, x2 = 0, 0
#         for i in range(len(out_H)):
#             if i < len(out_H) - 1:
#                 x1 += out_H[i] * HHx[i]
#             else:
#                 x1 += out_H[i] * LHx
#         for i in range(len(out_H)):
#             if i == 0:
#                 x2 += out_L[i] * LLx[i]
#             else:
#                 x2 += out_H[i] * HLx[i - 1]
#         out = torch.cat([x1, self.up(x2)], dim=1)
#         return out


class UnetPlusPlus(nn.Module):
    def __init__(self, in_ch, num_classes, deep_supervision=False):
        super(UnetPlusPlus, self).__init__()
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        self.filters = [64, 128, 256, 512, 1024]

        self.CONV3_1 = ContinusParalleConv(512 * 2, 512, pre_Batch_Norm=True)

        self.CONV2_2 = ContinusParalleConv(256 * 3, 256, pre_Batch_Norm=True)
        self.CONV2_1 = ContinusParalleConv(256 * 2, 256, pre_Batch_Norm=True)

        self.CONV1_1 = ContinusParalleConv(128 * 2, 128, pre_Batch_Norm=True)
        self.CONV1_2 = ContinusParalleConv(128 * 3, 128, pre_Batch_Norm=True)
        self.CONV1_3 = ContinusParalleConv(128 * 4, 128, pre_Batch_Norm=True)

        self.CONV0_1 = ContinusParalleConv(64 * 2, 64, pre_Batch_Norm=True)
        self.CONV0_2 = ContinusParalleConv(64 * 3, 64, pre_Batch_Norm=True)
        self.CONV0_3 = ContinusParalleConv(64 * 4, 64, pre_Batch_Norm=True)
        self.CONV0_4 = ContinusParalleConv(64 * 5, 64, pre_Batch_Norm=True)

        self.stage_0 = ContinusParalleConv(in_ch, 64, pre_Batch_Norm=False)
        self.stage_1 = ContinusParalleConv(64, 128, pre_Batch_Norm=False)
        self.stage_2 = ContinusParalleConv(128, 256, pre_Batch_Norm=False)
        self.stage_3 = ContinusParalleConv(256, 512, pre_Batch_Norm=False)
        self.stage_4 = ContinusParalleConv(512, 1024, pre_Batch_Norm=False)

        self.pool = nn.MaxPool2d(2)

        self.upsample_3_1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1)

        self.upsample_2_1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.upsample_2_2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1)

        self.upsample_1_1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.upsample_1_2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.upsample_1_3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)

        self.upsample_0_1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.upsample_0_2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.upsample_0_3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.upsample_0_4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)

        # 分割头
        self.final_super_0_1 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, self.num_classes, 3, padding=1),
        )
        self.final_super_0_2 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, self.num_classes, 3, padding=1),
        )
        self.final_super_0_3 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, self.num_classes, 3, padding=1),
        )
        self.final_super_0_4 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, self.num_classes, 3, padding=1),
        )

    def forward(self, x):
        x_0_0 = self.stage_0(x)
        x_1_0 = self.stage_1(self.pool(x_0_0))
        x_2_0 = self.stage_2(self.pool(x_1_0))
        x_3_0 = self.stage_3(self.pool(x_2_0))
        x_4_0 = self.stage_4(self.pool(x_3_0))

        x_0_1 = torch.cat([self.upsample_0_1(x_1_0), x_0_0], 1)
        x_0_1 = self.CONV0_1(x_0_1)

        x_1_1 = torch.cat([self.upsample_1_1(x_2_0), x_1_0], 1)
        x_1_1 = self.CONV1_1(x_1_1)

        x_2_1 = torch.cat([self.upsample_2_1(x_3_0), x_2_0], 1)
        x_2_1 = self.CONV2_1(x_2_1)

        x_3_1 = torch.cat([self.upsample_3_1(x_4_0), x_3_0], 1)
        x_3_1 = self.CONV3_1(x_3_1)

        x_2_2 = torch.cat([self.upsample_2_2(x_3_1), x_2_0, x_2_1], 1)
        x_2_2 = self.CONV2_2(x_2_2)

        x_1_2 = torch.cat([self.upsample_1_2(x_2_1), x_1_0, x_1_1], 1)
        x_1_2 = self.CONV1_2(x_1_2)

        x_1_3 = torch.cat([self.upsample_1_3(x_2_2), x_1_0, x_1_1, x_1_2], 1)
        x_1_3 = self.CONV1_3(x_1_3)

        x_0_2 = torch.cat([self.upsample_0_2(x_1_1), x_0_0, x_0_1], 1)
        x_0_2 = self.CONV0_2(x_0_2)

        x_0_3 = torch.cat([self.upsample_0_3(x_1_2), x_0_0, x_0_1, x_0_2], 1)
        x_0_3 = self.CONV0_3(x_0_3)

        x_0_4 = torch.cat([self.upsample_0_4(x_1_3), x_0_0, x_0_1, x_0_2, x_0_3], 1)
        x_0_4 = self.CONV0_4(x_0_4)

        if self.deep_supervision:
            out_put1 = self.final_super_0_1(x_0_1)
            out_put2 = self.final_super_0_2(x_0_2)
            out_put3 = self.final_super_0_3(x_0_3)
            out_put4 = self.final_super_0_4(x_0_4)
            return [out_put1, out_put2, out_put3, out_put4]
        else:
            return self.final_super_0_4(x_0_4)


class UnetPlusPlus_Mix(nn.Module):
    def __init__(self, in_ch, num_classes, deep_supervision=False):
        super(UnetPlusPlus_Mix, self).__init__()
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        self.filters = [64, 128, 256, 512, 1024]

        self.CONV3_1 = avgMixAdd(1024, 512, 512)

        self.CONV2_2 = ContinusParalleConv(256 * 3, 256, pre_Batch_Norm=True)
        self.CONV2_1 = avgMixAdd(512, 256, 256)

        self.CONV1_1 = avgMixAdd(256, 128, 128)
        self.CONV1_2 = ContinusParalleConv(128 * 3, 128, pre_Batch_Norm=True)
        self.CONV1_3 = ContinusParalleConv(128 * 4, 128, pre_Batch_Norm=True)

        self.CONV0_1 = avgMixAdd(128, 64, 64)
        self.CONV0_2 = ContinusParalleConv(64 * 3, 64, pre_Batch_Norm=True)
        self.CONV0_3 = ContinusParalleConv(64 * 4, 64, pre_Batch_Norm=True)
        self.CONV0_4 = ContinusParalleConv(64 * 5, 64, pre_Batch_Norm=True)

        self.stage_0 = ContinusParalleConv(in_ch, 64, pre_Batch_Norm=False)
        self.stage_1 = ContinusParalleConv(64, 128, pre_Batch_Norm=False)
        self.stage_2 = ContinusParalleConv(128, 256, pre_Batch_Norm=False)
        self.stage_3 = ContinusParalleConv(256, 512, pre_Batch_Norm=False)
        self.stage_4 = ContinusParalleConv(512, 1024, pre_Batch_Norm=False)

        self.pool = nn.MaxPool2d(2)

        self.upsample_2_2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1)

        self.upsample_1_2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.upsample_1_3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)

        self.upsample_0_2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.upsample_0_3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.upsample_0_4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)

        # 分割头
        self.final_super_0_1 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, self.num_classes, 3, padding=1),
        )
        self.final_super_0_2 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, self.num_classes, 3, padding=1),
        )
        self.final_super_0_3 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, self.num_classes, 3, padding=1),
        )
        self.final_super_0_4 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, self.num_classes, 3, padding=1),
        )

    def forward(self, x):
        x_0_0 = self.stage_0(x)
        x_1_0 = self.stage_1(self.pool(x_0_0))
        x_2_0 = self.stage_2(self.pool(x_1_0))
        x_3_0 = self.stage_3(self.pool(x_2_0))
        x_4_0 = self.stage_4(self.pool(x_3_0))

        x_0_1 = self.CONV0_1(x_1_0, x_0_0)

        x_1_1 = self.CONV1_1(x_2_0, x_1_0)

        x_2_1 = self.CONV2_1(x_3_0, x_2_0)

        x_3_1 = self.CONV3_1(x_4_0, x_3_0)

        x_2_2 = torch.cat([self.upsample_2_2(x_3_1), x_2_0, x_2_1], 1)
        x_2_2 = self.CONV2_2(x_2_2)

        x_1_2 = torch.cat([self.upsample_1_2(x_2_1), x_1_0, x_1_1], 1)
        x_1_2 = self.CONV1_2(x_1_2)

        x_1_3 = torch.cat([self.upsample_1_3(x_2_2), x_1_0, x_1_1, x_1_2], 1)
        x_1_3 = self.CONV1_3(x_1_3)

        x_0_2 = torch.cat([self.upsample_0_2(x_1_1), x_0_0, x_0_1], 1)
        x_0_2 = self.CONV0_2(x_0_2)

        x_0_3 = torch.cat([self.upsample_0_3(x_1_2), x_0_0, x_0_1, x_0_2], 1)
        x_0_3 = self.CONV0_3(x_0_3)

        x_0_4 = torch.cat([self.upsample_0_4(x_1_3), x_0_0, x_0_1, x_0_2, x_0_3], 1)
        x_0_4 = self.CONV0_4(x_0_4)

        if self.deep_supervision:
            out_put1 = self.final_super_0_1(x_0_1)
            out_put2 = self.final_super_0_2(x_0_2)
            out_put3 = self.final_super_0_3(x_0_3)
            out_put4 = self.final_super_0_4(x_0_4)
            return [out_put1, out_put2, out_put3, out_put4]
        else:
            return self.final_super_0_4(x_0_4)


