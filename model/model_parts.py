import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class DWC(nn.Module):
    def __init__(self, in_ch, out_ch, padding=0, kernel_size=3, bias=False, dilation=None, stride=1):
        super(DWC, self).__init__()
        if not dilation:
            self.depth_conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, groups=in_ch, bias=bias, stride=stride)
        else:
            self.depth_conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, groups=in_ch, bias=bias, dilation=dilation, stride=stride)

    def forward(self, x):
        return self.depth_conv(x)


class AxialDW(nn.Module):
    def __init__(self, dim, mixer_kernel, dilation=1):
        super().__init__()
        h, w = mixer_kernel
        self.dw_h = nn.Conv2d(dim, dim, kernel_size=(h, 1), padding='same', groups=dim, dilation=dilation)
        self.dw_w = nn.Conv2d(dim, dim, kernel_size=(1, w), padding='same', groups=dim, dilation=dilation)

    def forward(self, x):
        x = x + self.dw_h(x) + self.dw_w(x)
        return x


class PWC(nn.Module):
    def __init__(self, in_ch, out_ch, bias=False):
        super(PWC, self).__init__()
        self.point_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=bias)

    def forward(self, x):
        return self.point_conv(x)


class avgAdd(nn.Module):
    def __init__(self, Lin_ch, Hin_ch, out_ch, num=0.5):
        super(avgAdd, self).__init__()
        self.HH_ch = int(Hin_ch * num)
        self.HL_ch = Hin_ch - self.HH_ch
        self.LL_ch = int(Lin_ch * num)
        self.LH_ch = Lin_ch - self.LL_ch
        self.H_out = int(out_ch * num)
        self.L_out = out_ch - self.H_out

        self.LH = nn.Sequential(
            nn.Conv2d(self.LH_ch, self.H_out, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.HL = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(self.HL_ch, self.L_out, kernel_size=3, padding=1)
        )
        self.LL = nn.Sequential(
            nn.Conv2d(self.LL_ch, self.L_out, kernel_size=3, padding=1)
        )
        self.HH = nn.Sequential(
            nn.Conv2d(self.HH_ch, self.H_out, kernel_size=3, padding=1)
        )

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.ch_attention = nn.Sequential(
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU()
        )

    def softmax_add(self, x1, x2):
        x1_ = self.avgpool(x1)
        b_, c_, _, _ = x1_.size()
        x1_ = x1_.view(b_, c_, 1)
        x2_ = self.avgpool(x2)
        b_, c_, _, _ = x2_.size()
        x2_ = x2_.view(b_, c_, 1)
        y = torch.cat([x1_, x2_], dim=2)
        y = F.softmax(y, dim=2)
        out1, out2 = torch.split(y, y.size(2) // 2, dim=2)
        out1 = out1.unsqueeze(dim=2)
        out2 = out2.unsqueeze(dim=2)
        return out1, out2

    def forward(self, Lx, Hx):
        LHx = self.LH(Lx[:, self.LL_ch:, :, :])
        HHx = self.HH(Hx[:, :self.HH_ch, :, :])
        HLx = self.HL(Hx[:, self.HH_ch:, :, :])
        LLx = self.LL(Lx[:, :self.LL_ch, :, :])

        out1, out2 = self.softmax_add(HHx, LHx)
        out3, out4 = self.softmax_add(LLx, HLx)
        x1 = out1 * HHx + out2 * LHx
        x2 = out3 * LLx + out4 * HLx
        out = self.ch_attention(torch.cat([x1, self.up(x2)], dim=1))
        return out


class avgAdd_V0(nn.Module):
    def __init__(self, Lin_ch, Hin_ch, out_ch, num=0.5):
        super(avgAdd_V0, self).__init__()
        self.HH_ch = int(Hin_ch * num)
        self.HL_ch = Hin_ch - self.HH_ch
        self.LL_ch = int(Lin_ch * num)
        self.LH_ch = Lin_ch - self.LL_ch
        self.H_out = int(out_ch * num)
        self.L_out = out_ch - self.H_out

        self.LH = nn.Sequential(
            nn.Conv2d(self.LH_ch, self.H_out, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.HL = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(self.HL_ch, self.L_out, kernel_size=3, padding=1)
        )
        self.LL = nn.Sequential(
            nn.Conv2d(self.LL_ch, self.L_out, kernel_size=3, padding=1)
        )
        self.HH = nn.Sequential(
            nn.Conv2d(self.HH_ch, self.H_out, kernel_size=3, padding=1)
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def softmax_add(self, x1, x2):
        x1_ = self.avgpool(x1)
        b_, c_, _, _ = x1_.size()
        x1_ = x1_.view(b_, c_, 1)
        x2_ = self.avgpool(x2)
        b_, c_, _, _ = x2_.size()
        x2_ = x2_.view(b_, c_, 1)
        y = torch.cat([x1_, x2_], dim=2)
        y = F.softmax(y, dim=2)
        out1, out2 = torch.split(y, y.size(2) // 2, dim=2)
        out1 = out1.unsqueeze(dim=2)
        out2 = out2.unsqueeze(dim=2)
        return out1, out2

    def forward(self, Lx, Hx):
        LHx = self.LH(Lx[:, self.LL_ch:, :, :])
        HHx = self.HH(Hx[:, :self.HH_ch, :, :])
        HLx = self.HL(Hx[:, self.HH_ch:, :, :])
        LLx = self.LL(Lx[:, :self.LL_ch, :, :])

        out1, out2 = self.softmax_add(HHx, LHx)
        out3, out4 = self.softmax_add(LLx, HLx)
        x1 = out1 * HHx + out2 * LHx
        x2 = out3 * LLx + out4 * HLx
        return x2, x1

#
# class MixAdd_V0(nn.Module):
#     def __init__(self, Lin_ch, Hin_ch, out_ch, num=0.5):
#         super(MixAdd_V0, self).__init__()
#         self.HH_ch = int(Hin_ch * num)
#         self.HL_ch = Hin_ch - self.HH_ch
#         self.LL_ch = int(Lin_ch * num)
#         self.LH_ch = Lin_ch - self.LL_ch
#         self.oc_h = int(num * out_ch)
#         self.ou_l = out_ch - self.oc_h
#
#         self.LH = nn.Sequential(
#             nn.Conv2d(self.LH_ch, self.HH_ch, kernel_size=3, padding=1),
#             nn.BatchNorm2d(self.HH_ch),
#             nn.GELU(),
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         )
#         self.HL = nn.Sequential(
#             nn.AvgPool2d(2),
#             nn.Conv2d(self.HL_ch, self.LL_ch, kernel_size=3, padding=1),
#             nn.BatchNorm2d(self.LL_ch),
#             nn.GELU()
#         )
#         self.h_conv = nn.Sequential(
#             nn.Conv2d(self.HH_ch, self.oc_h, kernel_size=3, padding=1),
#             nn.BatchNorm2d(self.oc_h)
#         )
#         self.l_conv = nn.Sequential(
#             nn.Conv2d(self.LL_ch, self.ou_l, kernel_size=3, padding=1),
#             nn.BatchNorm2d(self.ou_l)
#         )
#         self.gelu = nn.GELU()
#         self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         self.ch_attention = nn.Conv2d(out_ch, out_ch, kernel_size=1)
#
#     def forward(self, Lx, Hx):
#         LHx = self.LH(Lx[:, self.LL_ch:, :, :])
#         HHx = Hx[:, :self.HH_ch, :, :]
#         HLx = self.HL(Hx[:, self.HH_ch:, :, :])
#         LLx = Lx[:, :self.LL_ch, :, :]
#
#         x1 = HHx + LHx
#         x2 = LLx + HLx
#         y1 = self.h_conv(x1)
#         y2 = self.l_conv(x2)
#         return y2, y1
#
#
# class MixAdd(nn.Module):
#     def __init__(self, Lin_ch, Hin_ch, out_ch, num=0.5):
#         super(MixAdd, self).__init__()
#         self.HH_ch = int(Hin_ch * num)
#         self.HL_ch = Hin_ch - self.HH_ch
#         self.LL_ch = int(Lin_ch * num)
#         self.LH_ch = Lin_ch - self.LL_ch
#         self.oc_h = int(num * out_ch)
#         self.ou_l = out_ch - self.oc_h
#
#         self.LH = nn.Sequential(
#             nn.Conv2d(self.LH_ch, self.HH_ch, kernel_size=3, padding=1),
#             nn.BatchNorm2d(self.HH_ch),
#             nn.GELU(),
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         )
#         self.HL = nn.Sequential(
#             nn.AvgPool2d(2),
#             nn.Conv2d(self.HL_ch, self.LL_ch, kernel_size=3, padding=1),
#             nn.BatchNorm2d(self.LL_ch),
#             nn.GELU()
#         )
#         self.h_conv = nn.Sequential(
#             nn.Conv2d(self.HH_ch, self.oc_h, kernel_size=3, padding=1),
#             nn.BatchNorm2d(self.oc_h)
#         )
#         self.l_conv = nn.Sequential(
#             nn.Conv2d(self.LL_ch, self.ou_l, kernel_size=3, padding=1),
#             nn.BatchNorm2d(self.ou_l)
#         )
#         self.gelu = nn.GELU()
#         self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         self.bn = nn.BatchNorm2d(out_ch)
#         self.ch_attention = nn.Conv2d(out_ch, out_ch, kernel_size=1)
#
#     def forward(self, Lx, Hx):
#         LHx = self.LH(Lx[:, self.LL_ch:, :, :])
#         HHx = Hx[:, :self.HH_ch, :, :]
#         HLx = self.HL(Hx[:, self.HH_ch:, :, :])
#         LLx = Lx[:, :self.LL_ch, :, :]
#
#         x1 = HHx + LHx
#         x2 = LLx + HLx
#         y1 = self.h_conv(x1)
#         y2 = self.l_conv(x2)
#         out = self.ch_attention(torch.cat([y1, self.up(y2)], dim=1))
#         return self.gelu(out)


class DiffAdd(nn.Module):
    def __init__(self, Lin_ch, Hin_ch, out_ch, num=0.5):
        super(DiffAdd, self).__init__()
        self.HH_ch = int(Hin_ch * num)
        self.HL_ch = Hin_ch - self.HH_ch
        self.LL_ch = int(Lin_ch * num)
        self.LH_ch = Lin_ch - self.LL_ch
        self.oc_h = int(num * out_ch)
        self.ou_l = out_ch - self.oc_h

        self.LH = nn.Sequential(
            nn.Conv2d(self.LL_ch, self.HH_ch, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.HL = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(self.HL_ch, self.LH_ch, kernel_size=3, padding=1)
        )
        self.h_conv = nn.Sequential(
            nn.Conv2d(2 * self.HH_ch, self.oc_h, kernel_size=3, padding=1)
        )
        self.l_conv = nn.Sequential(
            nn.Conv2d(2 * self.LH_ch, self.ou_l, kernel_size=3, padding=1)
        )
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.ch_attention = nn.Sequential(
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU()
        )

    def forward(self, Lx, Hx):
        LHx = self.LH(Lx[:, :self.LL_ch, :, :])
        HHx = Hx[:, :self.HH_ch, :, :]
        HLx = self.HL(Hx[:, self.HH_ch:, :, :])
        LLx = Lx[:, self.LL_ch:, :, :]

        x1 = self.h_conv(torch.cat([HHx, HHx - LHx], dim=1))
        x2 = self.l_conv(torch.cat([LLx, LLx - HLx], dim=1))
        out = self.ch_attention(torch.cat([x1, self.up(x2)], dim=1))
        return out


class DiffAdd_V0(nn.Module):
    def __init__(self, Lin_ch, Hin_ch, out_ch, num=0.5):
        super(DiffAdd_V0, self).__init__()
        self.HH_ch = int(Hin_ch * num)
        self.HL_ch = Hin_ch - self.HH_ch
        self.LL_ch = int(Lin_ch * num)
        self.LH_ch = Lin_ch - self.LL_ch
        self.oc_h = int(num * out_ch)
        self.ou_l = out_ch - self.oc_h

        self.LH = nn.Sequential(
            nn.Conv2d(self.LL_ch, self.HH_ch, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.HL = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(self.HL_ch, self.LH_ch, kernel_size=3, padding=1)
        )
        self.h_conv = nn.Sequential(
            nn.Conv2d(2 * self.HH_ch, self.oc_h, kernel_size=3, padding=1)
        )
        self.l_conv = nn.Sequential(
            nn.Conv2d(2 * self.LH_ch, self.ou_l, kernel_size=3, padding=1)
        )

    def forward(self, Lx, Hx):
        LHx = self.LH(Lx[:, :self.LL_ch, :, :])
        HHx = Hx[:, :self.HH_ch, :, :]
        HLx = self.HL(Hx[:, self.HH_ch:, :, :])
        LLx = Lx[:, self.LL_ch:, :, :]

        # x1 = self.act(self.bn1(self.h_conv(torch.cat([HHx, HHx - LHx], dim=1))))
        # x2 = self.act(self.bn2(self.l_conv(torch.cat([LLx, LLx - HLx], dim=1))))
        x1 = self.h_conv(torch.cat([HHx, HHx - LHx], dim=1))
        x2 = self.l_conv(torch.cat([LLx, LLx - HLx], dim=1))
        return x2, x1


# class avgMixAdd_V0(nn.Module):
#     def __init__(self, Lin_ch, Hin_ch, out_ch, num=0.5):
#         super(avgMixAdd_V0, self).__init__()
#         mid_ch = int(0.5 * out_ch)
#         self.mixadd = MixAdd(Lin_ch, Hin_ch, mid_ch, num)
#         self.front_ch = int(mid_ch * num)
#         self.late_ch = mid_ch - self.front_ch
#         self.front_out = int(out_ch * num)
#         self.late_out = out_ch - self.front_out
#
#         self.LL = nn.Sequential(
#             nn.Conv2d(self.front_ch, self.front_out, kernel_size=3, padding=1),
#             nn.BatchNorm2d(self.front_out)
#         )
#         self.LH = nn.Sequential(
#             nn.Conv2d(self.late_ch, self.late_out, kernel_size=3, padding=1),
#             nn.BatchNorm2d(self.late_out),
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         )
#         self.HH = nn.Sequential(
#             nn.Conv2d(self.front_ch, self.late_out, kernel_size=3, padding=1),
#             nn.BatchNorm2d(self.late_out)
#         )
#         self.HL = nn.Sequential(
#             nn.AvgPool2d(2),
#             nn.Conv2d(self.late_ch, self.front_out, kernel_size=3, padding=1),
#             nn.BatchNorm2d(self.front_out)
#         )
#         self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         self.ch_attention = nn.Conv2d(out_ch, out_ch, kernel_size=1)
#         self.avgpool = nn.AdaptiveAvgPool2d(1)
#         self.gelu = nn.GELU()
#
#     def softmax_add(self, x1, x2):
#         x1_ = self.avgpool(x1)
#         b_, c_, _, _ = x1_.size()
#         x1_ = x1_.view(b_, c_, 1)
#         x2_ = self.avgpool(x2)
#         b_, c_, _, _ = x2_.size()
#         x2_ = x2_.view(b_, c_, 1)
#         y = torch.cat([x1_, x2_], dim=2)
#         y = F.softmax(y, dim=2)
#         out1, out2 = torch.split(y, y.size(2) // 2, dim=2)
#         out1 = out1.unsqueeze(dim=2)
#         out2 = out2.unsqueeze(dim=2)
#         return out1, out2
#
#     def forward(self, Lx, Hx):
#         y1, y2 = self.mixadd(Lx, Hx)
#         LLx = self.LL(y2[:, :self.front_ch, :, :])
#         HHx = self.HH(y1[:, :self.front_ch, :, :])
#         LHx = self.LH(y2[:, self.front_ch:, :, :])
#         HLx = self.HL(y1[:, self.front_ch:, :, :])
#
#         out1, out2 = self.softmax_add(HHx, LHx)
#         out3, out4 = self.softmax_add(LLx, HLx)
#         x1 = out1 * HHx + out2 * LHx
#         x2 = out3 * LLx + out4 * HLx
#         # x1 = HHx + LHx
#         # x2 = LLx + HLx
#         out = self.ch_attention(torch.cat([x1, self.up(x2)], dim=1))
#         return self.gelu(out)


class avgMixAdd(nn.Module):
    def __init__(self, Lin_ch, Hin_ch, out_ch, num0=0.5, num1=0.5, mid_ch=0):
        super(avgMixAdd, self).__init__()
        if mid_ch == 0:
            mid_ch = out_ch
        self.avgAdd = avgAdd_V0(Lin_ch, Hin_ch, mid_ch, num0)
        self.diffAdd = DiffAdd_V0(mid_ch - int(num0 * mid_ch), int(num0 * mid_ch), out_ch, num1)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.ch_attention = nn.Sequential(
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU()
        )

    def forward(self, Lx, Hx):
        y1, y2 = self.avgAdd(Lx, Hx)
        out1, out2 = self.diffAdd(y1, y2)
        out = self.ch_attention(torch.cat([out2, self.up(out1)], dim=1))
        return out


class avgMixAdd_V1(nn.Module):
    def __init__(self, Lin_ch, Hin_ch, out_ch, mid_ch=0, num0=0.5, num1=0.5):
        super(avgMixAdd_V1, self).__init__()
        if mid_ch == 0:
            mid_ch = out_ch
        self.avgAdd = avgAdd_V0(Lin_ch, Hin_ch, mid_ch, num0)
        self.diffAdd = DiffAdd_V0(int(num0 * mid_ch), mid_ch - int(num0 * mid_ch), out_ch, num1)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.bn = nn.BatchNorm2d(out_ch)
        self.gelu = nn.GELU()

    def forward(self, Lx, Hx):
        y1, y2 = self.avgAdd(Lx, Hx)
        out1, out2 = self.diffAdd(y1, y2)
        out = self.gelu(self.bn(torch.cat([out2, self.up(out1)], dim=1)))
        return out


#
# class avgAdd_V0(nn.Module):
#     def __init__(self, Lin_ch, Hin_ch, out_ch):
#         super(avgAdd_V0, self).__init__()
#         self.HH_ch = int(Hin_ch / 2)
#         self.HL_ch = Hin_ch - self.HH_ch
#         self.LL_ch = int(Lin_ch / 2)
#         self.LH_ch = Lin_ch - self.LL_ch
#         self.LL_HH_out = int(out_ch / 2)
#         self.LH_HL_out = out_ch - self.LL_HH_out
#
#         self.LL = nn.Sequential(
#             nn.Conv2d(self.LL_ch, self.LL_HH_out, kernel_size=3, padding=1),
#             nn.BatchNorm2d(self.LL_HH_out)
#         )
#         self.LH = nn.Sequential(
#             nn.Conv2d(self.LH_ch, self.LH_HL_out, kernel_size=3, padding=1),
#             nn.BatchNorm2d(self.LH_HL_out),
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         )
#         self.HH = nn.Sequential(
#             nn.Conv2d(self.HH_ch, self.LL_HH_out, kernel_size=3, padding=1),
#             nn.BatchNorm2d(self.LL_HH_out)
#         )
#         self.HL = nn.Sequential(
#             nn.Conv2d(self.HL_ch, self.LH_HL_out, kernel_size=3, padding=1),
#             nn.BatchNorm2d(self.LH_HL_out),
#             nn.AvgPool2d(2)
#         )
#         self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         self.avgpool = nn.AdaptiveAvgPool2d(1)
#         self.gelu = nn.GELU()
#
#     def forward(self, Lx, Hx):
#         LHx = self.LH(Lx[:, :self.LL_ch, :, :])
#         HHx = self.HH(Hx[:, :self.HH_ch, :, :])
#         HLx = self.HL(Hx[:, self.HH_ch:, :, :])
#         LLx = self.LL(Lx[:, self.LL_ch:, :, :])
#
#         HHx_ = self.avgpool(HHx)
#         b_, c_, _, _ = HHx_.size()
#         HHx_ = HHx_.view(b_, c_, 1)
#         LHx_ = self.avgpool(LHx)
#         LHx_ = LHx_.view(b_, c_, 1)
#         H_ = torch.cat([HHx_, LHx_], dim=2)
#         H_ = F.softmax(H_, dim=2)
#         out1, out2 = torch.split(H_, H_.size(2) // 2, dim=2)
#         out1 = out1.unsqueeze(dim=2)
#         out2 = out2.unsqueeze(dim=2)
#
#         HLx_ = self.avgpool(HLx)
#         b_, c_, _, _ = HLx_.size()
#         HLx_ = HLx_.view(b_, c_, 1)
#         LLx_ = self.avgpool(LLx)
#         LLx_ = LLx_.view(b_, c_, 1)
#         L_ = torch.cat([LLx_, HLx_], dim=2)
#         L_ = F.softmax(L_, dim=2)
#         out3, out4 = torch.split(L_, L_.size(2) // 2, dim=2)
#         out3 = out3.unsqueeze(dim=2)
#         out4 = out4.unsqueeze(dim=2)
#
#         x1 = out1 * HHx + out2 * LHx
#         x2 = out3 * LLx + out4 * HLx
#         out = torch.cat([x1, self.up(x2)], dim=1)
#         return self.gelu(out)
#
#
# class MixAdd_V0(nn.Module):
#     def __init__(self, Lin_ch, Hin_ch, out_ch):
#         super(MixAdd_V0, self).__init__()
#         self.HH_ch = int(Hin_ch / 2)
#         self.HL_ch = Hin_ch - self.HH_ch
#         self.LL_ch = int(Lin_ch / 2)
#         self.LH_ch = Lin_ch - self.LL_ch
#         self.LL_HH_out = int(out_ch / 2)
#         self.LH_HL_out = out_ch - self.LL_HH_out
#
#         self.LL = nn.Sequential(
#             nn.Conv2d(self.LL_ch, self.LL_HH_out, kernel_size=3, padding=1),
#             nn.BatchNorm2d(self.LL_HH_out)
#         )
#         self.LH = nn.Sequential(
#             nn.Conv2d(self.LH_ch, self.LH_HL_out, kernel_size=3, padding=1),
#             nn.BatchNorm2d(self.LH_HL_out),
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         )
#         self.HH = nn.Sequential(
#             nn.Conv2d(self.HH_ch, self.LL_HH_out, kernel_size=3, padding=1),
#             nn.BatchNorm2d(self.LL_HH_out)
#         )
#         self.HL = nn.Sequential(
#             nn.Conv2d(self.HL_ch, self.LH_HL_out, kernel_size=3, padding=1),
#             nn.BatchNorm2d(self.LH_HL_out),
#             nn.AvgPool2d(2)
#         )
#         self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         self.gelu = nn.GELU()
#
#     def forward(self, Lx, Hx):
#         LHx = self.LH(Lx[:, self.LL_ch:, :, :])
#         HHx = self.HH(Hx[:, :self.HH_ch, :, :])
#         HLx = self.HL(Hx[:, self.HH_ch:, :, :])
#         LLx = self.LL(Lx[:, :self.LL_ch, :, :])
#         x1 = HHx + LHx
#         x2 = LLx + HLx
#         out = torch.cat([x1, self.up(x2)], dim=1)
#         return self.gelu(out)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        out = self.double_conv(x)
        return out


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        out = self.maxpool_conv(x)
        return out


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x0 = self.up(x1)
        # # input is CHW
        # diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        # diffX = torch.tensor([x2.size()[3] - x1.size()[3]])
        #
        # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                 diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x0], dim=1)
        out = self.conv(x)
        return out


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.conv(x)
        return out


class BasicBlock1(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch=None):
        super(BasicBlock1, self).__init__()
        if not mid_ch:
            mid_ch = out_ch // 4
        self.path1 = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_ch),
            nn.GELU(),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch)
        )
        self.path2 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch)
        )
        self.GELU = nn.GELU()

    def forward(self, x):
        x1 = self.path1(x)
        x2 = self.path2(x)
        x0 = x1 + x2
        x0 = self.GELU(x0)
        return x0


class BasicBlock2(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch=None):
        super(BasicBlock2, self).__init__()
        if not mid_ch:
            mid_ch = out_ch // 4
        self.path1 = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_ch),
            nn.GELU(),
            nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_ch),
            nn.GELU(),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch)
        )
        self.path2 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1),
            nn.BatchNorm2d(out_ch)
        )
        self.GELU = nn.GELU()

    def forward(self, x):
        x1 = self.path1(x)
        x2 = self.path2(x)
        x = x1 + x2
        x = self.GELU(x)
        return x


class BasicBlock3(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch=None):
        super(BasicBlock3, self).__init__()
        if not mid_ch:
            mid_ch = out_ch
        self.path1 = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(mid_ch),
            nn.GELU(),
            nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(mid_ch),
            nn.GELU(),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(out_ch)
        )
        self.path2 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1),
            nn.BatchNorm2d(out_ch)
        )
        self.GELU = nn.GELU()

    def forward(self, x):
        x1 = self.path1(x)
        x2 = self.path2(x)
        x = x1 + x2
        x = self.GELU(x)
        return x


class Down1(nn.Module):
    def __init__(self, in_ch, out_ch, pooling=False):
        super(Down1, self).__init__()
        if pooling:
            self.down_conv = nn.Sequential(
                nn.MaxPool2d(2),
                BasicBlock1(in_ch, out_ch)
            )
        else:
            self.down_conv = nn.Sequential(
                nn.MaxPool2d(2),
                BasicBlock1(in_ch, out_ch)
            )

    def forward(self, x):
        return self.down_conv(x)


