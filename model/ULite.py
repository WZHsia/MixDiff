import torch
import torch.nn as nn
import torch.nn.functional as F


class AxialDW(nn.Module):
    def __init__(self, dim, mixer_kernel, dilation=1):
        super().__init__()
        h, w = mixer_kernel
        self.dw_h = nn.Conv2d(dim, dim, kernel_size=(h, 1), padding='same', groups=dim, dilation=dilation)
        self.dw_w = nn.Conv2d(dim, dim, kernel_size=(1, w), padding='same', groups=dim, dilation=dilation)

    def forward(self, x):
        x = x + self.dw_h(x) + self.dw_w(x)
        return x


class EncoderBlock(nn.Module):
    """Encoding then downsampling"""

    def __init__(self, in_c, out_c, mixer_kernel=(7, 7)):
        super().__init__()
        self.dw = AxialDW(in_c, mixer_kernel=(7, 7))
        self.bn = nn.BatchNorm2d(in_c)
        self.pw = nn.Conv2d(in_c, out_c, kernel_size=1)
        self.down = nn.MaxPool2d((2, 2))
        self.act = nn.GELU()

    def forward(self, x):
        skip = self.bn(self.dw(x))
        x = self.act(self.down(self.pw(skip)))
        return x, skip


class DecoderBlock(nn.Module):
    """Upsampling then decoding"""

    def __init__(self, in_c, out_c, mixer_kernel=(7, 7)):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.pw = nn.Conv2d(in_c + out_c, out_c, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_c)
        self.dw = AxialDW(out_c, mixer_kernel=(7, 7))
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(out_c, out_c, kernel_size=1)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.act(self.pw2(self.dw(self.bn(self.pw(x)))))
        return x


# class MixAdd(nn.Module):
#     def __init__(self, Lin_ch, Hin_ch, out_ch, num):
#         super(MixAdd, self).__init__()
#         self.HH_ch = int(Hin_ch * num)
#         self.HL_ch = Hin_ch - self.HH_ch
#         self.LL_ch = int(Lin_ch * num)
#         self.LH_ch = Lin_ch - self.LL_ch
#
#         self.LH = nn.Sequential(
#             nn.Conv2d(self.LH_ch, self.HH_ch, kernel_size=1),
#             nn.BatchNorm2d(self.HH_ch),
#             AxialDW(self.HH_ch, mixer_kernel=(7, 7)),
#             nn.GELU(),
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         )
#         self.HL = nn.Sequential(
#             nn.AvgPool2d(2),
#             nn.Conv2d(self.HL_ch, self.LL_ch, kernel_size=1),
#             nn.BatchNorm2d(self.LL_ch),
#             AxialDW(self.LL_ch, mixer_kernel=(7, 7)),
#             nn.GELU()
#         )
#         self.h_conv = nn.Sequential(
#             nn.Conv2d(self.HH_ch, out_ch, kernel_size=1)
#             # nn.BatchNorm2d(out_ch)
#         )
#         self.l_conv = nn.Sequential(
#             nn.Conv2d(self.LL_ch, out_ch, kernel_size=1)
#             # 可去
#             # nn.BatchNorm2d(out_ch)
#         )
#         self.gelu = nn.GELU()
#
#     def forward(self, Lx, Hx):
#         LHx = self.LH(Lx[:, self.LL_ch:, :, :])
#         HHx = Hx[:, :self.HH_ch, :, :]
#         HLx = self.HL(Hx[:, self.HH_ch:, :, :])
#         LLx = Lx[:, :self.LL_ch, :, :]
#
#         x1 = HHx + LHx
#         x2 = LLx + HLx
#         y1 = self.gelu(self.h_conv(x1))
#         y2 = self.gelu(self.l_conv(x2))
#         return y1, y2


# class avgMixAdd(nn.Module):
#     def __init__(self, Lin_ch, Hin_ch, out_ch, num=0.5):
#         super(avgMixAdd, self).__init__()
#         mid_ch = int(0.5 * out_ch)
#         self.mixadd = MixAdd(Lin_ch, Hin_ch, mid_ch, num)
#         self.front_ch = int(mid_ch * num)
#         self.late_ch = mid_ch - self.front_ch
#         self.front_out = int(out_ch * num)
#         self.late_out = out_ch - self.front_out
#
#         self.LL = nn.Sequential(
#             nn.Conv2d(self.front_ch, self.front_out, kernel_size=1),
#             nn.BatchNorm2d(self.front_out),
#             AxialDW(self.front_out, mixer_kernel=(7, 7))
#         )
#         self.LH = nn.Sequential(
#             nn.Conv2d(self.late_ch, self.late_out, kernel_size=1),
#             nn.BatchNorm2d(self.late_out),
#             AxialDW(self.late_out, mixer_kernel=(7, 7)),
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         )
#         self.HH = nn.Sequential(
#             nn.Conv2d(self.front_ch, self.late_out, kernel_size=1),
#             nn.BatchNorm2d(self.late_out),
#             AxialDW(self.late_out, mixer_kernel=(7, 7))
#         )
#         self.HL = nn.Sequential(
#             nn.AvgPool2d(2),
#             nn.Conv2d(self.late_ch, self.front_out, kernel_size=1),
#             nn.BatchNorm2d(self.front_out),
#             AxialDW(self.front_out, mixer_kernel=(7, 7))
#         )
#         self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         # 可用
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
            nn.Conv2d(self.LH_ch, self.H_out, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.HL = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(self.HL_ch, self.L_out, kernel_size=1)
        )
        self.LL = nn.Sequential(
            nn.Conv2d(self.LL_ch, self.L_out, kernel_size=1)
        )
        self.HH = nn.Sequential(
            nn.Conv2d(self.HH_ch, self.H_out, kernel_size=1)
        )

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
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
            nn.Conv2d(self.LL_ch, self.HH_ch, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.HL = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(self.HL_ch, self.LH_ch, kernel_size=1)
        )
        self.h_conv = nn.Sequential(
            nn.Conv2d(2 * self.HH_ch, self.oc_h, kernel_size=1)
        )
        self.l_conv = nn.Sequential(
            nn.Conv2d(2 * self.LH_ch, self.ou_l, kernel_size=1)
        )

    def forward(self, Lx, Hx):
        LHx = self.LH(Lx[:, :self.LL_ch, :, :])
        HHx = Hx[:, :self.HH_ch, :, :]
        HLx = self.HL(Hx[:, self.HH_ch:, :, :])
        LLx = Lx[:, self.LL_ch:, :, :]

        x1 = self.h_conv(torch.cat([HHx, HHx - LHx], dim=1))
        x2 = self.l_conv(torch.cat([LLx, LLx - HLx], dim=1))
        return x2, x1


class avgMixAdd(nn.Module):
    def __init__(self, Lin_ch, Hin_ch, out_ch, mid_ch=0, num0=0.5, num1=0.5):
        super(avgMixAdd, self).__init__()
        if mid_ch == 0:
            mid_ch = out_ch
        self.avgAdd = avgAdd_V0(Lin_ch, Hin_ch, mid_ch, num0)
        self.diffAdd = DiffAdd_V0(int(num0 * mid_ch), mid_ch - int(num0 * mid_ch), out_ch, num1)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.bn = nn.BatchNorm2d(out_ch)
        self.dw = AxialDW(out_ch, mixer_kernel=(7, 7))
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(out_ch, out_ch, kernel_size=1)

    def forward(self, Lx, Hx):
        y1, y2 = self.avgAdd(Lx, Hx)
        out1, out2 = self.diffAdd(y1, y2)
        x = torch.cat([out2, self.up(out1)], dim=1)
        out = self.act(self.pw2(self.dw(self.bn(x))))
        return out


class BottleNeckBlock(nn.Module):
    """Axial dilated DW convolution"""

    def __init__(self, dim):
        super().__init__()

        gc = dim // 4
        self.pw1 = nn.Conv2d(dim, gc, kernel_size=1)
        self.dw1 = AxialDW(gc, mixer_kernel=(3, 3), dilation=1)
        self.dw2 = AxialDW(gc, mixer_kernel=(3, 3), dilation=2)
        self.dw3 = AxialDW(gc, mixer_kernel=(3, 3), dilation=3)

        self.bn = nn.BatchNorm2d(4 * gc)
        self.pw2 = nn.Conv2d(4 * gc, dim, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.pw1(x)
        x = torch.cat([x, self.dw1(x), self.dw2(x), self.dw3(x)], 1)
        x = self.act(self.pw2(self.bn(x)))
        return x


class ULite(nn.Module):
    def __init__(self, in_ch, n_classes):
        super().__init__()

        """Encoder"""
        # skip in x out
        self.conv_in = nn.Conv2d(in_ch, 16, kernel_size=7, padding='same')
        self.e1 = EncoderBlock(16, 32)
        self.e2 = EncoderBlock(32, 64)
        self.e3 = EncoderBlock(64, 128)
        self.e4 = EncoderBlock(128, 256)
        self.e5 = EncoderBlock(256, 512)

        """Bottle Neck"""
        self.b5 = BottleNeckBlock(512)

        """Decoder"""
        self.d5 = DecoderBlock(512, 256)
        self.d4 = DecoderBlock(256, 128)
        self.d3 = DecoderBlock(128, 64)
        self.d2 = DecoderBlock(64, 32)
        self.d1 = DecoderBlock(32, 16)
        self.conv_out = nn.Conv2d(16, n_classes, kernel_size=1)

    def forward(self, x):
        """Encoder"""
        x = self.conv_in(x)
        x, skip1 = self.e1(x)
        x, skip2 = self.e2(x)
        x, skip3 = self.e3(x)
        x, skip4 = self.e4(x)
        x, skip5 = self.e5(x)

        """BottleNeck"""
        x = self.b5(x)  # (512, 8, 8)

        """Decoder"""
        x = self.d5(x, skip5)
        x = self.d4(x, skip4)
        x = self.d3(x, skip3)
        x = self.d2(x, skip2)
        x = self.d1(x, skip1)
        x = self.conv_out(x)
        return x


class ULite_Mixadd(nn.Module):
    def __init__(self, in_ch, n_classes):
        super().__init__()

        """Encoder"""
        self.conv_in = nn.Conv2d(in_ch, 16, kernel_size=7, padding='same')
        self.e1 = EncoderBlock(16, 32)
        self.e2 = EncoderBlock(32, 64)
        self.e3 = EncoderBlock(64, 128)
        self.e4 = EncoderBlock(128, 256)
        self.e5 = EncoderBlock(256, 512)

        """Bottle Neck"""
        self.b5 = BottleNeckBlock(512)

        """Decoder"""
        self.d5 = avgMixAdd(512, 256, 256)
        self.d4 = avgMixAdd(256, 128, 128)
        self.d3 = avgMixAdd(128, 64, 64)
        self.d2 = avgMixAdd(64, 32, 32)
        self.d1 = avgMixAdd(32, 16, 16)
        self.conv_out = nn.Conv2d(16, n_classes, kernel_size=1)

    def forward(self, x):
        """Encoder"""
        x = self.conv_in(x)
        x, skip1 = self.e1(x)
        x, skip2 = self.e2(x)
        x, skip3 = self.e3(x)
        x, skip4 = self.e4(x)
        x, skip5 = self.e5(x)

        """BottleNeck"""
        x = self.b5(x)  # (512, 8, 8)

        """Decoder"""
        x = self.d5(x, skip5)
        x = self.d4(x, skip4)
        x = self.d3(x, skip3)
        x = self.d2(x, skip2)
        x = self.d1(x, skip1)
        x = self.conv_out(x)
        return x




# class Mixadd_V1(nn.Module):
#     def __init__(self, in_c, out_c, num=0.5, mixer_kernel=(7, 7)):
#         super(Mixadd_V1, self).__init__()
#         self.HH_ch = int(out_c * num)
#         self.HL_ch = out_c - self.HH_ch
#         self.LL_ch = int(in_c * num)
#         self.LH_ch = in_c - self.LL_ch
#         self.H_out = int(out_c * num)
#         self.L_out = out_c - self.H_out
#
#         self.LH = nn.Sequential(
#             nn.Conv2d(self.LH_ch, self.HH_ch, kernel_size=1),
#             nn.BatchNorm2d(self.HH_ch),
#             AxialDW(self.HH_ch, mixer_kernel=(7, 7)),
#             nn.Conv2d(self.HH_ch, self.HH_ch, kernel_size=1),
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         )
#         self.HL = nn.Sequential(
#             nn.AvgPool2d(2),
#             nn.Conv2d(self.HL_ch, self.LL_ch, kernel_size=1),
#             AxialDW(self.LL_ch, mixer_kernel=(7, 7)),
#             nn.Conv2d(self.LL_ch, self.LL_ch, kernel_size=1),
#             nn.BatchNorm2d(self.LL_ch)
#         )
#         self.h_conv = nn.Sequential(
#             AxialDW(self.HH_ch, mixer_kernel=(7, 7)),
#             nn.Conv2d(self.HH_ch, self.H_out, kernel_size=1),
#             nn.BatchNorm2d(self.H_out)
#         )
#         self.l_conv = nn.Sequential(
#             AxialDW(self.LL_ch, mixer_kernel=(7, 7)),
#             nn.Conv2d(self.LL_ch, self.L_out, kernel_size=1),
#             nn.BatchNorm2d(self.L_out)
#         )
#         self.avgpool = nn.AdaptiveAvgPool2d(1)
#         self.maxpool = nn.AdaptiveMaxPool2d(1)
#         self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
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
#     def forward(self, x, skip):
#         LHx = self.LH(x[:, self.LL_ch:, :, :])
#         HHx = skip[:, :self.HH_ch, :, :]
#         HLx = self.HL(skip[:, self.HH_ch:, :, :])
#         LLx = x[:, :self.LL_ch, :, :]
#
#         x1 = HHx + LHx
#         x2 = LLx + HLx
#         out = torch.cat([self.h_conv(x1), self.up(self.l_conv(x2))], dim=1)
#         return self.gelu(out)
#
#
# class Mixadd_V0(nn.Module):
#     def __init__(self, in_c, out_c, num=0.5, mixer_kernel=(7, 7)):
#         super(Mixadd_V0, self).__init__()
#         self.HH_ch = int(out_c * num)
#         self.HL_ch = out_c - self.HH_ch
#         self.LL_ch = int(in_c * num)
#         self.LH_ch = in_c - self.LL_ch
#
#         self.LH = nn.Sequential(
#             nn.Conv2d(self.LH_ch, self.HH_ch, kernel_size=1),
#             nn.BatchNorm2d(self.HH_ch),
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         )
#         self.HL = nn.Sequential(
#             nn.AvgPool2d(2),
#             nn.Conv2d(self.HL_ch, self.LL_ch, kernel_size=1),
#             nn.BatchNorm2d(self.LL_ch)
#         )
#         self.conv = nn.Sequential(
#             AxialDW(self.HH_ch + self.LL_ch, mixer_kernel=(7, 7)),
#             nn.Conv2d(self.HH_ch + self.LL_ch, out_c, kernel_size=1),
#             nn.BatchNorm2d(out_c)
#         )
#         self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         self.gelu = nn.GELU()
#
#     def forward(self, x, skip):
#         LHx = self.LH(x[:, self.LL_ch:, :, :])
#         HHx = skip[:, :self.HH_ch, :, :]
#         HLx = self.HL(skip[:, self.HH_ch:, :, :])
#         LLx = x[:, :self.LL_ch, :, :]
#
#         x1 = HHx + LHx
#         x2 = LLx + HLx
#         y = torch.cat([x1, self.up(x2)], dim=1)
#         out = self.conv(y)
#         return self.gelu(out)
#
#
# class avgMixadd(nn.Module):
#     def __init__(self, in_c, out_c, num=0.5, mixer_kernel=(7, 7)):
#         super(avgMixadd, self).__init__()
#         self.HH_ch = int(out_c * num)
#         self.HL_ch = out_c - self.HH_ch
#         self.LL_ch = int(in_c * num)
#         self.LH_ch = in_c - self.LL_ch
#         self.H_out = int(out_c * num)
#         self.L_out = out_c - self.H_out
#
#         self.HH = nn.Sequential(
#             nn.Conv2d(self.HH_ch, self.H_out, kernel_size=1),
#             nn.BatchNorm2d(self.H_out)
#         )
#         self.LH = nn.Sequential(
#             nn.Conv2d(self.LH_ch, self.H_out, kernel_size=1),
#             nn.BatchNorm2d(self.H_out),
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         )
#         self.HL = nn.Sequential(
#             nn.AvgPool2d(2),
#             nn.Conv2d(self.HL_ch, self.L_out, kernel_size=1),
#             nn.BatchNorm2d(self.L_out)
#         )
#         self.LL = nn.Sequential(
#             nn.Conv2d(self.LL_ch, self.L_out, kernel_size=1),
#             nn.BatchNorm2d(self.L_out)
#         )
#         self.avgpool = nn.AdaptiveAvgPool2d(1)
#         self.maxpool = nn.AdaptiveMaxPool2d(1)
#         self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
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
#     def forward(self, x, skip):
#         LHx = self.LH(x[:, self.LL_ch:, :, :])
#         HHx = self.HH(skip[:, :self.HH_ch, :, :])
#         HLx = self.HL(skip[:, self.HH_ch:, :, :])
#         LLx = self.LL(x[:, :self.LL_ch, :, :])
#
#         out1, out2 = self.softmax_add(HHx, LHx)
#         out3, out4 = self.softmax_add(LLx, HLx)
#         x1 = out1 * HHx + out2 * LHx
#         x2 = out3 * LLx + out4 * HLx
#         # y = torch.cat([x1, self.up(x2)], dim=1)
#         # out = self.conv(y)
#         out = torch.cat([x1, self.up(x2)], dim=1)
#         return self.gelu(out)

