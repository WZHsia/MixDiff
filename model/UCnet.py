""" Full assembly of the parts to form the complete network """
from torch.utils.checkpoint import checkpoint
from .model_parts import *


# input_size = (256, 256, 1)
class UCNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, num=0.5, bilinear=True):
        super(UCNet, self).__init__()
        self.checkpoint = False
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = avgAdd(512, 512, 256, num)
        self.up2 = avgAdd(256, 256, 128, num)
        self.up3 = avgAdd(128, 128, 64, num)
        self.up4 = avgAdd(64, 64, 64, num)
        self.outc = OutConv(64, n_classes)

    # x = (5, 1, 256, 256)
    def forward(self, x):
        x1 = self.inc(x)        # (5, 16, 256, 256)
        x2 = self.down1(x1)     # (5, 32, 128, 128)
        x3 = self.down2(x2)     # (5, 64, 64, 64)
        x4 = self.down3(x3)     # (5, 128, 32, 32)
        x5 = self.down4(x4)     # (5, 256, 16, 16)
        x = self.up1(x5, x4)    # (4, 128, 32, 32)
        x = self.up2(x, x3)     # (4, 64, 64, 64)
        x = self.up3(x, x2)     # (4, 32, 128, 128)
        x = self.up4(x, x1)  # (4, 16, 256, 256)
        logits = self.outc(x)   # (4, 1, 256, 256)
        return logits

    def use_checkpointing(self):
        self.checkpoint = True


class UANet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, bilinear=True, num0=0.5, num1=0.5):
        super(UANet, self).__init__()
        self.checkpoint = False
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = avgMixAdd(512, 512, 256, num0, num1)
        self.up2 = avgMixAdd(256, 256, 128, num0, num1)
        self.up3 = avgMixAdd(128, 128, 64, num0, num1)
        self.up4 = avgMixAdd(64, 64, 64, num0, num1)
        self.outc = OutConv(64, n_classes)

    # x = (5, 1, 256, 256)
    def forward(self, x):
        x1 = self.inc(x)        # (5, 16, 256, 256)
        x2 = self.down1(x1)     # (5, 32, 128, 128)
        x3 = self.down2(x2)     # (5, 64, 64, 64)
        x4 = self.down3(x3)     # (5, 128, 32, 32)
        x5 = self.down4(x4)     # (5, 256, 16, 16)
        x = self.up1(x5, x4)    # (4, 128, 32, 32)
        x = self.up2(x, x3)     # (4, 64, 64, 64)
        x = self.up3(x, x2)     # (4, 32, 128, 128)
        x = self.up4(x, x1)  # (4, 16, 256, 256)
        logits = self.outc(x)   # (4, 1, 256, 256)
        return logits

    def use_checkpointing(self):
        self.checkpoint = True


class UGNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, num=0.5, bilinear=True):
        super(UGNet, self).__init__()
        self.checkpoint = False
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = MixAdd(512, 512, 256, num)
        self.up2 = MixAdd(256, 256, 128, num)
        self.up3 = MixAdd(128, 128, 64, num)
        self.up4 = MixAdd(64, 64, 64, num)
        self.outc = OutConv(64, n_classes)

    # x = (5, 1, 256, 256)
    def forward(self, x):
        x1 = self.inc(x)        # (5, 16, 256, 256)
        x2 = self.down1(x1)     # (5, 32, 128, 128)
        x3 = self.down2(x2)     # (5, 64, 64, 64)
        x4 = self.down3(x3)     # (5, 128, 32, 32)
        x5 = self.down4(x4)     # (5, 256, 16, 16)
        x = self.up1(x5, x4)    # (4, 128, 32, 32)
        x = self.up2(x, x3)     # (4, 64, 64, 64)
        x = self.up3(x, x2)     # (4, 32, 128, 128)
        x = self.up4(x, x1)  # (4, 16, 256, 256)
        logits = self.outc(x)   # (4, 1, 256, 256)
        return logits

    def use_checkpointing(self):
        self.checkpoint = True


class UDNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, bilinear=True, num=0.5):
        super(UDNet, self).__init__()
        self.checkpoint = False
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = DiffAdd(512, 512, 256, num)
        self.up2 = DiffAdd(256, 256, 128, num)
        self.up3 = DiffAdd(128, 128, 64, num)
        self.up4 = DiffAdd(64, 64, 64, num)
        self.outc = OutConv(64, n_classes)

    # x = (5, 1, 256, 256)
    def forward(self, x):
        x1 = self.inc(x)        # (5, 16, 256, 256)
        x2 = self.down1(x1)     # (5, 32, 128, 128)
        x3 = self.down2(x2)     # (5, 64, 64, 64)
        x4 = self.down3(x3)     # (5, 128, 32, 32)
        x5 = self.down4(x4)     # (5, 256, 16, 16)
        x = self.up1(x5, x4)    # (4, 128, 32, 32)
        x = self.up2(x, x3)     # (4, 64, 64, 64)
        x = self.up3(x, x2)     # (4, 32, 128, 128)
        x = self.up4(x, x1)  # (4, 16, 256, 256)
        logits = self.outc(x)   # (4, 1, 256, 256)
        return logits

    def use_checkpointing(self):
        self.checkpoint = True

