import numpy as np
import torch


class F1:
    __name__ = 'F1 macro'
    def __init__(self, n=1):
        self.n = n
        self.TP = 0
        self.FP = 0
        self.FN = 0

    def __call__(self, preds, targs, th=0.0):
        preds = (preds > th).int()
        targs = targs.int()
        self.TP += (preds * targs).float().sum()
        self.FP += (preds > targs).float().sum()
        self.FN += (preds < targs).float().sum()

    def reset(self):
        score = (2.0 * self.TP / (2.0 * self.TP + self.FP + self.FN + 1e-6))
        mIoU = self.TP / (self.TP + self.FP + self.FN + 1e-6)
        print('F1 macro:{}--mIoU:{}'.format(score.mean(), mIoU), flush=True)
        self.TP = 0
        self.FP = 0
        self.FN = 0


def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    iou = (intersection + smooth) / (union + smooth)
    dice = (2 * iou) / (iou+1)
    return iou, dice

