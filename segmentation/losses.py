"""Segmentation training losses."""

import torch.nn.functional as F


def dice_loss(logits, target, classes):
    prob = F.softmax(logits.float(), dim=1)
    hot = F.one_hot(target, classes).permute(0, 3, 1, 2).float()
    inter = (prob * hot).sum((2, 3))
    union = prob.sum((2, 3)) + hot.sum((2, 3))
    return 1 - (2 * inter / (union + 1e-6)).mean()
