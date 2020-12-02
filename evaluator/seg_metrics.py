import torch
import torch.nn as nn


def dice_score(true, pred):
    smooth = 0.0
    i = torch.sum(true)
    j = torch.sum(pred)
    intersection = torch.sum(true * pred)
    score = (2. * intersection + smooth) / (i + j + smooth)
    return score


def dice_score_1label(true, pred, threshold=0.5):
    #pred = torch.sigmoid(pred)
    pred[pred > threshold] = 1
    pred[pred <= threshold] = 0
    return dice_score(true, pred)


def dice_score_2label(true, pred, threshold=0.5):
    pred = torch.sigmoid(pred)
    pred[pred > threshold] = 1
    pred[pred <= threshold] = 0
    return dice_score(true[0, :, :], pred[0, :, :]), dice_score(true[1, :, :], pred[1, :, :])


def dice_loss(true, pred, threshold=0.5):
    loss = 1 - dice_score_1label(true, pred, threshold).item()
    return loss


# class DiceLoss(object):
#     def __init__(self, threshold=0.5):
#         self.threshold = threshold
#
#     def __call__(self, true, pred):
#         loss = 1 - dice_score_1label(true, pred, self.threshold).item()
#         return loss


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        smooth = 1

        input_flat = input.view(1, -1)
        target_flat = target.view(1, -1)

        intersection = input_flat * target_flat

        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum()

        return loss
