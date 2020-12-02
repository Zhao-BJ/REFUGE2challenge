import scipy
import numpy as np
import torch
from skimage.measure import label, regionprops


# This loss combines a Sigmoid layer and BCELoss in one single class
bce = torch.nn.BCEWithLogitsLoss(reduction='none')


def BW_img(input_img, thresholding=None):
    if thresholding is not None:
        if input_img.max() > thresholding:
            binary = input_img > thresholding
        else:
            binary = input_img > input_img.max() / 2.0
    else:
        mid = (input_img.max() - input_img.min()) / 2.0
        binary = input_img > mid
    label_image = label(binary)
    regions = regionprops(label_image)
    area_list = []
    for region in regions:
        area_list.append(region.area)
    if area_list:
        idx_max = np.argmax(area_list)
        binary[label_image != idx_max+1] = 0
    return scipy.ndimage.binary_fill_holes(np.asarray(binary).astype(int))


def dice_coefficient_numpy(binary_segmentation, binary_gt_label):
    # Turn all variables to booleans
    binary_segmentation = np.asarray(binary_segmentation, dtype=np.bool)
    binary_gt_label = np.asarray(binary_gt_label, dtype=np.bool)

    # Convert the intersection
    intersection = np.logical_and(binary_segmentation, binary_gt_label)
    #intersection = np.sum(binary_segmentation * binary_gt_label)

    # Count the number of true pixels in the binary segmentation, gt_label and intersection
    segmentation_pixels = float(np.sum(binary_segmentation.flatten()))
    gt_label_pixels = float(np.sum(binary_gt_label.flatten()))
    intersection = float(np.sum(intersection.flatten()))

    # Compute the Dice coefficient
    dice_value = (2 * intersection + 1.0) / (1.0 + segmentation_pixels + gt_label_pixels)
    return dice_value


def dice_coeff(pred, target):
    target = target.data.cpu()
    #pred = torch.sigmoid(pred)
    pred = pred - pred.min()
    pred = pred / pred.max()
    pred = pred.data.cpu()
    pred[pred > 0.7] = 1
    pred[pred <= 0.7] = 0
    return dice_coefficient_numpy(pred, target)


def dice_coeff_withBW(pred, target):
    target = target.data.cpu()
    pred = torch.sigmoid(pred)
    pred = pred.data.cpu()
    target = np.array(target)
    target = BW_img(target, thresholding=0.7)
    pred = np.array(pred)
    pred = BW_img(pred, thresholding=0.7)
    return dice_coefficient_numpy(pred, target)


def dice_coeff_2label(pred, target):
    target = target.data.cpu()
    pred = torch.sigmoid(pred)
    pred = pred.data.cpu()
    pred[pred > 0.5] = 1
    pred[pred <= 0.5] = 0
    return dice_coefficient_numpy(pred[0, :, :], target[0, :, :]), \
           dice_coefficient_numpy(pred[1, :, :], target[1, :, :])


def dice_coeff_2label_withBW(pred, target):
    target = target.data.cpu()
    pred = torch.sigmoid(pred)
    pred = pred.data.cpu()
    target = np.array(target)
    target = BW_img(target, thresholding=0.5)
    pred = np.array(pred)
    pred = BW_img(pred, thresholding=0.5)
    return dice_coefficient_numpy(pred[0, :, :], target[0, :, :]), \
           dice_coefficient_numpy(pred[1, :, :], target[1, :, :])


def DiceLoss(pred, target):
    oc_dice, od_dice = dice_coeff_2label(pred, target)
    dice = 0.5 * oc_dice + 0.5 * od_dice
    loss = torch.tensor(1 - dice, requires_grad=True).float()
    return loss


def DiceLoss_withBW(input, target):
    return torch.tensor(1 - dice_coeff_withBW(input, target), requires_grad=True).float()
