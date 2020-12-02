import numpy as np
import scipy
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import ToPILImage
from PIL import Image
from skimage.measure import label, regionprops
from skimage import io
from scipy import ndimage


def BW_img(input, thresholding):
    if input.max() > thresholding:
        binary = input > thresholding
    else:
        binary = input > input.max() / 2.0

    label_image = label(binary)
    regions = regionprops(label_image)
    area_list = []
    for region in regions:
        area_list.append(region.area)
    if area_list:
        idx_max = np.argmax(area_list)
        binary[label_image != idx_max + 1] = 0
    return scipy.ndimage.binary_fill_holes(np.asarray(binary).astype(int))


def disc_crop(org_img, DiscROI_size, C_x, C_y):
    tmp_size = int(DiscROI_size / 2);
    disc_region = np.zeros((DiscROI_size, DiscROI_size, 3), dtype=org_img.dtype)
    crop_coord = np.array([C_x - tmp_size, C_x + tmp_size, C_y - tmp_size, C_y + tmp_size], dtype=int)
    err_coord = [0, DiscROI_size, 0, DiscROI_size]
    if crop_coord[0] < 0:
        err_coord[0] = abs(crop_coord[0])
        crop_coord[0] = 0
    if crop_coord[2] < 0:
        err_coord[2] = abs(crop_coord[2])
        crop_coord[2] = 0
    if crop_coord[1] > org_img.shape[0]:
        err_coord[1] = err_coord[1] - (crop_coord[1] - org_img.shape[0])
        crop_coord[1] = org_img.shape[0]
    if crop_coord[3] > org_img.shape[1]:
        err_coord[3] = err_coord[3] - (crop_coord[3] - org_img.shape[1])
        crop_coord[3] = org_img.shape[1]
    disc_region[err_coord[0]:err_coord[1], err_coord[2]:err_coord[3], ] = org_img[crop_coord[0]:crop_coord[1],
                                                                          crop_coord[2]:crop_coord[3], ]
    return disc_region, err_coord, crop_coord


def save_pred_map(prob, save_dir, name):
    prob = torch.sigmoid(prob)
    prob = prob.cpu().numpy() * 255
    prob = np.squeeze(prob)
    prob = np.uint8(prob)
    img = Image.fromarray(prob)
    img.save(save_dir + name)


def save_seg_map_v2(prob, save_dir, name):
    prob = torch.sigmoid(prob)
    prob = prob.cpu()
    prob = ToPILImage()(prob)
    prob.save(save_dir + name)


def save_coord_map(img, coords, save_dir, name):
    img[coords[0] - 3:coords[0] + 3, coords[1] - 3:coords[1] + 3, 0] = 1
    img[coords[0] - 3:coords[0] + 3, coords[1] - 3:coords[1] + 3, 1:] = 0
    img = np.uint8(img * 255)
    io.imsave(save_dir + "/" + name, img)


def save_od_fov_coord_map(img, od_coords, fov_coords, save_dir, name):
    img[od_coords[0] - 3:od_coords[0] + 3, od_coords[1] - 3:od_coords[1] + 3, 0] = 1
    img[od_coords[0] - 3:od_coords[0] + 3, od_coords[1] - 3:od_coords[1] + 3, 1:] = 0

    img[fov_coords[0] - 3:fov_coords[0] + 3, fov_coords[1] - 3:fov_coords[1] + 3, 0] = 0
    img[fov_coords[0] - 3:fov_coords[0] + 3, fov_coords[1] - 3:fov_coords[1] + 3, 1] = 1
    img[fov_coords[0] - 3:fov_coords[0] + 3, fov_coords[1] - 3:fov_coords[1] + 3, 2] = 0

    img = np.uint8(img * 255)
    io.imsave(save_dir + "/" + name, img)


def draw_oc_od_boundary_from_validing(img, mask, save_dir, name, width=1):
    mask = torch.sigmoid(mask)
    mask = mask.cpu().data.numpy()
    od = mask[0, :, :]
    od = BW_img(od, 0.5)
    od = np.uint(od)
    dila_od = ndimage.binary_dilation(od, iterations=width).astype(od.dtype)
    eros_od = ndimage.binary_erosion(od, iterations=width).astype(od.dtype)
    od = dila_od + eros_od
    oc = mask[1, :, :]
    oc = BW_img(oc, 0.5)
    oc = np.uint(oc)
    dila_oc = ndimage.binary_dilation(oc, iterations=width).astype(oc.dtype)
    eros_oc = ndimage.binary_erosion(oc, iterations=width).astype(oc.dtype)
    oc = dila_oc + eros_oc
    od[od == 2] = 0
    oc[oc == 2] = 0
    boundary = (od + oc) > 0

    img = img.cpu().data.numpy() * 255
    img = np.transpose(img, (1, 2, 0)).astype(np.uint8)
    img[boundary == True, 0] = 255
    img[boundary == True, 1:] = 0
    io.imsave(save_dir + name, img)