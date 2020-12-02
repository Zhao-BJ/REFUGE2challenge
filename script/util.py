import logging
import datetime
import numpy as np
import scipy
from skimage.measure import label, regionprops


def dt():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def trainlog(logfilepath, head='%(message)s'):
    logger = logging.getLogger('mylogger')       # Return a logger with specified name
    logging.basicConfig(filename=logfilepath, level=logging.INFO, format=head)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(head)
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def img_center_cut(org_img, cut_size, channel=3):
    """
    Cut the central area of the original image according to the cut_size.
    :param org_img: orginial image, should be PIL.Image type.
    :param cut_size: image size after cutting.
    :param channel: image channel.
    :return: the cutted image
    """
    org_img = np.array(org_img)
    cutted_img = np.zeros((cut_size, cut_size, channel), dtype=org_img.dtype)
    cut_coord_left = int(org_img.shape[1] / 2 - cut_size / 2)
    cut_coord_right = int(org_img.shape[1] / 2 + cut_size / 2)
    cut_coord_top = int(org_img.shape[0] / 2 - cut_size / 2)
    cut_coord_bottom = int(org_img.shape[0] / 2 + cut_size / 2)
    cutted_img[0:cut_size-1, 0:cut_size-1, ] \
        = org_img[cut_coord_left:cut_coord_right-1, cut_coord_top:cut_coord_bottom-1, ]
    return Image.fromarray(cutted_img)


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


def disc_crop(org_img, DiscROI_size, C_x, C_y, img_mode='RGB'):
    tmp_size = int(DiscROI_size / 2)
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
    if img_mode == 'RGB':
        disc_region = np.zeros((DiscROI_size, DiscROI_size, 3), dtype=org_img.dtype)
        disc_region[err_coord[0]:err_coord[1], err_coord[2]:err_coord[3], ] = org_img[crop_coord[0]:crop_coord[1],
                                                                            crop_coord[2]:crop_coord[3], ]
    elif img_mode == 'L':
        disc_region = np.zeros((DiscROI_size, DiscROI_size), dtype=org_img.dtype)
        disc_region[err_coord[0]:err_coord[1], err_coord[2]:err_coord[3]] = org_img[crop_coord[0]:crop_coord[1],
                                                                              crop_coord[2]:crop_coord[3]]
    return disc_region, err_coord, crop_coord


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from PIL import Image

    img_path = '/home/ubuntu/zhaobenjian/dataset/AMD/original/Annotation-DF-Training400/Training400/Disc_Fovea_Illustration/A0001.jpg'

    img = Image.open(img_path)
    img = img_center_cut(img, 448)
    plt.imshow(img)
    img = np.array(img)
    print(img.shape)