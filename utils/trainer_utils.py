import cv2
import logging
import datetime
import numpy as np
import torch
from torchvision.transforms import functional as TF
from skimage import io


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


def norm_image(image):
    """
    norm image
    :param image: [H, W, C]
    :return:
    """
    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    image *= 255
    return np.uint8(image)


def draw_GAP_featmap_with_threshold(module, pred, feat, img, save_dir, name, mark, threshold=64):
    img_size = img.size(2)
    img = img.cpu().numpy()
    img = np.float32(img)

    weight = list(module.weight)
    weight = weight[pred]
    weight = weight.cpu().data.numpy()
    feat = feat.cpu().data.numpy()
    feat = np.squeeze(feat)
    cam = feat * weight[:, np.newaxis, np.newaxis]
    cam = np.sum(cam, axis=0)

    cam = np.maximum(cam, 0)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    cam_img = np.uint8(255 * cam_img)
    cam_img[cam_img <= threshold] = 0
    cam_img = cv2.resize(cam_img, (img_size, img_size))
    heatmap = cv2.applyColorMap(np.uint8(cam_img), cv2.COLORMAP_JET)
    heatmap = heatmap[..., ::-1]
    heatmap = np.float32(heatmap) / 255

    img = np.squeeze(img)
    img = np.transpose(img, (1, 2, 0))
    imgheatmap = heatmap + img
    imgheatmap = norm_image(imgheatmap)
    heatmap = np.uint8(heatmap * 255)
    io.imsave(save_dir + "{}-{}.jpg".format(name[:-4], mark), imgheatmap)
    io.imsave(save_dir + "{}-{}-{}.jpg".format(name[:-4], mark, "heatmap"), heatmap)


def draw_GAP_featmap(module, pred, feat, img, save_dir, name):
    img_size = img.size(2)
    img = img.cpu().numpy()
    img = np.float32(img)

    weight = list(module.weight)
    weight = weight[pred]
    weight = weight.cpu().data.numpy()
    feat = feat.cpu().data.numpy()
    feat = np.squeeze(feat)
    cam = feat * weight[:, np.newaxis, np.newaxis]
    cam = np.sum(cam, axis=0)

    cam = np.maximum(cam, 0)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    cam_img = np.uint8(255 * cam_img)
    cam_img = cv2.resize(cam_img, (img_size, img_size))
    heatmap = cv2.applyColorMap(np.uint8(cam_img), cv2.COLORMAP_JET)
    heatmap = heatmap[..., ::-1]
    heatmap = np.float32(heatmap) / 255

    img = np.squeeze(img)
    img = np.transpose(img, (1, 2, 0))
    imgheatmap = heatmap * 0.3 + img * 0.7
    imgheatmap = norm_image(imgheatmap)
    heatmap = np.uint8(heatmap * 255)
    io.imsave(save_dir + "{}.jpg".format(name[:-4]), imgheatmap)
    io.imsave(save_dir + "{}-{}.jpg".format(name[:-4], "heatmap"), heatmap)


def draw_oam(pred, feat, img, save_dir, name):
    img_size = img.size(2)
    img = img.cpu().numpy()
    img = np.float32(img)

    feat = feat[pred]
    feat = feat.cpu().data.numpy()
    feat = np.squeeze(feat)

    cam = np.maximum(feat, 0)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    cam_img = np.uint8(255 * cam_img)
    cam_img = cv2.resize(cam_img, (img_size, img_size))
    heatmap = cv2.applyColorMap(np.uint8(cam_img), cv2.COLORMAP_JET)
    heatmap = heatmap[..., ::-1]
    heatmap = np.float32(heatmap) / 255

    img = np.squeeze(img)
    img = np.transpose(img, (1, 2, 0))
    imgheatmap = heatmap * 0.3 + img * 0.7
    imgheatmap = norm_image(imgheatmap)
    heatmap = np.uint8(heatmap * 255)
    io.imsave(save_dir + "{}.jpg".format(name[:-4]), imgheatmap)
    io.imsave(save_dir + "{}-{}.jpg".format(name[:-4], "heatmap"), heatmap)


def get_cam(module, pred, feat):
    weight = list(module.weight)
    weight = weight[pred]
    feat = torch.squeeze(feat)
    cam = feat * weight[:, np.newaxis, np.newaxis]
    cam = torch.sum(cam, dim=0)
    zero_tensor = torch.zeros(size=cam.size()).cuda()
    cam = torch.max(cam, other=zero_tensor)
    cam = cam - torch.min(cam)
    cam = cam / torch.max(cam)
    return cam


def get_featmap(module, pred, feat, img, save_dir, name):
    img_size = img.size(2)

    weight = list(module.weight)
    weight = weight[pred]
    weight = weight.cpu().data.numpy()
    feat = feat.cpu().data.numpy()
    feat = np.squeeze(feat)
    cam = feat * weight[:, np.newaxis, np.newaxis]
    cam = np.sum(cam, axis=0)

    cam = np.maximum(cam, 0)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    cam_img = np.uint8(255 * cam_img)
    cam_img = cv2.resize(cam_img, (img_size, img_size))
    io.imsave(save_dir + "{}-{}.jpg".format(name[:-4], "featmap"), cam_img)
