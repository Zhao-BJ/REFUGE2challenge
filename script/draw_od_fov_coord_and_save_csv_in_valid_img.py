import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.transforms import Compose, ToTensor
from PIL import Image
from model import UNet
from data.img_set import Iset
from utils.coords_utils import get_peak_coordinates, determine_od
from utils.seg_utils import save_od_fov_coord_map


os.environ['CUDA_VISIBLE_DEVICES'] = '3'


# hyper parameters
batch_size = 1
num_classes = 1
data_root = ''
project_root = ''
resume = os.path.join(project_root, 'REFUGE2challenge/demo/OD_Fovea_Reg/results/UNet/UNet_for_OD_Fovea_Reg.pth')


# data
pro_img_dir = os.path.join(data_root, 'Glaucoma/REFUGE/resize512/test/')
org_img_dir = os.path.join(data_root, 'Glaucoma/REFUGE/original/REFUGE2-Test/')
pred_map_save_dir = os.path.join(project_root, 'REFUGE2challenge/demo/OD_Fovea_Reg/results/UNet/UNet_for_OD_Fovea_Reg-test-draw')
org_fov_coord_dir = os.path.join(project_root, 'REFUGE2challenge/demo/OD_Fovea_Reg/results/UNet/UNet_for_OD_Fovea_Reg-test/fovea_location_results.csv')
if not os.path.exists(pred_map_save_dir):
    os.makedirs(pred_map_save_dir)
if not os.path.exists(org_fov_coord_dir):
    os.makedirs(org_fov_coord_dir)
transform = Compose([ToTensor()])
data_set = Iset(data_dir=pro_img_dir, transform=transform)
data_loader = DataLoader(dataset=data_set, batch_size=batch_size, shuffle=False)


# model
model = UNet.UNet(num_classes=num_classes)
if resume:
    model.load_state_dict(torch.load(resume))
model = model.cuda()


# draw and save coords
img_name = []
fov_x = []
fov_y = []
model.train(False)
with torch.no_grad():
    for batch_idx, (img, name) in enumerate(data_loader):
        img = Variable(img.cuda())
        pred = model(img)
        for i in range(pred.size(0)):
            print('processing the image %s' % name[i])
            temp_pred = pred[i, 0, :, :]
            temp_pred = temp_pred - temp_pred.min()
            temp_pred = temp_pred / temp_pred.max()
            temp_pred = temp_pred.cpu().data.numpy()
            peak_coords = get_peak_coordinates(temp_pred, threshold=0.2)

            img = img[i, :, :, :]
            img = img.cpu().data.numpy()
            img = np.transpose(img, (1, 2, 0))
            od_coords, fov_coords = determine_od(img, peak_coords, neigh=9)
            save_od_fov_coord_map(img, od_coords, fov_coords, pred_map_save_dir, name[i])

            org_img = Image.open(org_img_dir + name[i])
            org_img = np.array(org_img)

            rescale_x_factor = np.float(org_img.shape[0] / img.shape[0])
            rescale_y_factor = np.float(org_img.shape[1] / img.shape[1])

            org_fov_x = fov_coords[1] * rescale_x_factor
            org_fov_y = fov_coords[0] * rescale_y_factor
            img_name.append(name[i])
            fov_x.append(org_fov_x)
            fov_y.append(org_fov_y)
fovea_location_results = pd.DataFrame({'ImageName': img_name, 'Fovea_X': fov_x, 'Fovea_Y': fov_y})
fovea_location_results.to_csv(org_fov_coord_dir)
