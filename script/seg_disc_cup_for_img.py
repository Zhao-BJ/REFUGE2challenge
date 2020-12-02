import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.transforms import ToTensor, Compose
from PIL import Image
from model import CENet
from utils.seg_utils import BW_img
from data.img_set import Iset


os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# hyper parameters
num_classes = 2
data_root = ''
project_root = ''
resume = os.path.join(project_root,
        'REFUGE2challenge/demo/CENet_for_Seg/results/CENet_for_REFUGE1/CENet_for_REFUGE1.pth')


# data
data_type = '.jpg'
crop_img_dir = os.path.join(data_root, 'Glaucoma/REFUGE/crop512/test/')
crop_coord_dir = os.path.join(data_root, 'Glaucoma/REFUGE/crop512/test/coord/')
orig_img_dir = os.path.join(data_root, 'Glaucoma/REFUGE/original/REFUGE2-Test/')
map_org_save_dir = os.path.join(project_root,
        'REFUGE2challenge/demo/CENet_for_Seg/results/CENet_for_REFUGE1/CENet_for_REFUGE1-test')
if not os.path.exists(map_org_save_dir):
    os.makedirs(map_org_save_dir)
transform = Compose([ToTensor()])
img_set = Iset(data_dir=crop_img_dir, transform=transform)
img_loader = DataLoader(img_set, batch_size=1, shuffle=False)


# model
model = CENet.CENet(num_classes=num_classes)
if resume:
    model.load_state_dict(torch.load(resume))
model = model.cuda()


model.train(False)
with torch.no_grad():
    for i, (img, name) in enumerate(img_loader):
        print(name)
        img = Variable(img.cuda())
        pred = model(img)
        img = pred[0, :, :, :]
        img = torch.sigmoid(img)
        img = img.cpu().data.numpy()
        img = np.squeeze(img)
        img_od = np.array(BW_img(img[0, :, :], 0.5), dtype=int)
        img_oc = np.array(BW_img(img[1, :, :], 0.5), dtype=int)
        new_img = np.ones((img.shape[1], img.shape[1]), dtype=np.uint8) * 255
        new_img[img_od > 0.5] = 128
        new_img[img_oc > 0.5] = 0
        new_img = new_img.astype(np.uint8)

        org_img = Image.open(orig_img_dir+name[0])
        org_img = np.array(org_img)
        org_map = (np.ones((org_img.shape[0], org_img.shape[1])) * 255).astype(np.uint8)
        coord = np.loadtxt(crop_coord_dir+name[0][:-4]+'.txt', dtype=np.int64)
        org_map[coord[4]:coord[5], coord[6]:coord[7]] = new_img[coord[0]:coord[1], coord[2]:coord[3]]
        org_map = Image.fromarray(org_map, mode='L')
        org_map.save(map_org_save_dir + '/' + name[0][:-4] + ".png")
