import os
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD, lr_scheduler
from torchvision.transforms import Compose, ToTensor
from utils.trainer_utils import trainlog
from data.img_mask_set import IMset
from model import UNet
from trainer.Trainer_FD_MS_MV_SM import Trainer
from utils.RAdam import RAdam


os.environ["CUDA_VISIBLE_DEVICES"] = "5"


# Hyper paramter
REFUGE1_batch_size = 4
num_classes = 1
lr = 0.001
epoch_num = 200
resume = None
visdom_env = 'REFUGE2challenge/OD_Fovea_Reg/UNet'


# Data
data_root = ''
project_root = ''
REFUGE1_dir = os.path.join(data_root, 'Glaucoma/REFUGE/resize512/REFUGE1/')
save_dir = os.path.join(project_root, 'REFUGE2challenge/demo/OD_Fovea_Reg/results/UNet')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
logfile = "%s/trainlog.log" % save_dir
trainlog(logfile)
transform = Compose([ToTensor()])
REFUGE1_set = IMset(data_dir=REFUGE1_dir, transform=transform, mask='reg_mask', mask_mode='L')
REFUGE1_loader = DataLoader(dataset=REFUGE1_set, batch_size=REFUGE1_batch_size, shuffle=True)
tloaders = [REFUGE1_loader]
vloaders = {"REFUGE1": REFUGE1_loader}


# Model
model = UNet.UNet(num_classes=num_classes)
if resume:
    logging.info('resuming fintune from %s' % resume)
    model.load_state_dict(torch.load(resume))
model = model.cuda()
criterion = nn.MSELoss()
#optim = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
optim = RAdam(model.parameters(), lr=lr, betas=(0.9, 0.99))
exp_lr_scheduler = lr_scheduler.StepLR(optim, step_size=20, gamma=0.1)


# Train
trainer = Trainer(
    model=model,
    optim=optim,
    criterion=criterion,
    tloaders=tloaders,
    vloaders=vloaders,
    epoch_num=epoch_num,
    exp_lr_scheduler=exp_lr_scheduler,
    save_dir=save_dir,
    visdom_env=visdom_env
)
trainer.train()
