import os
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD, lr_scheduler
from torchvision.transforms import Compose, ToTensor
from utils.trainer_utils import trainlog
from data.img_mask_set import IMset
from model import CENet
from trainer.Trainer_CD_MS_MV_SM import Trainer
from utils.RAdam import RAdam


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# Hyper paramter
REFUGE1_batch_size = 4
num_classes = 2
lr = 0.001
epoch_num = 200
data_root = ''
project_root = ''
resume = None
visdom_env = 'REFUGE2challenge/CENet_for_Seg/CENet_for_REFUGE1'


# Data
REFUGE1_dir = os.path.join(data_root, 'Glaucoma/REFUGE/crop512/REFUGE1/')
save_dir = os.path.join(project_root, 'REFUGE2challenge/demo/CENet_for_Seg/results/CENet_for_REFUGE1')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
logfile = "%s/trainlog.log" % save_dir
trainlog(logfile)
transform = Compose([ToTensor()])
REFUGE1_set = IMset(data_dir=REFUGE1_dir, transform=transform)
REFUGE1_loader = DataLoader(dataset=REFUGE1_set, batch_size=REFUGE1_batch_size, shuffle=True)
tloaders = [REFUGE1_loader]
vloaders = {"REFUGE1": REFUGE1_loader}


# Model
model = CENet.CENet(num_classes=num_classes)
if resume:
    logging.info('resuming fintune from %s' % resume)
    model.load_state_dict(torch.load(resume))
model = model.cuda()
criterion = nn.BCELoss()
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
