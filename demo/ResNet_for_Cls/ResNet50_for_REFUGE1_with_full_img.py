import os
import logging
import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor
from utils.trainer_utils import trainlog
from trainer.Trainer_Cls_basemodel import Trainer
from model import Res
from data.img_label_set import ILset
from utils.RAdam import RAdam


os.environ["CUDA_VISIBLE_DEVICES"] = "4"


# Hyper parameter
base_lr = 0.001
resume = None
epoch_num = 200
batch_size = 16
num_classes = 2
visdom_env = 'REFUGE2challenge/ResNet_for_Cls/ResNet50_for_REFUGE1_with_full_img'


# Data prepare
data_root = '/home/ubuntu/zhaobenjian/dataset/'
project_root = '/home/ubuntu/zhaobenjian/Challenge/'
train_dir = os.path.join(data_root, 'Glaucoma/REFUGE/resize512/REFUGE1/')
valid_dir = os.path.join(data_root, 'Glaucoma/REFUGE/resize512/REFUGE1/')
save_dir = os.path.join(project_root, 'REFUGE2challenge/demo/ResNet_for_Cls/results/ResNet50_for_REFUGE1_with_full_img')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
logfile = "%s/trainlog.log" % save_dir
trainlog(logfile)
transform = Compose([
    ToTensor()
])
tset = ILset(data_dir=train_dir, transform=transform)
vset = ILset(data_dir=train_dir, transform=transform)
tloader = DataLoader(dataset=tset, batch_size=batch_size, shuffle=True)
vloader = DataLoader(dataset=vset, batch_size=1, shuffle=False)


# Model
torch.manual_seed(1)
model = Res.resnet50(num_classes=num_classes, pretrained=True)
if resume:
    logging.info('resuming finetune from %s' % resume)
    model.load_state_dict(torch.load(resume))
model = model.cuda()
optim = RAdam(model.parameters(), lr=base_lr, betas=(0.9, 0.99))
criterion = torch.nn.CrossEntropyLoss()
criterion = criterion.cuda()
exp_lr_scheduler = lr_scheduler.StepLR(optim, step_size=100, gamma=0.8)

# Train
trainer = Trainer(
    model=model,
    optim=optim,
    criterion=criterion,
    tloader=tloader,
    vloader=vloader,
    epoch_num=epoch_num,
    exp_lr_scheduler=exp_lr_scheduler,
    save_dir=save_dir,
    visdom_env=visdom_env
)
trainer.train()
