import os
import pandas as pd
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor
from data.img_set import Iset
from model import Res


os.environ["CUDA_VISIBLE_DEVICES"] = "2"


# Hyper parameters
num_classes = 2
data_root = '/home/ubuntu/zhaobenjian/dataset/'
project_root = '/home/ubuntu/zhaobenjian/Challenge/'
resume = os.path.join(project_root,
        'REFUGE2challenge/demo/ResNet_for_Cls/results/ResNet50_with_ONH_img_v2/ResNet50_with_ONH_img.pth')


# Data
valid_dir = os.path.join(data_root, 'Glaucoma/REFUGE/crop512/test/')
save_dir = os.path.join(project_root,
        'REFUGE2challenge/demo/ResNet_for_Cls/results/ResNet50_with_ONH_img_v2/ResNet50_with_ONH_img-test')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
transform = Compose([ToTensor()])
vset = Iset(data_dir=valid_dir, transform=transform)
vloader = DataLoader(dataset=vset, batch_size=1, shuffle=False)

# Model
model = Res.resnet50(num_classes=2, pretrained=True)
if resume:
    model.load_state_dict(torch.load(resume))
model = model.cuda()

# Computing
model.train(False)
img_name = []
prob0 = []
prob1 = []
with torch.no_grad():
    for batchidx, (img, name) in enumerate(vloader):
        print(name)
        img = Variable(img.cuda())
        img_size = img.size(0)
        prob = model(img)
        prob = F.softmax(prob, dim=1)
        img_name.append(name)
        for i in range(img.size(0)):
            prob0.append(prob[i, 0].item())
            prob1.append(prob[i, 1].item())
print(len(img_name))
pddata = pd.DataFrame({"FileName": img_name, "Glaucoma Risk": prob1})
pddata.to_csv(save_dir + "/classification_results.csv")
