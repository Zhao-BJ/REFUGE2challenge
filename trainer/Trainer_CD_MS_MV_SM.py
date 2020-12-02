import os
import logging
import time
import visdom
import torch
from torch.autograd import Variable
from utils.trainer_utils import dt
from evaluator.metrics import dice_coeff_2label_withBW
from utils.seg_utils import draw_oc_od_boundary_from_validing


class Trainer:
    def __init__(self, model, optim, criterion, tloaders, vloaders, epoch_num,
                 exp_lr_scheduler, save_dir, visdom_env):
        self.model = model
        self.optim = optim
        self.criterion = criterion
        self.tloaders = tloaders
        self.vloaders = vloaders
        self.epoch_num = epoch_num
        self.exp_lr_scheduler = exp_lr_scheduler
        self.save_dir = save_dir
        self.vis = visdom.Visdom('localhost', env=visdom_env)

    def valid(self, epoch):
        logging.info('current lr:%s' % self.exp_lr_scheduler.get_lr())
        for k, v in self.vloaders.items():
            boundary_save_dir = self.save_dir + "/weights-" + str(epoch) + "-" + k + '/'
            if not os.path.exists(boundary_save_dir):
                os.makedirs(boundary_save_dir)
            with torch.no_grad():
                self.model.train(False)
                dice_oc_total = 0.0
                dice_od_total = 0.0
                count_total = 0
                for batch_idx, (img, mask, name) in enumerate(v):
                    img = Variable(img.cuda())
                    mask = Variable(mask.cuda())
                    map = self.model(img)
                    dice_oc_batch = 0.0
                    dice_od_batch = 0.0
                    for i in range(map.size(0)):
                        temp_map = map[i, :, :, :]
                        temp_mask = mask[i, :2, :, :]
                        dice_od, dice_oc = dice_coeff_2label_withBW(temp_map, temp_mask)
                        dice_oc_batch += dice_oc
                        dice_od_batch += dice_od
                        draw_oc_od_boundary_from_validing(img[i], map[i], boundary_save_dir, name[i])
                    dice_oc_total += dice_oc_batch
                    dice_od_total += dice_od_batch
                    count_total += map.size(0)
                oc_dice = dice_oc_total / count_total
                od_dice = dice_od_total / count_total
                self.vis.line(Y=[[oc_dice, od_dice]], X=[epoch], win=(k+'_dice'),
                    update=None if epoch == 1 else 'append', opts=dict(title=(k+'_dice'), legend=['oc', 'od']))
                logging.info("Epoch:%3d |OC dice:%.4f |OD dice:%.4f |Dataset:%s" % (epoch, oc_dice, od_dice, k))
        save_path = os.path.join(self.save_dir, 'weights-%d.pth' % epoch)
        torch.save(self.model.state_dict(), save_path)
        logging.info('saved model to %s' % save_path)

    def train(self):
        logging.info('==' * 30)
        logging.info(dt())
        start_time = time.time()
        for epoch in range(1, self.epoch_num+1):
            self.exp_lr_scheduler.step(epoch)
            self.model.train(True)
            for ite in range(300):
                self.optim.zero_grad()
                ite_loss_batch = 0.0
                for loader_idx in range(len(self.tloaders)):
                    loader = self.tloaders[loader_idx]
                    loader = enumerate(loader)
                    id_, (img, mask, name) = next(loader)
                    img = Variable(img.cuda())
                    mask = Variable(mask.cuda())
                    pred = self.model(img)
                    loader_loss_batch = 0.0
                    for i in range(pred.size(0)):
                        od_pred = pred[i, 0, :, :]
                        od_mask = mask[i, 0, :, :]
                        oc_pred = pred[i, 1, :, :]
                        oc_mask = mask[i, 1, :, :]
                        od_loss = self.criterion(torch.sigmoid(od_pred), od_mask)
                        loader_loss_batch += od_loss
                        oc_loss = self.criterion(torch.sigmoid(oc_pred), oc_mask)
                        loader_loss_batch += oc_loss
                    ite_loss_batch += loader_loss_batch
                ite_loss_batch.backward()
                self.optim.step()
                if (ite + 1) % 20 == 0:
                    print("Epoch:%3d |Batch_idx:%3d |Loss:%.3f" % (epoch, ite + 1, ite_loss_batch.item()))
            self.valid(epoch)
            logging.info('Time elapsed: %4.2f' % ((time.time() - start_time) / 60))
            logging.info('--' * 30)
