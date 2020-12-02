import os
import logging
import time
import visdom
import torch
from torch.autograd import Variable
from utils.trainer_utils import dt
from utils.seg_utils import save_pred_map


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
            pred_save_dir = self.save_dir + "/weights-" + str(epoch) + "-" + k + '/'
            if not os.path.exists(pred_save_dir):
                os.makedirs(pred_save_dir)
            with torch.no_grad():
                self.model.train(False)
                for batch_idx, (img, mask, name) in enumerate(v):
                    img = Variable(img.cuda())
                    pred = self.model(img)
                    for i in range(pred.size(0)):
                        save_pred_map(pred[i, :, :, :], pred_save_dir, name[i])
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
                    loader_loss_batch = self.criterion(pred, mask)
                    ite_loss_batch += loader_loss_batch
                ite_loss_batch.backward()
                self.optim.step()
                if (ite + 1) % 20 == 0:
                    print("Epoch:%3d |Batch_idx:%3d |Loss:%.3f" % (epoch, ite + 1, ite_loss_batch.item()))
            self.valid(epoch)
            logging.info('Time elapsed: %4.2f' % ((time.time() - start_time) / 60))
            logging.info('--' * 30)
