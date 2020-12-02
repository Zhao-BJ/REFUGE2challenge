import os
import time
import logging
import visdom
import numpy as np
import torch
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score
from utils.trainer_utils import dt


class Trainer:
    def __init__(self, model, optim, criterion, tloader, vloader, epoch_num, exp_lr_scheduler, save_dir,
                 visdom_env):
        self.model = model
        self.optim = optim
        self.criterion = criterion
        self.tloader = tloader
        self.vloader = vloader
        self.epoch_num = epoch_num
        self.exp_lr_scheduler = exp_lr_scheduler
        self.save_dir = save_dir
        self.vis = visdom.Visdom('localhost', env=visdom_env)

    def valid(self, epoch):
        logging.info('current lr:%s' % self.exp_lr_scheduler.get_lr())
        with torch.no_grad():
            self.model.train(False)
            loss_total = 0.0
            acc_total = 0
            count_total = 0
            x_list = []
            y_list = []
            for batch_idx, (img, label, name) in enumerate(self.vloader):
                img = Variable(img.cuda())
                label = Variable(torch.from_numpy(np.array(label)).long().cuda())
                count_total += img.size(0)
                prob = self.model(img)
                loss = self.criterion(prob, label)
                loss_total += loss.item()
                _, pred = torch.max(prob, 1)
                batch_corrects = torch.sum((pred == label)).item()
                acc_total += batch_corrects
                for batch_idx in range(label.size(0)):
                    x_list.append(prob[batch_idx, 1].item())
                    y_list.append(label[batch_idx].item())
            loss = loss_total / count_total
            acc = 1.0 * float(acc_total) / count_total
            auc = roc_auc_score(y_list, x_list)
            logging.info("Epoch:%3d |ACC:%.4f |AUC:%.4f" % (epoch, acc, auc))
            self.vis.line(Y=[[acc, auc]], X=[epoch], win='acc_auc', update=None if epoch == 1 else 'append',
                          opts=dict(title='acc_auc', legend=['acc', 'auc']))
            self.vis.line(Y=[loss], X=[epoch], win='loss', update=None if epoch == 1 else 'append',
                          opts=dict(title='loss'))
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
            for batchidx, (img, label, name) in enumerate(self.tloader):
                img = Variable(img.cuda())
                label = Variable(torch.from_numpy(np.array(label)).long().cuda())
                self.optim.zero_grad()
                prob = self.model(img)
                loss = self.criterion(prob, label)
                loss.backward()
                self.optim.step()
                if (batchidx + 1) % 20 == 0:
                    print("Epoch:%3d |Batchidx:%3d |Loss:%.3f" % (epoch, batchidx + 1, loss.item()))
            self.valid(epoch)
            logging.info('Time elapsed: %4.2f' % ((time.time() - start_time) / 60))
            logging.info('--' * 30)
