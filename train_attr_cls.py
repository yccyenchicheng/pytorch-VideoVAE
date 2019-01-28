import os
import sys
import time
import logging
import argparse
import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from src.datasets import WeizmannActionClassificationDataset
from src.model import VideoVAE, Classifier

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--use_cuda', action='store_true')
parser.add_argument('--exp', type=str, default='exp_cls{}'.format(time.strftime("%m%d")))
args = parser.parse_args()

def accuracy(model, loader, args):
    model.eval()
    n_data = len(loader) * loader.batch_size
    correct_act = 0
    correct_id = 0
    for batch_ix, data in enumerate(loader):
        img, act_y, id_y, action, identity = data
        if args.use_cuda:
            img = img.cuda()
            act_y = act_y.cuda()
            id_y = id_y.cuda()

        out_act, out_id = model(img)
        
        _, pred_act = out_act.max(1)
        _, pred_id = out_id.max(1)

        correct_act += (pred_act == act_y).sum().item()
        correct_id  += (pred_id == id_y).sum().item()

    acc_act = correct_act / n_data
    acc_id = correct_id / n_data

    return acc_act, acc_id

class RunningAverageMeter(object):
    def __init__(self, momentum=0.9):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.val = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)

        self.val = val

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)
    with open(filepath, "r") as f:
        logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger

def make_dirs(d):
    if not os.path.exists(d):
        os.makedirs(d)

if __name__ == "__main__":
    # exp
    args.exp = os.path.join('ExperimentAttr', args.exp)
    make_dirs(args.exp)
    make_dirs(os.path.join(args.exp, 'checkpoint'))

    # log
    logger = get_logger(logpath=os.path.join(args.exp, 'logs'), filepath=__file__)
    tboard_dir = os.path.join(args.exp, 'tboard')
    writer = SummaryWriter(log_dir=tboard_dir)

    args.logger = logger
    args.writer = writer
    
    # params
    batch_size = args.batch_size
    lr = args.lr
    epochs = 20
    log_interval = 15
    # cls
    in_c = 3
    z_dim = 512
    h_dim = 512
    n_act = 10
    n_id = 9

    # data
    trfs = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    train_set = WeizmannActionClassificationDataset(root='data', train=True, transform=trfs)
    test_set = WeizmannActionClassificationDataset(root='data', train=False, transform=trfs)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    model = Classifier(in_c=in_c, z_dim=z_dim, h_dim=128, n_act=n_act, n_id=n_id)

    crit_act = nn.CrossEntropyLoss()
    crit_id = nn.CrossEntropyLoss()

    if args.use_cuda:
        model.cuda()
        crit_act.cuda()
        crit_id.cuda()

    opt = optim.Adam(model.parameters(), lr=lr)
    
    batch_timer = RunningAverageMeter()
    end = time.time()

    for epoch_ix in range(epochs):
        # train
        model.train()
        for batch_ix, data in enumerate(train_loader):
            img, act_y, id_y, action, identity = data
            if args.use_cuda:
                img = img.cuda()
                act_y = act_y.cuda()
                id_y = id_y.cuda()

            out_act, out_id = model(img)
            loss_act = crit_act(out_act, act_y)
            loss_id = crit_id(out_id, id_y)
            loss = loss_act + loss_id
            
            opt.zero_grad()
            loss.backward()
            opt.step()

            batch_timer.update(time.time() - end)
            end = time.time()

            if batch_ix % log_interval == 0:
                niter = epoch_ix * len(train_loader) + batch_ix
                logger.info("Epoch: {} | [{}/{}] Time: {:.4f} ({:.4f}) | Loss: {:.4f}".format(
                    epoch_ix, batch_ix, len(train_loader), batch_timer.val, batch_timer.avg, loss.item()
                ))
                writer.add_scalar("Train/Loss", loss.item(), niter)

        # test
        niter = (epoch_ix+1) * len(train_loader)
        tr_act_acc, tr_id_acc = accuracy(model, train_loader, args)
        tt_act_acc, tt_id_acc = accuracy(model, test_loader, args)
        logger.info("Epoch: {} | [{}/{}] Time: {:.4f} ({:.4f}) | Loss: {:.4f}".format(
                    epoch_ix, batch_ix, len(train_loader), batch_timer.val, batch_timer.avg, loss.item()
                ))

        logger.info("Epoch: {} | Train: Act Acc {:.4f} Id Acc {:.4f} | Test: Act Acc {:.4f} Id Acc {:.4f}".format(
            epoch_ix, tr_act_acc, tr_id_acc, tt_act_acc, tt_id_acc))
            
        writer.add_scalar("Train/ActionAccuracy", tr_act_acc, niter)
        writer.add_scalar("Train/IdentityAccuracy", tr_id_acc, niter)
        writer.add_scalar("Test/ActionAccuracy", tt_act_acc, niter)
        writer.add_scalar("Test/IdentityAccuracy", tt_id_acc, niter)

        if epoch_ix % 1 == 0:
            state_dict = {
                'encoder': model.enc.state_dict(),
                'attr_net': model.attr_net.state_dict(),
            }
            torch.save(state_dict, os.path.join(args.exp, 'checkpoint', 'classifier_{}.pth'.format(epoch_ix)))