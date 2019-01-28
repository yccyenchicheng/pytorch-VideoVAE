import os
import sys
import time
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.utils as vutils

from src.datasets import WeizmannActionSequenceDataset
from src.synthesize import synthesize_test
from src.loss import VideoVAELoss
from src.model import VideoVAE

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--use_cuda', action='store_true',
                    help='whether to use gpu.')
parser.add_argument('--seed', type=int, default=1,
                    help='Set seed for reproducible experiment.')
parser.add_argument('--cls_weight', type=str, default=None,
                    help="path to attribute cls's weight.")
parser.add_argument('--ckpt', type=str, default=None,
                    help="path to model and opt's state_dict")
parser.add_argument('--exp', type=str, default='exp{}'.format(time.strftime('%m%d')),
                    help='experiment directory for checkpoint, logs, tensorboard, ..., etc')

args = parser.parse_args()

class RunningAverageMeter(object):
    def __init__(self, momentum=0.9):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
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

if __name__ == '__main__':
    # exp dir
    args.exp = os.path.join('ExperimentVideoVAE', args.exp)
    make_dirs(args.exp)
    make_dirs(os.path.join(args.exp, 'checkpoint'))
    
    # logger
    logger = get_logger(logpath=os.path.join(args.exp, 'logs'), filepath=__file__)
    logger.info('=> Current time: {}'.format(time.strftime('%Y/%m/%d, %H:%M:%S')))
    logger.info(args)

    tboard_dir = os.path.join(args.exp, 'tboard')
    writer = SummaryWriter(log_dir=tboard_dir)
    args.logger = logger
    args.writer = writer

    # seed
    import torch.backends.cudnn as cudnn
    import random
    cudnn.benchmark = False
    cudnn.deterministic = True

    # get random seed
    if args.seed == 0:
        args.seed = random.randint(1, 10000)

    logger.info("=> seed: {}".format(args.seed))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # params
    epochs = 10000
    batch_size = args.batch_size
    test_batch_size = 1
    lr = args.lr
    seq_len = 10
    im_shape = (3, 64, 64)
    z_dim = 512
    h_dim = 512 # for LSTM hidden
    n_act = 10
    n_id = 9
    start_epoch = 0

    # data
    trfs = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    train_set = WeizmannActionSequenceDataset(root='data', transform=trfs)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)

    test_set = WeizmannActionSequenceDataset(root='data', train=False, transform=trfs)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=True)

    # for class:ix's change
    # this is for tensorboard
    args.ix_to_act = test_set.ix_to_act
    args.ix_to_id = test_set.ix_to_id

    # model
    model = VideoVAE(z_dim=z_dim, h_dim=h_dim, n_act=n_act, n_id=n_id)
    logger.info("=> Number of parameters: {}".format(count_parameters(model)))
    logger.info(model)

    # ckpt
    if args.ckpt is not None:
        states = torch.load(args.ckpt)
        logger.info("=> Loading model's weights from {}".format(args.ckpt))
        model.load_state_dict(states['model'])
        start_epoch = states['epoch']

    if args.cls_weight is not None:
        logger.info("=> Loading Classifier's weights (encoder + attr_net) from {}".format(args.cls_weight))
        model.load_cls_net(args.cls_weight)

    # criterion
    criterion = VideoVAELoss()

    if args.use_cuda:
        model.cuda()
        criterion.cuda()

    # opt
    opt = optim.Adam(model.parameters(), lr=lr)
    batch_timer = RunningAverageMeter()

    if args.ckpt is not None:
        opt.load_state_dict(states['optimizer'])

    for epoch_ix in range(start_epoch, epochs):
        
        model.train()
        end = time.time()
        for batch_ix, data in enumerate(train_loader):
            # reset loss and lstm_hidden
            h_prev, c_prev = model.reset(batch_size=batch_size)

            # for bookkeeping
            loss = 0
            loss_l1_total = 0
            loss_kl_total = 0

            # img_seq: (b, t, c, h, w)
            img_seq, act_label_seq, id_label_seq, acts, ids = data
            
            if args.use_cuda:
                img_seq = img_seq.cuda()
                act_label_seq = act_label_seq.cuda()
                id_label_seq = id_label_seq.cuda()
                h_prev = h_prev.cuda()
                c_prev = c_prev.cuda()

            with torch.no_grad():
                pred_act_seq, pred_id_seq = model.seq_cls(img_seq)

            # propagate *seq_len* timesteps
            for t in range(seq_len):
                x_t, act_t, id_t = img_seq[:, t, :, :, :], pred_act_seq[:, t], pred_id_seq[:, t]

                # foward pass
                recon_x_t, z_t, lstm_output, [h_t, c_t], [mu_p, logvar_p], [mu_dy, logvar_dy] = model(x_t, act_t, id_t, h_prev, c_prev)
                h_prev, c_prev = h_t, c_t

                # loss
                loss_all, loss_l1, loss_kl = criterion(recon_x_t, x_t, [mu_p, logvar_p], [mu_dy, logvar_dy])
                loss_l1_total += loss_l1 # bookkeeping
                loss_kl_total += loss_kl # bookkeeping

                loss += loss_all

            opt.zero_grad()
            loss.backward()
            opt.step()

            batch_timer.update(time.time() - end)
            end = time.time()

            if batch_ix % 10 == 0:
                niter = epoch_ix * len(train_loader) + batch_ix
                logger.info('Epoch: {} | [{}/{}] Time: {:.4f} ({:.4f}) | Loss: {:.4f} | L1: {:.4f} | KL: {:.4f}'.format(
                             epoch_ix, batch_ix, len(train_loader), batch_timer.val, batch_timer.avg,
                             loss.item(), loss_l1_total.item(), loss_kl_total.item()))
                writer.add_scalar('Train/Loss', loss.item(), niter)
                writer.add_scalar('Train/LossL1', loss_l1_total.item(), niter)
                writer.add_scalar('Train/LossKL', loss_kl_total.item(), niter)

                for name, param in model.named_parameters():
                    if param.requires_grad:
                        name_ = name.replace('.', '/')
                        writer.add_histogram(name_, param.clone().cpu().data, niter)
                
        # testing here (synthesize)
        model.eval()
        only_prior = True
        synthesize_test(epoch_ix, model, test_loader, args, only_prior=only_prior)
        only_prior = False
        synthesize_test(epoch_ix, model, test_loader, args, only_prior=only_prior)

        if epoch_ix % 3 == 0:
            states = {
                'epoch': epoch_ix+1,
                'model': model.state_dict(),
                'optimizer': opt.state_dict(),
            }
            torch.save(states, os.path.join(args.exp, 'checkpoint', 'video_vae_{}.pth'.format(epoch_ix)))