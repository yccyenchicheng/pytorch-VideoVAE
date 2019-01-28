import os
import sys
import time
import argparse
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader
sys.path.append('/home/charles/pytorch-VideoVAE')
from src.datasets import WeizmannActionSequenceDataset

# We handcrafted some transitions for testing.
seq_len = 15
bend_seq = [0] * 15
jack_jump_seq = [1] * 8 + [2] * 7
jump_skip_seq = [2] * 8 + [6] * 7
pjump_side_seq = [3] * 8 + [5] * 7
run_skip_seq = [4] * 8 + [6] * 7
side_walk_side_seq = [5] * 5 + [7] * 5 + [5] * 5
skip_walk_skip_seq = [6] * 5 + [7] * 5 + [6] * 5
walk_r_walk_seq = [7] * 5 + [4] * 5 + [7] * 5
wave1_jump_seq = [8] * 8 + [2] * 7
wave2_jack_seq = [9] * 8 + [1] * 7

transient_dict = {0: bend_seq, 1: jack_jump_seq, 2: jump_skip_seq, 3: pjump_side_seq,
                  4: run_skip_seq, 5: side_walk_side_seq, 6: skip_walk_skip_seq, 
                  7: walk_r_walk_seq, 8: wave1_jump_seq, 9: wave2_jack_seq}

def syn_first_frame(model, img_0, h_0, c_0, holistic_attr, args, only_prior=False):
    """Synthesize the first frame for the 2 settings. """
    # setting 1:
    if only_prior:
        img_1, h_1, c_1 = model.synthesize(None, h_0, c_0, holistic_attr, only_prior=only_prior, 
                                            is_first_frame=True)
        init_seq = [img_1]
    # setting 2:
    else:
        recon_img_0, h_1, c_1 = model.synthesize(img_0, h_0, c_0, holistic_attr, only_prior=only_prior,
                                                is_first_frame=True)
        init_seq = [img_0, recon_img_0]
        img_1 = recon_img_0

    return init_seq, img_1, h_1, c_1

def syn_sequence(model, seq_len, init_seq, img_prev, h_prev, c_prev, holistic_attr, act_seq, args, only_prior=False):
    for t in range(1, seq_len):
        pred_act, pred_id = model.pred_attr(img_prev)
        holistic_attr["action"] = act_seq[t] # change action here

        img_t, h_t, c_t = model.synthesize(img_prev, h_prev, c_prev, holistic_attr, only_prior=only_prior, is_first_frame=False)
        h_prev, c_prev = h_t, c_t

        img_prev = img_t
        img_display = img_prev.clone().detach()
        init_seq.append(img_display)

    gen_seq = torch.cat(init_seq, dim=0)
    return gen_seq

def synthesize_test(epoch_ix, model, loader, args, only_prior=True, write_out=False):
    model.eval()
    
    with torch.no_grad():
        
        # have 10 act, 9 id
        n_act, n_id = 10, 9
        attr_seen = np.zeros((n_act, n_id))

        for batch_ix, data in enumerate(loader):
            img_0, act_y0, id_y0, action, identity = data

            # want to log all act, id pair but only once
            act_np, id_np = act_y0.item(), id_y0.item()
            if attr_seen[act_np, id_np] == 0:
                attr_seen[act_np, id_np] = 1
            else:
                continue

            # synthesize until all (act, id) combination were seen.
            if np.sum(attr_seen) == (n_act * n_id):
                break

            if args.use_cuda:
                img_0 = img_0.cuda()
                act_y0 = act_y0.cuda()
                id_y0 = id_y0.cuda()

            # setting 1:
            if only_prior:
                h_0, c_0 = model.reset(batch_size=loader.batch_size, reset='random')
                h_0, c_0 = h_0.to(img_0), c_0.to(img_0)
                act_seq_trans = transient_dict[act_y0.item()]
                act_seq_fixed = [act_y0.item()] * len(act_seq_trans)
                
                holistic_attr_fixed = {"action": act_seq_fixed[0],
                                       "identity": id_y0, # just gives the current id
                                       "is_fixed": True,}

                holistic_attr_transient = {"action": act_seq_trans[0],
                                           "identity": id_y0, # just gives the current id
                                           "is_fixed": False,}
                
            # setting 2:
            else:                
                h_0, c_0 = model.reset(batch_size=loader.batch_size, reset='zeros')
                h_0, c_0 = h_0.to(img_0), c_0.to(img_0)

                pred_act, pred_id = model.pred_attr(img_0)
                act_seq_trans = transient_dict[pred_act.item()]
                act_seq_fixed = [pred_act.item()] * len(act_seq_trans)

                holistic_attr_fixed = {
                    "action": act_seq_fixed[0],
                    "identity": pred_id.item(),
                }
                
                holistic_attr_transient = {
                    "action": act_seq_trans[0],
                    "identity": pred_id.item(),
                }

            seq_len = len(act_seq_trans)
            # gen fixed sequence
            init_seq, img_1, h_1, c_1 = syn_first_frame(model, img_0, h_0, c_0, holistic_attr_fixed, args, only_prior=only_prior)
            gen_seq_fixed = syn_sequence(model, seq_len, init_seq, img_1, h_1, c_1, 
                                         holistic_attr_fixed, act_seq_fixed, args, only_prior=only_prior)

            # gen transient sequence
            init_seq, img_1, h_1, c_1 = syn_first_frame(model, img_0, h_0, c_0, holistic_attr_transient, args, only_prior=only_prior)
            gen_seq_trans = syn_sequence(model, seq_len, init_seq, img_1, h_1, c_1, 
                                         holistic_attr_transient, act_seq_trans, args, only_prior=only_prior)

            gen_seq_summary = torch.cat([gen_seq_fixed, gen_seq_trans], dim=0)
            nrow = gen_seq_summary.size(0) // 2
            gen_seq_summary = vutils.make_grid(gen_seq_summary, nrow=nrow, normalize=True, scale_each=True)
            niter = (epoch_ix+1)

            # write to log
            act_str = args.ix_to_act[act_np]
            id_str  = args.ix_to_id[id_np]

            if only_prior:
                args.writer.add_image("{}/{}/OnlyPrior".format(id_str, act_str), gen_seq_summary.clone().cpu().data, niter)
            else:
                args.writer.add_image("{}/{}/FixFirst".format(id_str, act_str), gen_seq_summary.clone().cpu().data, niter)
