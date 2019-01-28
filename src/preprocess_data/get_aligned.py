import os
import sys
import pickle
import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='../../data/')
parser.add_argument('--out_root', type=str, default='../../data/weizmann_frames_aligned_by_video')
args = parser.parse_args()

def crop_and_aligned(img, bbox):
    # crop with bbox
    # resize to 64x64
    ymin, xmin, ymax, xmax = bbox
    h = ymax - ymin
    w = xmax - xmin

    max_length = max(h, w)
    pad_width = (max_length // 2)
    center_x, center_y = (xmin + xmax)//2 + 1 + pad_width, (ymin + ymax)//2 + 1 + pad_width # because we pad

    img_pad = np.pad(img, ((pad_width, pad_width), (pad_width, pad_width), (0, 0)), mode='mean')

    y_start, y_end = center_y - max_length//2, center_y + max_length//2
    x_start, x_end = center_x - max_length//2, center_x + max_length//2

    img_crop = img_pad[y_start:y_end, x_start:x_end]
    img_resize = cv2.resize(img_crop, (64, 64))

    return img_resize

def make_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

if __name__ == "__main__":
    make_dir(args.out_root)

    # frames
    frames_dir = os.path.join(args.data_root, 'weizmann_frames')
    actions = os.listdir(frames_dir)

    # bbox_dict
    bbox_dict = pickle.load(open('bboxes.pkl', 'rb'))

    for act in actions:
        frames = os.listdir(os.path.join(frames_dir, act))

        for f in frames:
            # bbox
            frame_basename, ext = os.path.splitext(f)

            # identity, action. per video
            # make dir by video
            identity, act_v, no_frame = frame_basename.split('_')
            make_dir(os.path.join(args.out_root, identity, act_v)) 

            try:
                bbox = bbox_dict[frame_basename]
            except Exception as e:
                # no bbox
                print(e)
                continue

            # read image
            frame_path = os.path.join(frames_dir, act, f)
            img = cv2.imread(frame_path)

            img_aligned = crop_and_aligned(img, bbox)
            cv2.imwrite(os.path.join(args.out_root, identity, act_v, '{}.jpg'.format(no_frame)), img_aligned)

