import os
import sys
import cv2
import argparse
import scipy.io

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='../../data/classification')
parser.add_argument('--out_root', type=str, default='../../data/weizmann_frames')
parser.add_argument('--batch_size', type=int, default=4)

args = parser.parse_args()

def make_dirs(d):
    if not os.path.exists(d):
        os.makedirs(d)

if __name__ == "__main__":
    # make dirs
    make_dirs(args.out_root)

    # params
    batch_size = args.batch_size
    data_root = args.data_root

    avi_dirs = [d for d in os.listdir(args.data_root) if os.path.isdir(os.path.join(args.data_root, d))]

    # loop through each action
    for act in avi_dirs:
        # each action contains 9 videos
        videos = sorted(os.listdir(os.path.join(args.data_root, act)))

        # where to store images
        make_dirs(os.path.join(args.out_root, act))
        for v in videos:
            v_name, _ = os.path.splitext(v)
            try:
                identity, act_v = v_name.split('_')
            except:
                # should skip folder: background
                continue

            cap = cv2.VideoCapture(os.path.join(args.data_root, act, v))
            count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                
                if ret == True:
                    #cv2.imshow('window-name', frame)
                    cv2.imwrite(os.path.join(args.out_root, act, "{}_{:03d}.jpg".format(v_name, count)), frame)
                    count += 1
                else:
                    break

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

    cap.release()
    cv2.destroyAllWindows()  # destroy all the opened windows